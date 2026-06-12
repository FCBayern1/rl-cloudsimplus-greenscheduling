"""
Gated Transformer-XL RLModule implementations for Multi-Datacenter Green Scheduling.

This module provides RLModules that utilise the GTrXL architecture for both
Local Agents (with Action Masking) and the Global Agent.

Architecture:
- Input (Dict/Flat) -> Embedding -> GTrXL (Gated Transformer) -> Heads
"""

import math
from typing import Any, Dict, Optional, List
import logging
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

# 2026-05-13 perf: enable TF32 tensor-core path on Ampere/Ada/Blackwell GPUs.
# float32 matmul is the dominant op in GTrXL's MultiheadAttention; TF32 cuts
# the matmul cost by 1.5-3× on 5080's tensor cores at the price of ~0.001%
# numerical noise on the matmul output — fine for training, completely
# imperceptible to PPO loss / KL stats.  This is what torch's compile-time
# warning was nudging us to do.
try:
    torch.set_float32_matmul_precision("high")
except Exception:  # older torch versions don't expose this knob
    pass

# Suppress the noisy "Online softmax is disabled on the fly" warning that
# torch._inductor emits from inside the GTrXL forward.  It's an internal
# decision by the compiler about how to lower a single softmax — informative
# but not actionable from our side, and it spams the driver log once per
# (re)compile.  Squelch *just* this category-message pair so genuine torch
# warnings still surface.
warnings.filterwarnings(
    "ignore",
    message=r".*Online softmax is disabled on the fly.*",
    category=UserWarning,
)

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI, ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchMultiCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from src.networks.gtrxl import GTrXL

logger = logging.getLogger(__name__)

# NaN/Inf health checks call torch.isfinite(...).all() and .item(), each of which
# forces a CUDA sync and serialises the GPU pipeline.  Per minibatch they cost
# ~1-2s on GH200 due to in-flight transformer kernels, dominating PPO update
# time (observed 85min/iter for an 8000-sample batch).  Enable only when
# debugging numerical issues by exporting GTRXL_DEBUG_NAN_CHECKS=1.
_DEBUG_NAN_CHECKS = os.environ.get("GTRXL_DEBUG_NAN_CHECKS", "0") == "1"


def _parse_gtrxl_state_in(
    batch: Dict[str, Any],
    batch_size: int,
    num_layers: int,
    mem_len: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    state_key: str = "gtrxl_mem",
) -> Optional[List[torch.Tensor]]:
    """Build per-layer memory list from RLlib Columns.STATE_IN, or None to zero-init.

    state_key: which sub-key in STATE_IN to read. Defaults to "gtrxl_mem" so
    every pre-2026-05-19 caller keeps working unchanged. Route 2.5
    dual-trunk modules pass "gtrxl_mem_actor" / "gtrxl_mem_critic".
    """
    si = batch.get(Columns.STATE_IN)
    if not isinstance(si, dict) or si is None:
        return None
    raw = si.get(state_key)
    if raw is None:
        return None
    try:
        t = torch.as_tensor(raw, device=device, dtype=dtype)
        if t.dim() == 3:
            t = t.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        elif t.dim() == 4:
            if t.shape[0] == 1 and batch_size > 1:
                t = t.expand(batch_size, -1, -1, -1).contiguous()
            elif t.shape[0] != batch_size:
                return None
        else:
            return None
        if t.shape[1] != num_layers or t.shape[2] != mem_len or t.shape[3] != d_model:
            return None
        return [t[:, i].contiguous() for i in range(num_layers)]
    except Exception:
        return None


def _gtrxl_state_out(
    memories: List[torch.Tensor],
    state_key: str = "gtrxl_mem",
) -> Dict[str, torch.Tensor]:
    return {state_key: torch.stack(memories, dim=1)}

class GTrXLMaskedActionRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """
    GTrXL-based RLModule for Local Agents with Action Masking.
    
    Structure:2
    - Observation Embedding (Linear)
    - GTrXL Backbone (Gated Transformer)
    - Policy Head (Linear)
    - Value Head (Linear)
    """

    @override(TorchRLModule)
    def setup(self):
        # Configuration
        model_config = self.model_config
        obs_space = self.observation_space
        action_space = self.action_space
        
        # Dimensions
        self.obs_dim = self._get_obs_dim(obs_space)
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        else:
            raise ValueError("Local Agents require Discrete action space")

        # Transformer Config
        d_model = model_config.get("d_model", 128)
        nhead = model_config.get("nhead", 4)
        num_layers = model_config.get("num_layers", 2)
        dim_feedforward = model_config.get("dim_feedforward", 256)
        dropout = model_config.get("dropout", 0.0)
        mem_len = int(model_config.get("mem_len", 16))
        max_seq_len = int(model_config.get("max_seq_len", 128))
        max_seq_len = max(max_seq_len, mem_len + 32)

        # Build GTrXL Backbone (Transformer-XL-style memory between env steps)
        self.gtrxl = GTrXL(
            input_dim=self.obs_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            mem_len=mem_len,
            max_seq_len=max_seq_len,
        )

        # 2026-05-13 Level D perf — REVERTED 2026-05-14 after smoke
        # 20260514_181716 showed PPO update went 12 min → 35 min instead of
        # speeding up.  Root cause: PPO's RLlib minibatch sampler produces
        # several distinct shapes per epoch (full mb 2048, partial mbs of
        # 128 and 1856 from the 80000/8000 sample buckets), and Inductor
        # recompiles for every new shape despite dynamic=True — 18+ recompile
        # events × ~90 sec each ≈ 25 min of pure compile cost.  Skipping
        # torch.compile entirely is cleaner than fighting Inductor's
        # recompile heuristics; Level B (sgd_minibatch_size 2048) gives us
        # most of the win anyway by cutting Python dispatch count 4×.
        # To re-enable for an experiment with fixed shapes, set
        # model_config["compile"] = True.
        if bool(model_config.get("compile", False)):
            try:
                self.gtrxl = torch.compile(self.gtrxl, mode="default", dynamic=True)
                logger.info(f"[{self.__class__.__name__}] torch.compile(GTrXL) enabled")
            except Exception as e:
                logger.warning(
                    f"[{self.__class__.__name__}] torch.compile failed (%s); falling back to eager",
                    e,
                )

        # Heads
        self.policy_head = nn.Linear(d_model, self.action_dim)
        self.value_head = nn.Linear(d_model, 1)

        # State management
        self._last_value = None
        self._debug_dumped = False  # For one-time debug dump

        # === DEBUG: Log computed obs_dim ===
        logger.info(f"[{self.__class__.__name__}] obs_space={self.observation_space}")
        logger.info(f"[{self.__class__.__name__}] computed obs_dim={self.obs_dim}")
        logger.info(f"[{self.__class__.__name__}] action_dim={self.action_dim}")
        logger.info(
            f"[{self.__class__.__name__}] d_model={d_model}, nhead={nhead}, "
            f"num_layers={num_layers}, mem_len={mem_len}"
        )

    @override(TorchRLModule)
    def get_initial_state(self):
        """Per-layer XL memory template (B dimension added by RLlib connectors)."""
        return {
            "gtrxl_mem": np.zeros(
                (self.gtrxl.num_layers, self.gtrxl.mem_len, self.gtrxl.d_model),
                dtype=np.float32,
            )
        }

    def _get_obs_dim(self, obs_space) -> int:
        """Calculate flat observation dimension (excluding action mask)."""
        import numpy as np
        if isinstance(obs_space, spaces.Box):
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, spaces.Dict):
            if "observation" in obs_space.spaces:
                return self._get_obs_dim(obs_space.spaces["observation"])
            total = 0
            for key, space in obs_space.spaces.items():
                total += self._get_obs_dim(space)
            return total
        elif isinstance(obs_space, spaces.Discrete):
            return 1
        return int(np.prod(obs_space.shape))

    def _flatten_obs(self, obs: Dict[str, TensorType]) -> TensorType:
        """Flatten Dict observation to a single tensor, handling mixed dims."""
        if isinstance(obs, torch.Tensor):
            return obs

        tensors = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, dict):
                val = self._flatten_obs(val)
            if isinstance(val, torch.Tensor):
                tensors.append(val.float())
        
        if not tensors:
            raise ValueError("No tensors found in observation")

        # Normalize dimensions
        max_dim = max(t.dim() for t in tensors)
        
        # Determine max sequence length if we have time dimension
        max_len = 1
        if max_dim == 3:
            max_len = max(t.shape[1] for t in tensors if t.dim() == 3)

        final_tensors = []
        for t in tensors:
            if t.dim() < max_dim:
                # Case: Mixed (Batch, Time) single-feature and (Batch, Time, Feat) multi-feature
                if max_dim == 3 and t.dim() == 2:
                    # 2D tensor is (Batch, Time) for single-feature obs (like Discrete)
                    # Add feature dim at the end: (Batch, Time) -> (Batch, Time, 1)
                    t = t.unsqueeze(-1)
                elif max_dim == 3 and t.dim() == 1:
                    t = t.unsqueeze(1).unsqueeze(2)
                elif max_dim == 2 and t.dim() == 1:
                    t = t.unsqueeze(1)
                
            final_tensors.append(t)
            
        return torch.cat(final_tensors, dim=-1)

    def _extract_obs_and_mask(self, batch: Dict[str, Any]) -> tuple:
        """Extract obs and mask, handling time dimension."""
        obs = batch.get(Columns.OBS, batch.get("obs", {}))
        
        # Handle Dict Obs
        if isinstance(obs, dict):
            if "observation" in obs:
                true_obs = obs["observation"]
                action_mask = obs.get("action_mask", None)
            else:
                true_obs = obs
                action_mask = None
        else:
            true_obs = obs
            action_mask = None

        flat_obs = self._flatten_obs(true_obs)
        
        # Convert mask
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, dtype=torch.float32)
        
        return flat_obs, action_mask

    def _forward_pass(self, batch: Dict[str, Any], state_in: Any = None):
        """Shared forward pass logic. Returns logits, values, STATE_OUT content."""
        flat_obs, action_mask = self._extract_obs_and_mask(batch)

        # === DEBUG: One-time dump of batch structure ===
        if not self._debug_dumped:
            self._debug_dumped = True
            logger.debug("=" * 70)
            logger.debug(f"=== [{self.__class__.__name__}] DEBUG DUMP (first batch) ===")
            logger.debug("=" * 70)
            logger.debug(f"self.obs_dim (from space) = {self.obs_dim}")
            logger.debug(f"flat_obs.shape = {tuple(flat_obs.shape)}")
            logger.debug(f"flat_obs.dtype = {flat_obs.dtype}")

            obs = batch.get(Columns.OBS, batch.get("obs", {}))
            logger.debug(f"raw obs type = {type(obs)}")
            if isinstance(obs, dict):
                logger.debug(f"raw obs keys = {list(obs.keys())}")

            # Print each key's tensor shape
            def dump_shapes(prefix, o):
                if isinstance(o, dict):
                    for k in sorted(o.keys()):
                        dump_shapes(f"{prefix}.{k}", o[k])
                elif isinstance(o, torch.Tensor):
                    logger.debug(f"{prefix}: tensor shape={tuple(o.shape)} dtype={o.dtype}")
                else:
                    try:
                        import numpy as np
                        arr = np.asarray(o)
                        logger.debug(f"{prefix}: array shape={arr.shape} dtype={arr.dtype}")
                    except Exception:
                        logger.debug(f"{prefix}: {type(o)}")

            if isinstance(obs, dict):
                dump_shapes("obs", obs)

            # Check if it looks like flattened sequence
            feat_dim = flat_obs.shape[-1]
            if feat_dim != self.obs_dim:
                logger.warning(f"DIMENSION MISMATCH: model expects {self.obs_dim}, got {feat_dim}")
                if feat_dim % self.obs_dim == 0 and feat_dim > self.obs_dim:
                    T = feat_dim // self.obs_dim
                    logger.warning(f"Looks like flattened sequence: {feat_dim} = {self.obs_dim} * T({T})")

            logger.debug("=" * 70)
            logger.debug("=== END DEBUG DUMP ===")
            logger.debug("=" * 70)

        # === FAIL FAST: Check feature dimension ===
        feat_dim = flat_obs.shape[-1]
        if feat_dim != self.obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Obs dim mismatch! "
                f"Model expects {self.obs_dim} (from observation_space), "
                f"but got {feat_dim} from batch flatten. "
                f"Likely cause: env output != space declaration, OR sequence flattened into features."
            )

        if _DEBUG_NAN_CHECKS:
            # === CHECKPOINT 1: Check if obs has NaN/Inf ===
            if not torch.isfinite(flat_obs).all():
                bad_ratio = (~torch.isfinite(flat_obs)).float().mean().item()
                logger.error(f"[{self.__class__.__name__}] flat_obs has non-finite values! ratio={bad_ratio:.4f}")
                # Log which positions have issues
                bad_mask = ~torch.isfinite(flat_obs)
                bad_indices = bad_mask.nonzero()[:10]  # First 10 bad positions
                logger.error(f"First bad positions: {bad_indices.tolist()}")
                raise ValueError(f"Non-finite values in flat_obs (ratio={bad_ratio:.4f})")

            # === CHECKPOINT 2: Check action_mask health ===
            if action_mask is not None:
                if not torch.isfinite(action_mask).all():
                    bad_ratio = (~torch.isfinite(action_mask)).float().mean().item()
                    logger.error(f"[{self.__class__.__name__}] action_mask has non-finite values! ratio={bad_ratio:.4f}")
                    raise ValueError(f"Non-finite values in action_mask (ratio={bad_ratio:.4f})")

                # Check for rows with no valid actions
                valid_cnt = (action_mask >= 0.5).sum(dim=-1)  # (B, T) or (B*T,)
                zero_rows = (valid_cnt == 0).sum().item()
                if zero_rows > 0:
                    total_rows = valid_cnt.numel()
                    logger.warning(f"[{self.__class__.__name__}] Found {zero_rows}/{total_rows} rows with NO valid actions!")

        # Handling Dimensions for Transformer
        # RLlib connectors usually provide (Batch, Time, Feat) if state is present
        # Or (Batch*Time, Feat) if flattened.
        
        obs_shape = flat_obs.shape
        if len(obs_shape) == 2:
            # (B*T, F) -> Reshape to (B, 1, F) if we assume T=1 or infer B
            # However, simpler to treat as (Batch, Sequence=1, Feature)
            # if we don't have explicit sequence info easily available.
            # But GTrXL expects (B, T, F).
            
            # If we are in training, we might have seq_lens.
            seq_lens = batch.get("seq_lens")
            if seq_lens is not None and torch.sum(seq_lens) == obs_shape[0]:
                 # Dynamic unpacking (complex), usually handled by RLlib's 'get_batch_size'
                 # Simplification: Assume (B, 1, F) for now if 2D
                 flat_obs = flat_obs.unsqueeze(1)
            else:
                 # Default to adding time dim 1
                 flat_obs = flat_obs.unsqueeze(1)

        B = flat_obs.shape[0]
        memories_in = _parse_gtrxl_state_in(
            batch,
            B,
            self.gtrxl.num_layers,
            self.gtrxl.mem_len,
            self.gtrxl.d_model,
            flat_obs.device,
            flat_obs.dtype,
        )
        features, memories_out = self.gtrxl(flat_obs, state=memories_in)
        state_out = _gtrxl_state_out(memories_out)

        if _DEBUG_NAN_CHECKS:
            # === CHECKPOINT 3: Check if GTrXL features have NaN/Inf ===
            if not torch.isfinite(features).all():
                bad_ratio = (~torch.isfinite(features)).float().mean().item()
                logger.error(f"[{self.__class__.__name__}] GTrXL features has non-finite values! ratio={bad_ratio:.4f}")
                raise ValueError(f"Non-finite values in GTrXL features (ratio={bad_ratio:.4f})")

        # Features: (B, T, d_model)

        # Full-sequence logits and values (preserve time dim — slicing to the
        # current step is a per-method concern, not a forward-pass concern.
        # Collapsing the time dim here breaks compute_values during GAE,
        # which expects (B, T) values for a (B, T, F) batch.
        logits = self.policy_head(features)  # (B, T, A)
        values = self.value_head(features).squeeze(-1)  # (B, T)

        return logits, values, state_out, action_mask

    def _apply_action_mask(
        self, logits: torch.Tensor, action_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply action masking to logits.  Handles both (B, A) and (B, T, A)."""
        if action_mask is None:
            return logits

        action_mask = torch.nan_to_num(action_mask, nan=0.0, posinf=0.0, neginf=0.0)

        if logits.dim() == 3 and action_mask.dim() == 2:
            action_mask = action_mask.unsqueeze(1).expand_as(logits)

        valid = action_mask >= 0.5
        valid_cnt = valid.sum(dim=-1, keepdim=True)
        no_valid = valid_cnt == 0
        if no_valid.any():
            valid = valid.clone()
            valid[..., 0] = valid[..., 0] | no_valid.squeeze(-1)

        return torch.where(valid, logits, torch.full_like(logits, -1e9))

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, values, state_out, action_mask = self._forward_pass(batch)
        logits = self._apply_action_mask(logits, action_mask)
        self._last_value = values
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
            # 2026-06-12 dead-critic fix: hand the GRAD-CARRYING values to the
            # learner. PPOTorchLearner builds the vf loss from
            # compute_values(batch, embeddings=fwd_out.get(EMBEDDINGS)); our
            # compute_values returns these directly, instead of re-running the
            # trunk under inference_mode (which silently zeroed the critic
            # gradient from 2026-05-12 onward). Zero extra memory — this
            # tensor's graph already exists for the policy loss.
            Columns.EMBEDDINGS: values,
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, _, state_out, action_mask = self._forward_pass(batch)
        # GTrXL uses history to predict the current step → take the last ts.
        logits = logits[:, -1, :]  # (B, A)
        if action_mask is not None and action_mask.dim() == 3:
            action_mask = action_mask[:, -1, :]
        logits = self._apply_action_mask(logits, action_mask)
        # Stateful modules must preserve the leading T=1 dim that RLlib's
        # AddSingleTsTimeRankToBatch connector inserted, so the matching
        # RemoveSingleTsTimeRankFromBatch connector can squeeze it back out.
        return {
            Columns.ACTION_DIST_INPUTS: logits.unsqueeze(1),
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, values, state_out, action_mask = self._forward_pass(batch)
        logits = logits[:, -1, :]  # (B, A)
        values = values[:, -1]     # (B,)
        if action_mask is not None and action_mask.dim() == 3:
            action_mask = action_mask[:, -1, :]
        logits = self._apply_action_mask(logits, action_mask)
        self._last_value = values
        return {
            Columns.ACTION_DIST_INPUTS: logits.unsqueeze(1),
            Columns.VF_PREDS: values.unsqueeze(1),
            Columns.STATE_OUT: state_out,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        # 2026-06-12 dead-critic fix: `embeddings` is the grad-carrying value
        # tensor `_forward_train` emitted under Columns.EMBEDDINGS. The PPO
        # learner passes it back here when building the vf loss — return it
        # as-is so the critic actually receives gradient. (From 2026-05-12 to
        # 2026-06-12 this method unconditionally re-ran the trunk under
        # inference_mode, so the vf loss was a CONSTANT w.r.t. parameters and
        # both critics trained on exactly zero gradient.)
        if embeddings is not None:
            return embeddings
        # 2026-05-12 OOM fix (GAE path — keep): GAE
        # (rllib/connectors/learner/general_advantage_estimation.py:96) calls
        # compute_values WITHOUT embeddings on the FULL post-rollout batch —
        # for our 10-DC v2 setup that's 8000 env steps × 10 shared-local
        # agents = 80 000 samples in a single forward. GTrXL.forward unrolls
        # T timesteps in a Python loop; without no_grad the autograd graph
        # from every intermediate tensor at every layer at every t is
        # retained, which alone consumed 10+ GB of VRAM ("16GB 5080 OOMs on
        # bumping d_model 96 → 128"). GAE only needs V(s) values, not
        # gradients, so the no-embeddings path skips graph construction.
        with torch.inference_mode():
            _, values, _, _ = self._forward_pass(batch)
        # inference_mode returns tensors that are read-only and don't have an `.grad_fn`.
        # That's fine for GAE — the downstream consumer (advantage calc) just reads values.
        # But to be safe against any future caller that expects a regular tensor, clone
        # into a normal one (loses inference_mode attribute, keeps gradient-detached state).
        return values.clone()

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return TorchCategorical

    def get_non_inference_attributes(self):
        """
        Return attributes that are not needed for inference.
        """
        return ["value_head", "_last_value"]


class GTrXLGlobalRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """
    GTrXL-based RLModule for Global Agent (Dict Observation, No Mask).
    """
    
    @override(TorchRLModule)
    def setup(self):
        model_config = self.model_config
        obs_space = self.observation_space
        action_space = self.action_space
        
        self.obs_dim = self._get_obs_dim(obs_space)
        
        # Action Space handling
        if isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = int(sum(action_space.nvec))
            self.action_dist_cls = self._get_multi_categorical_cls(action_space)
        elif isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self.action_dist_cls = TorchCategorical
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

        # Config
        d_model = model_config.get("d_model", 128)
        nhead = model_config.get("nhead", 4)
        num_layers = model_config.get("num_layers", 2)
        dim_feedforward = model_config.get("dim_feedforward", 256)
        dropout = model_config.get("dropout", 0.0)
        mem_len = int(model_config.get("mem_len", 16))
        max_seq_len = int(model_config.get("max_seq_len", 128))
        max_seq_len = max(max_seq_len, mem_len + 32)

        self.gtrxl = GTrXL(
            input_dim=self.obs_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            mem_len=mem_len,
            max_seq_len=max_seq_len,
        )

        # 2026-05-13 Level D perf — REVERTED, see Local module above for rationale.
        # Default OFF.  Opt in by setting model_config["compile"] = True.
        if bool(model_config.get("compile", False)):
            try:
                self.gtrxl = torch.compile(self.gtrxl, mode="default", dynamic=True)
                logger.info(f"[{self.__class__.__name__}] torch.compile(GTrXL) enabled")
            except Exception as e:
                logger.warning(
                    f"[{self.__class__.__name__}] torch.compile failed (%s); falling back to eager",
                    e,
                )

        self.policy_head = nn.Linear(d_model, self.action_dim)
        self.value_head = nn.Linear(d_model, 1)
        self._last_value = None
        self._debug_dumped = False  # For one-time debug dump

        # === DEBUG: Log computed obs_dim ===
        logger.info(f"[{self.__class__.__name__}] obs_space={self.observation_space}")
        logger.info(f"[{self.__class__.__name__}] computed obs_dim={self.obs_dim}")
        logger.info(f"[{self.__class__.__name__}] action_dim={self.action_dim}")
        logger.info(
            f"[{self.__class__.__name__}] d_model={d_model}, nhead={nhead}, "
            f"num_layers={num_layers}, mem_len={mem_len}"
        )

    @override(TorchRLModule)
    def get_initial_state(self):
        return {
            "gtrxl_mem": np.zeros(
                (self.gtrxl.num_layers, self.gtrxl.mem_len, self.gtrxl.d_model),
                dtype=np.float32,
            )
        }

    def _get_obs_dim(self, obs_space) -> int:
        import numpy as np
        if isinstance(obs_space, spaces.Box):
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, spaces.Dict):
            if "observation" in obs_space.spaces:
                return self._get_obs_dim(obs_space.spaces["observation"])
            total = 0
            for key, space in obs_space.spaces.items():
                total += self._get_obs_dim(space)
            return total
        elif isinstance(obs_space, spaces.Discrete):
            return 1
        elif isinstance(obs_space, spaces.MultiDiscrete):
            return len(obs_space.nvec)
        return 1

    def _flatten_obs(self, obs: Dict[str, TensorType]) -> TensorType:
        """Flatten Dict observation to a single tensor, handling mixed dims."""
        if isinstance(obs, torch.Tensor):
            return obs

        tensors = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, dict):
                val = self._flatten_obs(val)
            if isinstance(val, torch.Tensor):
                tensors.append(val.float())
        
        if not tensors:
            raise ValueError("No tensors found in observation")

        # Normalize dimensions
        max_dim = max(t.dim() for t in tensors)
        
        # Determine max sequence length if we have time dimension
        max_len = 1
        if max_dim == 3:
            max_len = max(t.shape[1] for t in tensors if t.dim() == 3)

        final_tensors = []
        for t in tensors:
            if t.dim() < max_dim:
                # Case: Mixed (Batch, Time) single-feature and (Batch, Time, Feat) multi-feature
                if max_dim == 3 and t.dim() == 2:
                    # 2D tensor is (Batch, Time) for single-feature obs (like Discrete)
                    # Add feature dim at the end: (Batch, Time) -> (Batch, Time, 1)
                    t = t.unsqueeze(-1)
                elif max_dim == 3 and t.dim() == 1:
                    t = t.unsqueeze(1).unsqueeze(2)
                elif max_dim == 2 and t.dim() == 1:
                    t = t.unsqueeze(1)
                
            final_tensors.append(t)
            
        return torch.cat(final_tensors, dim=-1)

    def _forward_pass(self, batch: Dict[str, Any], state_in: Any = None):
        obs = batch.get(Columns.OBS, batch.get("obs", {}))
        # Unwrap "observation" key if present (Connector wrapper)
        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        flat_obs = self._flatten_obs(obs)

        # === DEBUG: One-time dump of batch structure ===
        if not self._debug_dumped:
            self._debug_dumped = True
            logger.debug("=" * 70)
            logger.debug(f"=== [{self.__class__.__name__}] DEBUG DUMP (first batch) ===")
            logger.debug("=" * 70)
            logger.debug(f"self.obs_dim (from space) = {self.obs_dim}")
            logger.debug(f"flat_obs.shape = {tuple(flat_obs.shape)}")
            logger.debug(f"flat_obs.dtype = {flat_obs.dtype}")

            raw_obs = batch.get(Columns.OBS, batch.get("obs", {}))
            logger.debug(f"raw obs type = {type(raw_obs)}")
            if isinstance(raw_obs, dict):
                logger.debug(f"raw obs keys = {list(raw_obs.keys())}")

            # Print each key's tensor shape
            def dump_shapes(prefix, o):
                if isinstance(o, dict):
                    for k in sorted(o.keys()):
                        dump_shapes(f"{prefix}.{k}", o[k])
                elif isinstance(o, torch.Tensor):
                    logger.debug(f"{prefix}: tensor shape={tuple(o.shape)} dtype={o.dtype}")
                else:
                    try:
                        import numpy as np
                        arr = np.asarray(o)
                        logger.debug(f"{prefix}: array shape={arr.shape} dtype={arr.dtype}")
                    except Exception:
                        logger.debug(f"{prefix}: {type(o)}")

            if isinstance(raw_obs, dict):
                dump_shapes("obs", raw_obs)

            # Check if it looks like flattened sequence
            feat_dim = flat_obs.shape[-1]
            if feat_dim != self.obs_dim:
                logger.warning(f"DIMENSION MISMATCH: model expects {self.obs_dim}, got {feat_dim}")
                if feat_dim % self.obs_dim == 0 and feat_dim > self.obs_dim:
                    T = feat_dim // self.obs_dim
                    logger.warning(f"Looks like flattened sequence: {feat_dim} = {self.obs_dim} * T({T})")

            logger.debug("=" * 70)
            logger.debug("=== END DEBUG DUMP ===")
            logger.debug("=" * 70)

        # === FAIL FAST: Check feature dimension ===
        feat_dim = flat_obs.shape[-1]
        if feat_dim != self.obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Obs dim mismatch! "
                f"Model expects {self.obs_dim} (from observation_space), "
                f"but got {feat_dim} from batch flatten. "
                f"Likely cause: env output != space declaration, OR sequence flattened into features."
            )

        if _DEBUG_NAN_CHECKS:
            # === CHECKPOINT 1: Check if obs has NaN/Inf ===
            if not torch.isfinite(flat_obs).all():
                bad_ratio = (~torch.isfinite(flat_obs)).float().mean().item()
                logger.error(f"[{self.__class__.__name__}] flat_obs has non-finite values! ratio={bad_ratio:.4f}")
                bad_mask = ~torch.isfinite(flat_obs)
                bad_indices = bad_mask.nonzero()[:10]
                logger.error(f"First bad positions: {bad_indices.tolist()}")
                raise ValueError(f"Non-finite values in flat_obs (ratio={bad_ratio:.4f})")

        # Reshape for Transformer (B, 1, F) if flat
        if flat_obs.dim() == 2:
            flat_obs = flat_obs.unsqueeze(1)

        B = flat_obs.shape[0]
        memories_in = _parse_gtrxl_state_in(
            batch,
            B,
            self.gtrxl.num_layers,
            self.gtrxl.mem_len,
            self.gtrxl.d_model,
            flat_obs.device,
            flat_obs.dtype,
        )
        features, memories_out = self.gtrxl(flat_obs, state=memories_in)
        state_out = _gtrxl_state_out(memories_out)

        if _DEBUG_NAN_CHECKS:
            # === CHECKPOINT 3: Check if GTrXL features have NaN/Inf ===
            if not torch.isfinite(features).all():
                bad_ratio = (~torch.isfinite(features)).float().mean().item()
                logger.error(f"[{self.__class__.__name__}] GTrXL features has non-finite values! ratio={bad_ratio:.4f}")
                raise ValueError(f"Non-finite values in GTrXL features (ratio={bad_ratio:.4f})")

        # Full-sequence logits and values (slicing happens per-method to keep
        # compute_values working correctly for (B, T, F) batches during GAE).
        logits = self.policy_head(features)  # (B, T, A)
        values = self.value_head(features).squeeze(-1)  # (B, T)

        return logits, values, state_out

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, values, state_out = self._forward_pass(batch)
        self._last_value = values
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
            # 2026-06-12 dead-critic fix — see GTrXLMaskedActionRLModule.
            Columns.EMBEDDINGS: values,
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, _, state_out = self._forward_pass(batch)
        logits = logits[:, -1, :]  # (B, A)
        return {
            Columns.ACTION_DIST_INPUTS: logits.unsqueeze(1),
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, values, state_out = self._forward_pass(batch)
        logits = logits[:, -1, :]  # (B, A)
        values = values[:, -1]     # (B,)
        self._last_value = values
        return {
            Columns.ACTION_DIST_INPUTS: logits.unsqueeze(1),
            Columns.VF_PREDS: values.unsqueeze(1),
            Columns.STATE_OUT: state_out,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        # 2026-06-12 dead-critic fix — grad-carrying values from
        # _forward_train (learner loss path); see the Local module above.
        if embeddings is not None:
            return embeddings
        # See OOM-fix note on the Local module's compute_values above —
        # GAE forwards the full batch through GTrXL; without no_grad the
        # autograd graph from the per-T Python loop blows up VRAM.
        with torch.inference_mode():
            _, values, _ = self._forward_pass(batch)
        return values.clone()

    def _get_multi_categorical_cls(self, action_space):
        input_lens = list(action_space.nvec)
        class BoundMultiCategorical(TorchMultiCategorical):
            @staticmethod
            def from_logits(logits, **kwargs):
                return TorchMultiCategorical.from_logits(logits, input_lens=input_lens, **kwargs)
        return BoundMultiCategorical

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return self.action_dist_cls

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return self.action_dist_cls
        
    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return self.action_dist_cls

    def get_non_inference_attributes(self):
        """
        Return attributes that are not needed for inference.
        """
        return ["value_head", "_last_value"]


class GTrXLScoreBasedGlobalRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """
    Score-based Global RLModule for hierarchical multi-DC scheduling.

    Replaces 10 independent per-slot heads with a **structured pairwise score**
    function:

        logits[B, T, i, d] = <q_i, k_d> / sqrt(D)

    where
        q_i = cloudlet_emb[i] + ctx_to_cloudlet(context_feat)
        k_d = dc_emb[d]       + ctx_to_dc(context_feat)

    The per-cloudlet embedding and per-DC embedding share weights across
    cloudlets/DCs respectively, so the score is permutation-equivariant in
    both axes.  Combined with the softmax-per-cloudlet structure exposed via
    MultiDiscrete, this collapses an effective 10^10 joint action space into
    10 independent N_dc-way softmaxes — same API as the old module, but
    massively reduced sample complexity (10^5-10^8x).

    Why: old GTrXLGlobalRLModule had `policy_head = Linear(d, N_batch*N_dc)`,
    so each (slot, DC) entry got an independent column.  The agent could
    not generalize "DC 3 is overloaded" from one slot to another.  Score
    function shares the DC features across all slots → reading the
    green/load state of DC d helps every routing decision.
    """

    @override(TorchRLModule)
    def setup(self):
        model_config = self.model_config
        obs_space = self.observation_space
        action_space = self.action_space

        # === Action space ===
        if not isinstance(action_space, spaces.MultiDiscrete):
            raise ValueError(
                "GTrXLScoreBasedGlobalRLModule requires MultiDiscrete action "
                f"space; got {type(action_space)}"
            )
        nvec = [int(x) for x in action_space.nvec]
        self.num_dcs = nvec[0]
        self.num_batch_slots = len(nvec)
        if not all(n == self.num_dcs for n in nvec):
            raise ValueError(
                "Score-based module assumes all MultiDiscrete components are "
                f"equal (uniform per-slot DC choice); got nvec={nvec}"
            )
        self.action_dim = int(sum(nvec))
        self.action_dist_cls = self._get_multi_categorical_cls(action_space)

        # === Categorize obs keys ===
        inner = obs_space
        if isinstance(obs_space, spaces.Dict) and "observation" in obs_space.spaces:
            inner = obs_space.spaces["observation"]
        if not isinstance(inner, spaces.Dict):
            raise ValueError(
                "Score-based module expects a Dict observation; got "
                f"{type(inner)}"
            )
        self._categorize_keys(inner)

        # === Config ===
        d_model = int(model_config.get("d_model", 128))
        nhead = int(model_config.get("nhead", 4))
        num_layers = int(model_config.get("num_layers", 2))
        dim_feedforward = int(model_config.get("dim_feedforward", 256))
        dropout = float(model_config.get("dropout", 0.0))
        mem_len = int(model_config.get("mem_len", 16))
        max_seq_len = int(model_config.get("max_seq_len", 128))
        max_seq_len = max(max_seq_len, mem_len + 32)
        self.d_model = d_model

        # === GTrXL on global context ===
        self.gtrxl = GTrXL(
            input_dim=max(1, self.context_dim),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            mem_len=mem_len,
            max_seq_len=max_seq_len,
        )

        if bool(model_config.get("compile", False)):
            try:
                self.gtrxl = torch.compile(self.gtrxl, mode="default", dynamic=True)
                logger.info(f"[{self.__class__.__name__}] torch.compile(GTrXL) enabled")
            except Exception as e:
                logger.warning(
                    f"[{self.__class__.__name__}] torch.compile failed (%s); falling back to eager",
                    e,
                )

        # === Input normalization (2026-05-17 first-smoke fix) ===
        # First Stage 3 smoke (logs/.../20260517_012131): after iter 1 PPO update
        # we got global_entropy=0.348 (down from ~22), global_mean_kl=inf,
        # global_grad_norm=255.
        # Two compounding causes:
        #   (a) raw obs scale: batch_cloudlet_mi is up to 2e6 while pes is up to
        #       100.  cloudlet_encoder (Linear(2, D)) outputs ~|mi|·||W|| which
        #       totally dominates the embedding direction → all slots see the
        #       same "this cloudlet is big" feature → all logits move together.
        #   (b) 10 routing slots share dc_encoder + ctx_to_dc, so a PPO step
        #       that pushes one DC's prob down receives ~10× the gradient on
        #       the shared params → step too large → KL blows up.
        # We address (a) here with input scale buffers built from obs_space.high
        # (Box `high` of each feature); the inputs get divided by their high so
        # everything lives in roughly [-1, 1] before hitting the encoders.
        # Note: input normalization is just rescaling — no information loss
        # (unlike LayerNorm, which would strip magnitude post-encoding).
        # We address (b) via the config-level fix (smaller global lr + tighter
        # clip_range) — see config.yml global_model block.
        self._build_input_scale_buffers(inner)

        # === Encoders ===
        self.cloudlet_encoder = nn.Linear(max(1, self.cloudlet_feat_dim), d_model)
        self.dc_encoder = nn.Linear(max(1, self.dc_feat_dim), d_model)
        self.ctx_to_cloudlet = nn.Linear(d_model, d_model)
        self.ctx_to_dc = nn.Linear(d_model, d_model)

        # Smaller init on the per-axis encoders so that initial scores are
        # close to uniform (||q||,||k|| ≈ 0.1 instead of ≈ 1.0) — gives PPO a
        # gentle starting policy so the first update doesn't have a
        # 0.9 → 0.0 prob jump anywhere.
        small_init_gain = float(model_config.get("score_encoder_init_gain", 0.3))
        with torch.no_grad():
            self.cloudlet_encoder.weight.mul_(small_init_gain)
            self.dc_encoder.weight.mul_(small_init_gain)

        # score_temperature divides the cosine-style score, making initial
        # logits flatter still (softmax closer to uniform).
        self.score_temperature = float(model_config.get("score_temperature", 2.0))

        # === Route 2.5 (2026-05-19): independent critic trunk ===
        # 100-iter run 20260518_151653 ended with vf_explained_var ≈ 0 (oscillating
        # -0.25..+0.18) and the agent FOUND a good policy at iter 40 (c/c=2.063)
        # but couldn't STAY there — drifted back to c/c=2.10 by iter 100.
        # Diagnosis: value loss back-prop through the SHARED encoders (cloudlet_*
        # + dc_* + gtrxl) was injecting noise into the policy gradient, so the
        # agent kept walking away from the iter-40 optimum.
        # Fix: when `critic_separate_trunk=true`, build a parallel copy of the
        # input-encoder + GTrXL trunk dedicated to the value path; the policy
        # path is untouched.  This isolates the two gradient flows.
        # When false (default), keep the original shared-trunk behavior — old
        # checkpoints stay loadable and existing tests stay green.
        self._critic_separate_trunk = bool(
            model_config.get("critic_separate_trunk", False)
        )
        if self._critic_separate_trunk:
            # Mirror the actor-side encoders + GTrXL — independent params.
            self.critic_cloudlet_encoder = nn.Linear(
                max(1, self.cloudlet_feat_dim), d_model
            )
            self.critic_dc_encoder = nn.Linear(
                max(1, self.dc_feat_dim), d_model
            )
            self.critic_gtrxl = GTrXL(
                input_dim=max(1, self.context_dim),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                mem_len=mem_len,
                max_seq_len=max_seq_len,
            )
            # Optional compile, mirroring the actor branch.
            if bool(model_config.get("compile", False)):
                try:
                    self.critic_gtrxl = torch.compile(
                        self.critic_gtrxl, mode="default", dynamic=True
                    )
                except Exception as e:
                    logger.warning(
                        f"[{self.__class__.__name__}] torch.compile(critic_gtrxl) "
                        f"failed ({e}); using eager.",
                    )
            # Value head: LayerNorm + 2-layer MLP (MAPPO recipe — value loss
            # converges much faster than the original `Linear(3D, 1)`).
            self.value_head = nn.Sequential(
                nn.LayerNorm(3 * d_model),
                nn.Linear(3 * d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
        else:
            # Original shared-trunk path — kept for backward compat.
            self.value_head = nn.Linear(3 * d_model, 1)

        self._last_value = None
        self._debug_dumped = False

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[{self.__class__.__name__}] N_dc={self.num_dcs} "
            f"N_batch={self.num_batch_slots} d_model={d_model} "
            f"dc_feat_dim={self.dc_feat_dim} cloudlet_feat_dim={self.cloudlet_feat_dim} "
            f"context_dim={self.context_dim} total_params={total_params:,}"
        )
        logger.info(f"[{self.__class__.__name__}] dc_keys={self.dc_keys}")
        logger.info(f"[{self.__class__.__name__}] cloudlet_keys={self.cloudlet_keys}")
        logger.info(f"[{self.__class__.__name__}] context_keys={self.context_keys}")

        # 2026-05-18 BC warm-start: optionally seed weights from a behavioral-
        # cloning checkpoint produced by src.training.bc_warmstart.  The
        # checkpoint is a plain torch.save(state_dict) — same architecture is
        # required (we just check param-name compatibility via load_state_dict
        # with strict=True so a mismatch surfaces loudly instead of silently
        # zeroing-out half the weights).
        bc_ckpt = model_config.get("bc_checkpoint_path") or None
        if bc_ckpt:
            try:
                state = torch.load(bc_ckpt, map_location="cpu", weights_only=True)
                self.load_state_dict(state, strict=True)
                logger.info(
                    f"[{self.__class__.__name__}] BC warm-start: loaded "
                    f"weights from {bc_ckpt} ({len(state)} tensors)"
                )
            except Exception as e:
                # Loud failure: a misconfigured BC checkpoint would let PPO
                # quietly start from random init, exactly the failure mode
                # warm-start was supposed to prevent.  Raise rather than fall
                # back.
                raise RuntimeError(
                    f"BC warm-start: failed to load checkpoint from {bc_ckpt}: {e}"
                ) from e

    def _build_input_scale_buffers(self, obs_space: spaces.Dict) -> None:
        """
        Compute per-feature 1/high scale factors for per-DC and per-cloudlet
        inputs.  We expose them as buffers (not Parameters) so they move with
        the module to GPU but don't receive gradient.

        Stored as (1, 1, 1, F) tensors so broadcasting against (B, T, N, F)
        works on the last dim automatically.
        """
        def _key_scale(key: str) -> float:
            sub = obs_space.spaces[key]
            if not isinstance(sub, spaces.Box):
                return 1.0
            high = float(np.max(np.abs(sub.high)))
            if not np.isfinite(high) or high < 1e-6:
                return 1.0
            return 1.0 / high

        dc_scales = [_key_scale(k) for k in self.dc_keys] if self.dc_keys else [1.0]
        cloudlet_scales = [_key_scale(k) for k in self.cloudlet_keys] if self.cloudlet_keys else [1.0]

        # Shape (1, 1, 1, F) so broadcasting hits the F dim of (B, T, N, F).
        self.register_buffer(
            "_dc_scale",
            torch.tensor(dc_scales, dtype=torch.float32).view(1, 1, 1, -1),
        )
        self.register_buffer(
            "_cloudlet_scale",
            torch.tensor(cloudlet_scales, dtype=torch.float32).view(1, 1, 1, -1),
        )

    def _categorize_keys(self, obs_space: spaces.Dict) -> None:
        """
        Auto-bucket obs keys into per-DC, per-cloudlet, or context using
        prefix rules:
            - "dc_*"             → per-DC      (expects shape (N_dc,))
            - "batch_cloudlet_*" → per-cloudlet (expects shape (N_batch_slots,))
            - everything else    → context (flattened)
        Shape mismatches raise so we fail loudly instead of running with a
        degenerate score function.
        """
        self.dc_keys: List[str] = []
        self.cloudlet_keys: List[str] = []
        self.context_keys: List[str] = []
        self.dc_feat_dim = 0
        self.cloudlet_feat_dim = 0
        self.context_dim = 0

        for key in sorted(obs_space.spaces.keys()):
            sub = obs_space.spaces[key]
            if not isinstance(sub, spaces.Box):
                self.context_keys.append(key)
                self.context_dim += 1
                continue

            shape = tuple(sub.shape)
            if key.startswith("dc_"):
                if shape != (self.num_dcs,):
                    raise ValueError(
                        f"Per-DC key {key!r} has shape {shape}, expected ({self.num_dcs},)"
                    )
                self.dc_keys.append(key)
                self.dc_feat_dim += 1
            elif key.startswith("batch_cloudlet"):
                if shape != (self.num_batch_slots,):
                    raise ValueError(
                        f"Per-cloudlet key {key!r} has shape {shape}, expected "
                        f"({self.num_batch_slots},)"
                    )
                self.cloudlet_keys.append(key)
                self.cloudlet_feat_dim += 1
            else:
                self.context_keys.append(key)
                self.context_dim += int(np.prod(shape))

    @override(TorchRLModule)
    def get_initial_state(self):
        actor_mem = np.zeros(
            (self.gtrxl.num_layers, self.gtrxl.mem_len, self.gtrxl.d_model),
            dtype=np.float32,
        )
        if self._critic_separate_trunk:
            critic_mem = np.zeros(
                (
                    self.critic_gtrxl.num_layers,
                    self.critic_gtrxl.mem_len,
                    self.critic_gtrxl.d_model,
                ),
                dtype=np.float32,
            )
            return {
                "gtrxl_mem_actor": actor_mem,
                "gtrxl_mem_critic": critic_mem,
            }
        return {"gtrxl_mem": actor_mem}

    def _to_btD(self, t: torch.Tensor, trailing_len: int) -> torch.Tensor:
        """Normalize a per-step tensor with trailing length `trailing_len` to (B, T, trailing_len)."""
        t = t.float() if t.dtype != torch.float32 else t
        if t.dim() == 2 and t.shape[-1] == trailing_len:
            return t.unsqueeze(1)  # (B, D) → (B, 1, D)
        if t.dim() == 3 and t.shape[-1] == trailing_len:
            return t
        raise RuntimeError(
            f"Cannot normalize tensor of shape {tuple(t.shape)} to (B, T, {trailing_len})"
        )

    def _split_obs(self, obs_dict: Dict[str, torch.Tensor]):
        """
        Split obs dict into:
          per_dc       : (B, T, N_dc, F_dc)
          per_cloudlet : (B, T, N_batch_slots, F_c)
          context      : (B, T, F_ctx)
        """
        # Per-DC: each key is (B, [T,] N_dc).  Stack along new last axis.
        dc_tensors = [
            self._to_btD(obs_dict[k], self.num_dcs) for k in self.dc_keys
        ]
        cloudlet_tensors = [
            self._to_btD(obs_dict[k], self.num_batch_slots) for k in self.cloudlet_keys
        ]
        if dc_tensors:
            per_dc = torch.stack(dc_tensors, dim=-1)  # (B, T, N_dc, F_dc)
        else:
            raise RuntimeError("No per-DC features found in obs")
        if cloudlet_tensors:
            per_cloudlet = torch.stack(cloudlet_tensors, dim=-1)
        else:
            raise RuntimeError("No per-cloudlet features found in obs")

        # Context: concat — each key is (B, [T,] F_k)
        ctx_pieces: List[torch.Tensor] = []
        for k in self.context_keys:
            v = obs_dict[k]
            v = v.float() if v.dtype != torch.float32 else v
            if v.dim() == 1:
                v = v.unsqueeze(-1)  # (B,) → (B, 1)
            if v.dim() == 2:
                # (B, F) → (B, 1, F)
                v = v.unsqueeze(1)
            elif v.dim() == 3:
                pass
            else:
                raise ValueError(f"Context key {k!r}: unexpected shape {tuple(v.shape)}")
            v = v.reshape(v.shape[0], v.shape[1], -1)
            ctx_pieces.append(v)
        if not ctx_pieces:
            ctx_pieces = [torch.zeros((per_dc.shape[0], per_dc.shape[1], 1), device=per_dc.device)]
        context = torch.cat(ctx_pieces, dim=-1)  # (B, T, F_ctx)

        # Align T across per_dc / per_cloudlet / context.
        T_target = max(per_dc.shape[1], per_cloudlet.shape[1], context.shape[1])

        def _align(t):
            if t.shape[1] == T_target:
                return t
            if t.shape[1] == 1:
                return t.expand(t.shape[0], T_target, *t.shape[2:]).contiguous()
            raise RuntimeError(f"Mismatched T: have {t.shape[1]} vs target {T_target}")

        return _align(per_dc), _align(per_cloudlet), _align(context)

    def _forward_pass(self, batch: Dict[str, Any], state_in: Any = None):
        obs = batch.get(Columns.OBS, batch.get("obs", {}))
        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]
        if not isinstance(obs, dict):
            raise RuntimeError(
                f"[{self.__class__.__name__}] expected dict obs, got {type(obs)}"
            )

        per_dc, per_cloudlet, context = self._split_obs(obs)

        if not self._debug_dumped:
            self._debug_dumped = True
            logger.debug("=" * 70)
            logger.debug(f"=== [{self.__class__.__name__}] DEBUG DUMP ===")
            logger.debug(f"per_dc       : {tuple(per_dc.shape)}")
            logger.debug(f"per_cloudlet : {tuple(per_cloudlet.shape)}")
            logger.debug(f"context      : {tuple(context.shape)}")
            logger.debug("=" * 70)

        if _DEBUG_NAN_CHECKS:
            for name, t in [("per_dc", per_dc), ("per_cloudlet", per_cloudlet), ("context", context)]:
                if not torch.isfinite(t).all():
                    raise ValueError(f"Non-finite values in {name}")

        B = context.shape[0]

        # === Actor branch (always present) ===
        # State key: with dual-trunk we use "gtrxl_mem_actor" to keep actor and
        # critic memories separate; with shared-trunk we keep the legacy
        # "gtrxl_mem" so old checkpoints / get_initial_state contracts round-trip.
        actor_state_key = "gtrxl_mem_actor" if self._critic_separate_trunk else "gtrxl_mem"
        actor_memories_in = _parse_gtrxl_state_in(
            batch,
            B,
            self.gtrxl.num_layers,
            self.gtrxl.mem_len,
            self.gtrxl.d_model,
            context.device,
            context.dtype,
            state_key=actor_state_key,
        )
        ctx_features, actor_memories_out = self.gtrxl(
            context, state=actor_memories_in
        )

        if _DEBUG_NAN_CHECKS and not torch.isfinite(ctx_features).all():
            raise ValueError("Non-finite values in ctx_features")

        # Input scale fix: divide by obs_space.high so raw values (e.g. mi=2e6,
        # pes=100, green_power=5e6) all live in ~[-1, 1].  Preserves info
        # (unlike LayerNorm) and keeps the encoders' first-layer outputs O(1).
        per_dc = per_dc * self._dc_scale  # broadcast on last (feature) dim
        per_cloudlet = per_cloudlet * self._cloudlet_scale

        cloudlet_emb = self.cloudlet_encoder(per_cloudlet)  # (B, T, N_b, D)
        dc_emb = self.dc_encoder(per_dc)                    # (B, T, N_d, D)

        q = cloudlet_emb + self.ctx_to_cloudlet(ctx_features).unsqueeze(2)
        k = dc_emb       + self.ctx_to_dc(ctx_features).unsqueeze(2)

        scores = torch.einsum("btid,btjd->btij", q, k) / (
            math.sqrt(self.d_model) * self.score_temperature
        )
        T = scores.shape[1]
        logits = scores.reshape(B, T, self.action_dim)

        # === Critic branch ===
        if self._critic_separate_trunk:
            # Independent encoders + GTrXL trunk → value gradient never reaches
            # actor parameters.  Tested by test_critic_separate_trunk_actor_grad_isolated.
            crit_cloudlet_emb = self.critic_cloudlet_encoder(per_cloudlet)
            crit_dc_emb = self.critic_dc_encoder(per_dc)
            crit_memories_in = _parse_gtrxl_state_in(
                batch,
                B,
                self.critic_gtrxl.num_layers,
                self.critic_gtrxl.mem_len,
                self.critic_gtrxl.d_model,
                context.device,
                context.dtype,
                state_key="gtrxl_mem_critic",
            )
            crit_ctx_features, crit_memories_out = self.critic_gtrxl(
                context, state=crit_memories_in
            )
            crit_dc_pooled = crit_dc_emb.mean(dim=2)
            crit_cloudlet_pooled = crit_cloudlet_emb.mean(dim=2)
            value_input = torch.cat(
                [crit_ctx_features, crit_dc_pooled, crit_cloudlet_pooled],
                dim=-1,
            )
            values = self.value_head(value_input).squeeze(-1)  # (B, T)
            state_out = {
                **_gtrxl_state_out(actor_memories_out, state_key="gtrxl_mem_actor"),
                **_gtrxl_state_out(crit_memories_out, state_key="gtrxl_mem_critic"),
            }
        else:
            # Shared-trunk legacy path — value head reads the actor's encoders.
            dc_pooled = dc_emb.mean(dim=2)
            cloudlet_pooled = cloudlet_emb.mean(dim=2)
            value_input = torch.cat([ctx_features, dc_pooled, cloudlet_pooled], dim=-1)
            values = self.value_head(value_input).squeeze(-1)  # (B, T)
            state_out = _gtrxl_state_out(actor_memories_out, state_key="gtrxl_mem")

        return logits, values, state_out

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, values, state_out = self._forward_pass(batch)
        self._last_value = values
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
            # 2026-06-12 dead-critic fix — see GTrXLMaskedActionRLModule.
            # With critic_separate_trunk these values flow from the CRITIC
            # encoders/trunk only, so the vf gradient keeps actor isolation.
            Columns.EMBEDDINGS: values,
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, _, state_out = self._forward_pass(batch)
        logits = logits[:, -1, :]
        return {
            Columns.ACTION_DIST_INPUTS: logits.unsqueeze(1),
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, values, state_out = self._forward_pass(batch)
        logits = logits[:, -1, :]
        values = values[:, -1]
        self._last_value = values
        return {
            Columns.ACTION_DIST_INPUTS: logits.unsqueeze(1),
            Columns.VF_PREDS: values.unsqueeze(1),
            Columns.STATE_OUT: state_out,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        # 2026-06-12 dead-critic fix — grad-carrying values from
        # _forward_train (learner loss path); see GTrXLMaskedActionRLModule.
        if embeddings is not None:
            return embeddings
        # No-embeddings path (GAE full-batch bootstrap) — keep the
        # 2026-05-12 inference_mode OOM protection.
        with torch.inference_mode():
            _, values, _ = self._forward_pass(batch)
        return values.clone()

    def _get_multi_categorical_cls(self, action_space):
        input_lens = list(action_space.nvec)

        class BoundMultiCategorical(TorchMultiCategorical):
            @staticmethod
            def from_logits(logits, **kwargs):
                return TorchMultiCategorical.from_logits(
                    logits, input_lens=input_lens, **kwargs
                )

        return BoundMultiCategorical

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return self.action_dist_cls

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return self.action_dist_cls

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return self.action_dist_cls

    def get_non_inference_attributes(self):
        return ["value_head", "_last_value"]
