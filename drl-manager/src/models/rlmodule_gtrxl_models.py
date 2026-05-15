"""
Gated Transformer-XL RLModule implementations for Multi-Datacenter Green Scheduling.

This module provides RLModules that utilise the GTrXL architecture for both
Local Agents (with Action Masking) and the Global Agent.

Architecture:
- Input (Dict/Flat) -> Embedding -> GTrXL (Gated Transformer) -> Heads
"""

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
) -> Optional[List[torch.Tensor]]:
    """Build per-layer memory list from RLlib Columns.STATE_IN, or None to zero-init."""
    si = batch.get(Columns.STATE_IN)
    if not isinstance(si, dict) or si is None:
        return None
    raw = si.get("gtrxl_mem")
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


def _gtrxl_state_out(memories: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {"gtrxl_mem": torch.stack(memories, dim=1)}

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

        # 2026-05-13 Level D perf: optionally JIT-compile the GTrXL backbone.
        # The forward pass has a per-T Python loop (T ≈ 48 for our config) over
        # ~15 small ops per layer → ~1.4k PyTorch op-dispatches per minibatch
        # forward.  Each dispatch is ~50 µs of Python+kernel-launch overhead,
        # which dominates total PPO update time for our tiny 0.47M-param model.
        # torch.compile (PyTorch 2.x) fuses these small ops into a single
        # graph + reduces Python overhead, typical 2-5× speedup for dispatch-
        # bound models.  Defaults to ON; turn off via model_config["compile"]
        # = False if compilation fails on a particular env / pytorch version.
        compile_enabled = bool(model_config.get("compile", True))
        if compile_enabled:
            try:
                # mode='default' is the safe choice — it traces Python and
                # applies operator fusion without requiring static-shape
                # CUDAGraph capture (which can break with variable seq_lens).
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
        # 2026-05-12 OOM fix: GAE (rllib/connectors/learner/general_advantage_estimation.py:96)
        # calls compute_values on the FULL post-rollout batch — for our 10-DC v2 setup that's
        # 8000 env steps × 10 shared-local agents = 80 000 samples in a single forward.
        # GTrXL.forward unrolls T timesteps in a Python loop; without no_grad the autograd
        # graph from every intermediate tensor at every layer at every t is retained, which
        # alone consumed 10+ GB of VRAM and was the root cause of "16GB 5080 OOMs on bumping
        # d_model 96 → 128".  GAE only needs V(s) values, not gradients through them, so
        # wrap the forward in inference_mode (slightly cheaper than no_grad) to skip graph
        # construction entirely.  Same pattern applied to Global module below.
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

        # 2026-05-13 Level D perf — same torch.compile rationale as Local module above.
        compile_enabled = bool(model_config.get("compile", True))
        if compile_enabled:
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
