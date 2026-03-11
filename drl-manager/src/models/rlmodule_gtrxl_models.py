"""
Gated Transformer-XL RLModule implementations for Multi-Datacenter Green Scheduling.

This module provides RLModules that utilize the GTrXL architecture for both
Local Agents (with Action Masking) and the Global Agent.

Architecture:
- Input (Dict/Flat) -> Embedding -> GTrXL (Gated Transformer) -> Heads
"""

from typing import Any, Dict, Optional, List
import logging
import torch
import torch.nn as nn
from gymnasium import spaces

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI, ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchMultiCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from src.networks.gtrxl import GTrXL

logger = logging.getLogger(__name__)

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

        # Build GTrXL Backbone
        self.gtrxl = GTrXL(
            input_dim=self.obs_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
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
        logger.info(f"[{self.__class__.__name__}] d_model={d_model}, nhead={nhead}, num_layers={num_layers}")

    # NOTE: get_initial_state removed to avoid RLlib treating this as stateful/recurrent
    # which causes Connector v2 to expect time-major format (T, B, ...) instead of batch-first

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
        """Shared forward pass logic."""
        flat_obs, action_mask = self._extract_obs_and_mask(batch)

        # === DEBUG: One-time dump of batch structure ===
        if not self._debug_dumped:
            self._debug_dumped = True
            logger.error("=" * 70)
            logger.error(f"=== [{self.__class__.__name__}] DEBUG DUMP (first batch) ===")
            logger.error("=" * 70)
            logger.error(f"self.obs_dim (from space) = {self.obs_dim}")
            logger.error(f"flat_obs.shape = {tuple(flat_obs.shape)}")
            logger.error(f"flat_obs.dtype = {flat_obs.dtype}")

            obs = batch.get(Columns.OBS, batch.get("obs", {}))
            logger.error(f"raw obs type = {type(obs)}")
            if isinstance(obs, dict):
                logger.error(f"raw obs keys = {list(obs.keys())}")

            # Print each key's tensor shape
            def dump_shapes(prefix, o):
                if isinstance(o, dict):
                    for k in sorted(o.keys()):
                        dump_shapes(f"{prefix}.{k}", o[k])
                elif isinstance(o, torch.Tensor):
                    logger.error(f"{prefix}: tensor shape={tuple(o.shape)} dtype={o.dtype}")
                else:
                    try:
                        import numpy as np
                        arr = np.asarray(o)
                        logger.error(f"{prefix}: array shape={arr.shape} dtype={arr.dtype}")
                    except Exception:
                        logger.error(f"{prefix}: {type(o)}")

            if isinstance(obs, dict):
                dump_shapes("obs", obs)

            # Check if it looks like flattened sequence
            feat_dim = flat_obs.shape[-1]
            if feat_dim != self.obs_dim:
                logger.error(f"!!! DIMENSION MISMATCH: model expects {self.obs_dim}, got {feat_dim} !!!")
                if feat_dim % self.obs_dim == 0 and feat_dim > self.obs_dim:
                    T = feat_dim // self.obs_dim
                    logger.error(f"Looks like flattened sequence: {feat_dim} = {self.obs_dim} * T({T})")

            logger.error("=" * 70)
            logger.error("=== END DEBUG DUMP ===")
            logger.error("=" * 70)

        # === FAIL FAST: Check feature dimension ===
        feat_dim = flat_obs.shape[-1]
        if feat_dim != self.obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Obs dim mismatch! "
                f"Model expects {self.obs_dim} (from observation_space), "
                f"but got {feat_dim} from batch flatten. "
                f"Likely cause: env output != space declaration, OR sequence flattened into features."
            )

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
        
        # Pass through GTrXL
        # We ignore state passing for strictly GTrXL logic here
        # (relying on Attention over the sequence provided)
        # unless we implement full memory.
        # We pass dummy state to satisfy signature.
        features, _ = self.gtrxl(flat_obs, state=[])

        # === CHECKPOINT 3: Check if GTrXL features have NaN/Inf ===
        if not torch.isfinite(features).all():
            bad_ratio = (~torch.isfinite(features)).float().mean().item()
            logger.error(f"[{self.__class__.__name__}] GTrXL features has non-finite values! ratio={bad_ratio:.4f}")
            raise ValueError(f"Non-finite values in GTrXL features (ratio={bad_ratio:.4f})")

        # Features: (B, T, d_model)

        # Calculate logits and values over full sequence
        logits = self.policy_head(features)  # (B, T, A)
        values = self.value_head(features).squeeze(-1)  # (B, T)

        # === PPO expects per-sample outputs, not full sequences ===
        # Slice to last timestep only - GTrXL uses history to predict current step
        logits = logits[:, -1, :]  # (B, A)
        values = values[:, -1]     # (B,)

        # Apply Masking with numerical stability fixes
        if action_mask is not None:
            # Slice mask to last timestep as well
            if action_mask.dim() == 3:
                # (B, T, A) -> (B, A)
                action_mask = action_mask[:, -1, :]
            elif action_mask.dim() == 2:
                # Already (B, A) or (B*T, A)
                B = logits.shape[0]
                if action_mask.shape[0] != B:
                    # (B*T, A) - take last T entries for each B
                    # This is tricky; assume it's already aligned or reshape
                    T = action_mask.shape[0] // B
                    action_mask = action_mask.view(B, T, -1)[:, -1, :]

            # Clean mask: replace NaN with 0
            action_mask = torch.nan_to_num(action_mask, nan=0.0, posinf=0.0, neginf=0.0)

            valid = action_mask >= 0.5  # (B, A)

            # Fallback: if no valid action, force action 0 to be valid
            valid_cnt = valid.sum(dim=-1, keepdim=True)  # (B, 1)
            no_valid = valid_cnt == 0
            if no_valid.any():
                valid = valid.clone()
                valid[:, 0] = valid[:, 0] | no_valid.squeeze(-1)

            # Use -1e9 instead of -inf for numerical stability
            logits = torch.where(valid, logits, torch.full_like(logits, -1e9))

        return logits, values

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, values = self._forward_pass(batch)
        self._last_value = values
        # NOTE: STATE_OUT removed to avoid RLlib Connector v2 time-rank issues
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, _ = self._forward_pass(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        _, values = self._forward_pass(batch)
        return values

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

        self.gtrxl = GTrXL(
            input_dim=self.obs_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.policy_head = nn.Linear(d_model, self.action_dim)
        self.value_head = nn.Linear(d_model, 1)
        self._last_value = None
        self._debug_dumped = False  # For one-time debug dump

        # === DEBUG: Log computed obs_dim ===
        logger.info(f"[{self.__class__.__name__}] obs_space={self.observation_space}")
        logger.info(f"[{self.__class__.__name__}] computed obs_dim={self.obs_dim}")
        logger.info(f"[{self.__class__.__name__}] action_dim={self.action_dim}")
        logger.info(f"[{self.__class__.__name__}] d_model={d_model}, nhead={nhead}, num_layers={num_layers}")

    # NOTE: get_initial_state removed to avoid RLlib treating this as stateful/recurrent
    # which causes Connector v2 to expect time-major format (T, B, ...) instead of batch-first

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
            logger.error("=" * 70)
            logger.error(f"=== [{self.__class__.__name__}] DEBUG DUMP (first batch) ===")
            logger.error("=" * 70)
            logger.error(f"self.obs_dim (from space) = {self.obs_dim}")
            logger.error(f"flat_obs.shape = {tuple(flat_obs.shape)}")
            logger.error(f"flat_obs.dtype = {flat_obs.dtype}")

            raw_obs = batch.get(Columns.OBS, batch.get("obs", {}))
            logger.error(f"raw obs type = {type(raw_obs)}")
            if isinstance(raw_obs, dict):
                logger.error(f"raw obs keys = {list(raw_obs.keys())}")

            # Print each key's tensor shape
            def dump_shapes(prefix, o):
                if isinstance(o, dict):
                    for k in sorted(o.keys()):
                        dump_shapes(f"{prefix}.{k}", o[k])
                elif isinstance(o, torch.Tensor):
                    logger.error(f"{prefix}: tensor shape={tuple(o.shape)} dtype={o.dtype}")
                else:
                    try:
                        import numpy as np
                        arr = np.asarray(o)
                        logger.error(f"{prefix}: array shape={arr.shape} dtype={arr.dtype}")
                    except Exception:
                        logger.error(f"{prefix}: {type(o)}")

            if isinstance(raw_obs, dict):
                dump_shapes("obs", raw_obs)

            # Check if it looks like flattened sequence
            feat_dim = flat_obs.shape[-1]
            if feat_dim != self.obs_dim:
                logger.error(f"!!! DIMENSION MISMATCH: model expects {self.obs_dim}, got {feat_dim} !!!")
                if feat_dim % self.obs_dim == 0 and feat_dim > self.obs_dim:
                    T = feat_dim // self.obs_dim
                    logger.error(f"Looks like flattened sequence: {feat_dim} = {self.obs_dim} * T({T})")

            logger.error("=" * 70)
            logger.error("=== END DEBUG DUMP ===")
            logger.error("=" * 70)

        # === FAIL FAST: Check feature dimension ===
        feat_dim = flat_obs.shape[-1]
        if feat_dim != self.obs_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Obs dim mismatch! "
                f"Model expects {self.obs_dim} (from observation_space), "
                f"but got {feat_dim} from batch flatten. "
                f"Likely cause: env output != space declaration, OR sequence flattened into features."
            )

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

        features, _ = self.gtrxl(flat_obs, state=[])

        # === CHECKPOINT 3: Check if GTrXL features have NaN/Inf ===
        if not torch.isfinite(features).all():
            bad_ratio = (~torch.isfinite(features)).float().mean().item()
            logger.error(f"[{self.__class__.__name__}] GTrXL features has non-finite values! ratio={bad_ratio:.4f}")
            raise ValueError(f"Non-finite values in GTrXL features (ratio={bad_ratio:.4f})")

        # Calculate logits and values over full sequence
        logits = self.policy_head(features)  # (B, T, A)
        values = self.value_head(features).squeeze(-1)  # (B, T)

        # === PPO expects per-sample outputs, not full sequences ===
        # Slice to last timestep only - GTrXL uses history to predict current step
        logits = logits[:, -1, :]  # (B, A)
        values = values[:, -1]     # (B,)

        return logits, values

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, values = self._forward_pass(batch)
        self._last_value = values
        # NOTE: STATE_OUT removed to avoid RLlib Connector v2 time-rank issues
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        logits, _ = self._forward_pass(batch)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> TensorType:
        _, values = self._forward_pass(batch)
        return values

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
