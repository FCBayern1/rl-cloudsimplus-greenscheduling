"""
GTrXL-based RLModule implementations for Multi-Datacenter Green Scheduling.

Two RLModule classes:
1. GTrXLMaskedActionRLModule: For Local agents (Discrete actions + action masking)
2. GTrXLDictObsRLModule: For Global agent (MultiDiscrete actions)

Key RLlib integration points:
- get_initial_state(): Returns memory structure for STATE_IN
- _forward_*(): Process batch with Columns.STATE_IN, return Columns.STATE_OUT
- is_stateful(): Returns True to enable recurrent mode
"""

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI, ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchMultiCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

from .gtrxl_networks import GTrXLEncoder


class GTrXLMaskedActionRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """
    GTrXL-based PPO RLModule with action masking for Local agents.

    Features:
    - Gated Transformer-XL encoder for temporal modeling
    - Segment-level memory (default 64 steps)
    - Action masking for invalid VM assignments
    - Separate policy and value heads

    State Structure (STATE_IN/STATE_OUT):
    {
        "layer_0": Tensor(batch, mem_len, d_model),
        "layer_1": Tensor(batch, mem_len, d_model),
        ...
    }
    """

    @override(TorchRLModule)
    def setup(self):
        """Initialize GTrXL network and policy/value heads."""
        model_config = self.model_config
        action_space = self.action_space
        obs_space = self.observation_space

        # Validate action space
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        else:
            raise ValueError(f"GTrXLMaskedActionRLModule requires Discrete action space, got {type(action_space)}")

        # Calculate observation dimension
        self.obs_dim = self._get_obs_dim(obs_space)

        # GTrXL configuration
        self.d_model = model_config.get("d_model", 256)
        self.num_heads = model_config.get("num_heads", 4)
        self.num_layers = model_config.get("num_layers", 2)
        self.d_ff = model_config.get("d_ff", 512)
        self.mem_len = model_config.get("mem_len", 64)
        self.dropout = model_config.get("dropout", 0.1)

        # Build GTrXL encoder
        self.encoder = GTrXLEncoder(
            obs_dim=self.obs_dim,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            mem_len=self.mem_len,
            dropout=self.dropout,
        )

        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.action_dim),
        )

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def _get_obs_dim(self, obs_space) -> int:
        """Calculate total observation dimension from space."""
        if isinstance(obs_space, spaces.Box):
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, spaces.Dict):
            if "observation" in obs_space.spaces:
                return self._get_obs_dim(obs_space.spaces["observation"])
            return sum(self._get_obs_dim(s) for s in obs_space.spaces.values())
        elif isinstance(obs_space, spaces.Discrete):
            return 1
        elif isinstance(obs_space, spaces.MultiDiscrete):
            return len(obs_space.nvec)
        else:
            raise ValueError(f"Unsupported obs space: {type(obs_space)}")

    def _flatten_obs(self, obs: Dict[str, TensorType]) -> TensorType:
        """Flatten Dict observation to single tensor with robust dimension handling."""
        if isinstance(obs, torch.Tensor):
            if obs.dim() == 1:
                return obs.unsqueeze(0)
            return obs

        tensors = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, dict):
                val = self._flatten_obs(val)
            if isinstance(val, torch.Tensor):
                tensors.append(val.float())

        if len(tensors) == 0:
            raise ValueError("No valid tensors found in observation")

        # Robust concatenation: handle mixed scalar/vector features
        # e.g., (B, T) vs (B, T, F) -> unsqueeze scalars to (B, T, 1)
        max_dim = max(t.dim() for t in tensors)
        processed_tensors = []
        for t in tensors:
            while t.dim() < max_dim:
                t = t.unsqueeze(-1)
            processed_tensors.append(t)

        return torch.cat(processed_tensors, dim=-1)

    def _extract_obs_and_mask(self, batch: Dict[str, Any]) -> tuple:
        """Extract observation and action mask from batch."""
        obs = batch.get(Columns.OBS, batch.get("obs", {}))

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

        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, dtype=torch.float32)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

        return flat_obs, action_mask

    def _extract_state(self, batch: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Extract memory state from batch."""
        state_in = batch.get(Columns.STATE_IN, {})

        # If state is empty or None, return None to trigger initial state
        if not state_in:
            return None

        # Convert numpy arrays to tensors if needed
        converted_state = {}
        for key, val in state_in.items():
            if isinstance(val, np.ndarray):
                converted_state[key] = torch.from_numpy(val).float()
            elif isinstance(val, torch.Tensor):
                converted_state[key] = val
            else:
                converted_state[key] = val

        return converted_state

    def _apply_action_mask(
        self, logits: TensorType, action_mask: Optional[TensorType]
    ) -> TensorType:
        """Apply action mask: invalid actions get -inf logits."""
        if action_mask is None:
            return logits

        if action_mask.device != logits.device:
            action_mask = action_mask.to(logits.device)

        if action_mask.shape[0] != logits.shape[0]:
            if action_mask.shape[0] == 1:
                action_mask = action_mask.expand(logits.shape[0], -1)

        # Handle size mismatch
        if action_mask.shape[-1] > logits.shape[-1]:
            action_mask = action_mask[..., :logits.shape[-1]]
        elif action_mask.shape[-1] < logits.shape[-1]:
            pad_size = logits.shape[-1] - action_mask.shape[-1]
            action_mask = torch.cat([
                action_mask,
                torch.zeros(*action_mask.shape[:-1], pad_size, device=action_mask.device)
            ], dim=-1)

        invalid_mask = action_mask < 0.5
        return logits.masked_fill(invalid_mask, float("-inf"))

    def _get_initial_state_tensors(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Create initial zero state tensors."""
        return {
            f"layer_{i}": torch.zeros(batch_size, self.mem_len, self.d_model, device=device)
            for i in range(self.num_layers)
        }

    @override(TorchRLModule)
    def get_initial_state(self) -> Dict[str, np.ndarray]:
        """Return initial memory state for GTrXL."""
        initial_state = {}
        for i in range(self.num_layers):
            initial_state[f"layer_{i}"] = np.zeros(
                (self.mem_len, self.d_model), dtype=np.float32
            )
        return initial_state

    @override(TorchRLModule)
    def is_stateful(self) -> bool:
        """GTrXL is stateful - it maintains memory."""
        return True

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for training with state management."""
        flat_obs, action_mask = self._extract_obs_and_mask(batch)
        state_in = self._extract_state(batch)

        # Handle missing state (first step)
        if state_in is None:
            batch_size = flat_obs.shape[0]
            state_in = self._get_initial_state_tensors(batch_size, flat_obs.device)

        # Ensure state tensors are on correct device
        state_in = {k: v.to(flat_obs.device) for k, v in state_in.items()}

        # Encode with GTrXL
        embeddings, state_out = self.encoder(flat_obs, state_in)

        # Compute policy logits and values
        logits = self.policy_head(embeddings)
        values = self.value_head(embeddings).squeeze(-1)

        # Apply action masking
        masked_logits = self._apply_action_mask(logits, action_mask)

        return {
            Columns.ACTION_DIST_INPUTS: masked_logits,
            Columns.VF_PREDS: values,
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for inference (deterministic)."""
        flat_obs, action_mask = self._extract_obs_and_mask(batch)
        state_in = self._extract_state(batch)

        if state_in is None:
            batch_size = flat_obs.shape[0]
            state_in = self._get_initial_state_tensors(batch_size, flat_obs.device)

        state_in = {k: v.to(flat_obs.device) for k, v in state_in.items()}

        with torch.no_grad():
            embeddings, state_out = self.encoder(flat_obs, state_in)
            logits = self.policy_head(embeddings)

        masked_logits = self._apply_action_mask(logits, action_mask)

        return {
            Columns.ACTION_DIST_INPUTS: masked_logits,
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass for exploration (stochastic)."""
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        """Compute value function predictions."""
        if embeddings is not None:
            return self.value_head(embeddings).squeeze(-1)

        flat_obs, _ = self._extract_obs_and_mask(batch)
        state_in = self._extract_state(batch)

        if state_in is None:
            batch_size = flat_obs.shape[0]
            state_in = self._get_initial_state_tensors(batch_size, flat_obs.device)

        state_in = {k: v.to(flat_obs.device) for k, v in state_in.items()}

        embeddings, _ = self.encoder(flat_obs, state_in)
        return self.value_head(embeddings).squeeze(-1)

    def get_non_inference_attributes(self):
        """Return attributes not needed for inference."""
        return ["value_head"]

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return TorchCategorical


class GTrXLDictObsRLModule(TorchRLModule, InferenceOnlyAPI, ValueFunctionAPI):
    """
    GTrXL-based PPO RLModule for Global agent (MultiDiscrete actions).

    Same GTrXL architecture as Local, but:
    - No action masking
    - MultiDiscrete action space (one DC per cloudlet in batch)
    """

    @override(TorchRLModule)
    def setup(self):
        """Initialize GTrXL network and policy/value heads."""
        model_config = self.model_config
        action_space = self.action_space
        obs_space = self.observation_space

        # Handle action space
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_dim = int(sum(action_space.nvec))
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

        # Calculate observation dimension
        self.obs_dim = self._get_obs_dim(obs_space)

        # GTrXL configuration
        self.d_model = model_config.get("d_model", 256)
        self.num_heads = model_config.get("num_heads", 4)
        self.num_layers = model_config.get("num_layers", 2)
        self.d_ff = model_config.get("d_ff", 512)
        self.mem_len = model_config.get("mem_len", 64)
        self.dropout = model_config.get("dropout", 0.1)

        # Build GTrXL encoder
        self.encoder = GTrXLEncoder(
            obs_dim=self.obs_dim,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            mem_len=self.mem_len,
            dropout=self.dropout,
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def _get_obs_dim(self, obs_space) -> int:
        """Calculate total observation dimension."""
        if isinstance(obs_space, spaces.Box):
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, spaces.Dict):
            if "observation" in obs_space.spaces:
                return self._get_obs_dim(obs_space.spaces["observation"])
            return sum(self._get_obs_dim(s) for s in obs_space.spaces.values())
        elif isinstance(obs_space, spaces.Discrete):
            return 1
        elif isinstance(obs_space, spaces.MultiDiscrete):
            return len(obs_space.nvec)
        else:
            raise ValueError(f"Unsupported obs space: {type(obs_space)}")

    def _flatten_obs(self, obs: Dict[str, TensorType]) -> TensorType:
        """Flatten Dict observation to single tensor with robust dimension handling."""
        if isinstance(obs, torch.Tensor):
            if obs.dim() == 1:
                return obs.unsqueeze(0)
            return obs

        tensors = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, dict):
                val = self._flatten_obs(val)
            if isinstance(val, torch.Tensor):
                tensors.append(val.float())

        if len(tensors) == 0:
            raise ValueError("No valid tensors found in observation")

        # Robust concatenation: handle mixed scalar/vector features
        # e.g., (B, T) vs (B, T, F) -> unsqueeze scalars to (B, T, 1)
        max_dim = max(t.dim() for t in tensors)
        processed_tensors = []
        for t in tensors:
            while t.dim() < max_dim:
                t = t.unsqueeze(-1)
            processed_tensors.append(t)

        return torch.cat(processed_tensors, dim=-1)

    def _extract_obs(self, batch: Dict[str, Any]) -> TensorType:
        """Extract and flatten observation from batch."""
        obs = batch.get(Columns.OBS, batch.get("obs", {}))

        if isinstance(obs, dict):
            if "observation" in obs:
                true_obs = obs["observation"]
            else:
                true_obs = obs
        else:
            true_obs = obs

        return self._flatten_obs(true_obs)

    def _extract_state(self, batch: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Extract memory state from batch."""
        state_in = batch.get(Columns.STATE_IN, {})
        if not state_in:
            return None

        converted_state = {}
        for key, val in state_in.items():
            if isinstance(val, np.ndarray):
                converted_state[key] = torch.from_numpy(val).float()
            elif isinstance(val, torch.Tensor):
                converted_state[key] = val
            else:
                converted_state[key] = val

        return converted_state

    def _get_initial_state_tensors(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Create initial zero state tensors."""
        return {
            f"layer_{i}": torch.zeros(batch_size, self.mem_len, self.d_model, device=device)
            for i in range(self.num_layers)
        }

    @override(TorchRLModule)
    def get_initial_state(self) -> Dict[str, np.ndarray]:
        """Return initial memory state."""
        initial_state = {}
        for i in range(self.num_layers):
            initial_state[f"layer_{i}"] = np.zeros(
                (self.mem_len, self.d_model), dtype=np.float32
            )
        return initial_state

    @override(TorchRLModule)
    def is_stateful(self) -> bool:
        return True

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        flat_obs = self._extract_obs(batch)
        state_in = self._extract_state(batch)

        if state_in is None:
            batch_size = flat_obs.shape[0]
            state_in = self._get_initial_state_tensors(batch_size, flat_obs.device)

        state_in = {k: v.to(flat_obs.device) for k, v in state_in.items()}

        embeddings, state_out = self.encoder(flat_obs, state_in)
        logits = self.policy_head(embeddings)
        values = self.value_head(embeddings).squeeze(-1)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        flat_obs = self._extract_obs(batch)
        state_in = self._extract_state(batch)

        if state_in is None:
            batch_size = flat_obs.shape[0]
            state_in = self._get_initial_state_tensors(batch_size, flat_obs.device)

        state_in = {k: v.to(flat_obs.device) for k, v in state_in.items()}

        with torch.no_grad():
            embeddings, state_out = self.encoder(flat_obs, state_in)
            logits = self.policy_head(embeddings)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: state_out,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._forward_train(batch, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        if embeddings is not None:
            return self.value_head(embeddings).squeeze(-1)

        flat_obs = self._extract_obs(batch)
        state_in = self._extract_state(batch)

        if state_in is None:
            batch_size = flat_obs.shape[0]
            state_in = self._get_initial_state_tensors(batch_size, flat_obs.device)

        state_in = {k: v.to(flat_obs.device) for k, v in state_in.items()}

        embeddings, _ = self.encoder(flat_obs, state_in)
        return self.value_head(embeddings).squeeze(-1)

    def get_non_inference_attributes(self):
        """Return attributes not needed for inference."""
        return ["value_head"]

    def _get_multi_categorical_cls(self):
        """Create bound MultiCategorical class."""
        input_lens = list(self.action_space.nvec)

        class BoundMultiCategorical(TorchMultiCategorical):
            @staticmethod
            def from_logits(logits, **kwargs):
                return TorchMultiCategorical.from_logits(logits, input_lens=input_lens, **kwargs)

        return BoundMultiCategorical

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return self._get_multi_categorical_cls()
        return TorchCategorical

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return self._get_multi_categorical_cls()
        return TorchCategorical
