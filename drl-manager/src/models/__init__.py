"""Custom RLlib models for green scheduling."""

# Legacy API models (TorchModelV2 - enable_rl_module_and_learner=False)
from .masked_action_model import MaskedActionModel, DictObsModel

# New RLModule API models (enable_rl_module_and_learner=True)
from .rlmodule_models import MaskedActionRLModule, DictObsRLModule

__all__ = [
    # Legacy API (TorchModelV2)
    "MaskedActionModel",
    "DictObsModel",
    # New RLModule API - MLP based
    "MaskedActionRLModule",
    "DictObsRLModule",
]
