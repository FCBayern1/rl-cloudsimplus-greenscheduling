"""Custom RLlib models for green scheduling."""

# Legacy API models (TorchModelV2 - enable_rl_module_and_learner=False)
from .masked_action_model import MaskedActionModel, DictObsModel

# New RLModule API models (enable_rl_module_and_learner=True)
from .rlmodule_models import MaskedActionRLModule, DictObsRLModule
from .rlmodule_resmlp_models import ResMLPMaskedActionRLModule, ResMLPDictObsRLModule
from .rlmodule_gmlp_models import gMLPMaskedActionRLModule, gMLPDictObsRLModule

# GTrXL RLModule API models (Gated Transformer-XL)
# Note: Uncomment when gtrxl_rlmodule.py and gtrxl_networks.py are created
# from .gtrxl_rlmodule import GTrXLMaskedActionRLModule, GTrXLDictObsRLModule
# from .gtrxl_networks import GTrXLEncoder

__all__ = [
    # Legacy API (TorchModelV2)
    "MaskedActionModel",
    "DictObsModel",
    # New RLModule API - MLP based
    "MaskedActionRLModule",
    "DictObsRLModule",
    # New RLModule API - ResMLP based
    "ResMLPMaskedActionRLModule",
    "ResMLPDictObsRLModule",
    # New RLModule API - gMLP based
    "gMLPMaskedActionRLModule",
    "gMLPDictObsRLModule",
    # New RLModule API - GTrXL based (uncomment when files exist)
    # "GTrXLMaskedActionRLModule",
    # "GTrXLDictObsRLModule",
    # "GTrXLEncoder",
]
