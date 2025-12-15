"""Custom RLlib models for green scheduling."""

# Legacy API models (TorchModelV2 - enable_rl_module_and_learner=False)
from .masked_action_model import MaskedActionModel, DictObsModel
from .trxl_obsrec_model import TransformerXLObsRecModel
from .lstm_masked_model import LSTMMaskedActionModel, LSTMDictObsModel

# New RLModule API models (enable_rl_module_and_learner=True)
from .rlmodule_models import MaskedActionRLModule, DictObsRLModule
from .transformerxl_rlmodule import (
    TransformerXLMaskedRLModule,
    TransformerXLDictObsRLModule,
    TransformerXLEncoder,
)

__all__ = [
    # Legacy API (TorchModelV2)
    "MaskedActionModel",
    "DictObsModel",
    "TransformerXLObsRecModel",
    "LSTMMaskedActionModel",
    "LSTMDictObsModel",
    # New RLModule API - MLP based
    "MaskedActionRLModule",
    "DictObsRLModule",
    # New RLModule API - Transformer-XL based
    "TransformerXLMaskedRLModule",
    "TransformerXLDictObsRLModule",
    "TransformerXLEncoder",
]
