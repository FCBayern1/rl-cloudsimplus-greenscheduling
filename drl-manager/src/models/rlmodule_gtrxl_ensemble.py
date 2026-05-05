"""
Ensemble Q-head module for the EU-CRD framework (V+Q hybrid critic).

Design (see plan §M1):
- The existing GTrXL RLModules keep their `value_head` (V(s)) intact for PPO.
- This module adds K independently-initialized Q-head MLPs that share the
  GTrXL trunk's state representation, plus K *frozen* random prior networks
  of the same architecture (Osband et al., 2018) so that ensemble disagreement
  persists in data-sparse regions.
- The Q-heads are trained off-policy via TD bootstrap (target = r + γ V(s'))
  in M1.2; this file only contains the architecture and the RLModule wiring.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.torch import TorchRLModule

from src.models.rlmodule_gtrxl_models import (
    GTrXLMaskedActionRLModule,
    GTrXLGlobalRLModule,
)


logger = logging.getLogger(__name__)


# Custom batch column for the per-(s, a) Q-ensemble outputs. Used by the M1.2
# loss term and the M2 counterfactual callback. We avoid Columns.* names that
# RLlib reserves; this stays a private extension.
COL_Q_ENSEMBLE = "crd_q_ensemble"


def _make_head_mlp(d_model: int, hidden_dim: int, action_dim: int) -> nn.Sequential:
    """Two-layer MLP used as both a trainable q-head and a frozen prior."""
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
    )


class EnsembleQHeads(nn.Module):
    """
    K-head Q-value ensemble with frozen randomized priors.

    Each effective Q-head is:
        Q_i(s) = q_i(s) + λ · p_i(s)
    where q_i is trainable and p_i is a frozen random init of the same
    architecture. Disagreement Var_i[Q_i(s, a)] is the epistemic uncertainty
    consumed by the CRD soft-blending gate.

    Attributes:
        K: number of ensemble members.
        prior_lambda: scaling λ on the frozen prior contribution.
        action_dim: per-state action vocabulary (e.g., num_VMs+1 for local,
            or num_dc for per-cloudlet marginal Q on global routing).

    Forward signature:
        state_repr: (B, d_model)  — output of the GTrXL trunk
        returns:    (B, K, action_dim)
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        K: int = 5,
        prior_lambda: float = 3.0,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K
        self.action_dim = action_dim
        self.prior_lambda = prior_lambda
        self.hidden_dim = hidden_dim
        self.d_model = d_model

        self.q_heads = nn.ModuleList(
            [_make_head_mlp(d_model, hidden_dim, action_dim) for _ in range(K)]
        )
        self.priors = nn.ModuleList(
            [_make_head_mlp(d_model, hidden_dim, action_dim) for _ in range(K)]
        )

        # Independent re-initialization of each q-head so they start divergent.
        # nn.Linear's default kaiming_uniform_ uses the global RNG; if the user
        # has seeded torch, identical reinits would collapse diversity. Use
        # per-head sub-streams so each head has its own random init.
        for k, head in enumerate(self.q_heads):
            self._reinit_module(head, seed_offset=k)
        for k, prior in enumerate(self.priors):
            # Different seed offset domain so priors don't accidentally clone q-heads.
            self._reinit_module(prior, seed_offset=10_000 + k)

        # Freeze priors: no gradient, never enters the optimizer state.
        for p in self.priors.parameters():
            p.requires_grad = False

    @staticmethod
    def _reinit_module(module: nn.Module, seed_offset: int) -> None:
        """Re-initialize Linear layers with a head-specific RNG."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(torch.initial_seed() % (2**31 - 1) + seed_offset)
        for m in module.modules():
            if isinstance(m, nn.Linear):
                # Kaiming uniform like PyTorch's default, but with our generator.
                fan_in = m.weight.size(1)
                bound = (1.0 / fan_in) ** 0.5
                with torch.no_grad():
                    m.weight.uniform_(-bound, bound, generator=gen)
                    if m.bias is not None:
                        m.bias.uniform_(-bound, bound, generator=gen)

    def forward(self, state_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_repr: (B, d_model)
        Returns:
            (B, K, action_dim) — Q values across the ensemble, prior already added.
        """
        if state_repr.dim() != 2:
            raise ValueError(
                f"EnsembleQHeads expects (B, d_model), got shape {tuple(state_repr.shape)}"
            )

        # Stack along K dim so downstream lookups are simple.
        q_stack = torch.stack([h(state_repr) for h in self.q_heads], dim=1)
        # Detach prior — it's frozen, but detach makes intent explicit and
        # eliminates any chance of grad flowing into prior params.
        with torch.no_grad():
            p_stack = torch.stack([p(state_repr) for p in self.priors], dim=1)
        return q_stack + self.prior_lambda * p_stack

    def compute_q_for_action(
        self, state_repr: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Look up Q[:, :, action] and reduce over the ensemble dim.

        Args:
            state_repr: (B, d_model)
            action:     (B,) integer tensor in [0, action_dim)
        Returns:
            mu:  (B,)  ensemble-mean Q at the chosen action
            var: (B,)  ensemble-variance Q at the chosen action (unbiased=False)
        """
        if action.dim() != 1:
            raise ValueError(
                f"compute_q_for_action expects 1-D action, got shape {tuple(action.shape)}"
            )
        q_all = self.forward(state_repr)              # (B, K, A)
        # gather: index dim=2 with action expanded to (B, K, 1)
        idx = action.long().view(-1, 1, 1).expand(-1, self.K, 1)
        q_chosen = q_all.gather(2, idx).squeeze(-1)   # (B, K)
        mu = q_chosen.mean(dim=1)
        var = q_chosen.var(dim=1, unbiased=False)
        return mu, var


# ============================================================================
# RLModule subclasses — wire EnsembleQHeads onto existing GTrXL trunks.
# ============================================================================


def _read_crd_ensemble_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull ensemble settings from model_config['crd']['ensemble'] with defaults."""
    crd_cfg = model_config.get("crd", {}) or {}
    ens_cfg = crd_cfg.get("ensemble", {}) or {}
    return {
        "K": int(ens_cfg.get("K", 5)),
        "prior_lambda": float(ens_cfg.get("prior_lambda", 3.0)),
        "hidden_dim": int(ens_cfg.get("hidden_dim", 128)),
    }


class _GTrXLFeatureCapture:
    """
    Mixin-style helper installed on RLModule subclasses to capture the
    GTrXL trunk's output features without modifying the parent's _forward_pass.

    Mechanism: register a forward hook on `self.gtrxl` that stores its first
    output (the per-timestep feature tensor of shape (B, T, d_model)). The
    parent's _forward_pass continues to consume the same features for
    policy/value heads; we just stash a reference for our Q-head computation.
    """

    def _install_feature_capture(self) -> None:
        self._captured_features: Optional[torch.Tensor] = None

        def _hook(_module, _inputs, output):
            # GTrXL.forward returns (features, memories). Take only features.
            if isinstance(output, tuple) and len(output) >= 1:
                self._captured_features = output[0]
            else:
                self._captured_features = output

        self.gtrxl.register_forward_hook(_hook)


class GTrXLEnsembleMaskedActionRLModule(_GTrXLFeatureCapture, GTrXLMaskedActionRLModule):
    """
    Local-agent RLModule with the V+Q hybrid critic.

    Inherits the full policy + V-head pipeline from GTrXLMaskedActionRLModule
    (PPO sees no behavioral change), then adds:
      - `self.q_heads`: EnsembleQHeads instance over the same trunk features
      - `_forward_train` emits `crd_q_ensemble` of shape (B, T, K, A)
      - `compute_q_ensemble(batch, action)` for the M2 counterfactual callback
    """

    @override(TorchRLModule)
    def setup(self):
        super().setup()
        cfg = _read_crd_ensemble_config(self.model_config)
        self.q_heads = EnsembleQHeads(
            d_model=self.gtrxl.d_model,
            action_dim=self.action_dim,
            K=cfg["K"],
            prior_lambda=cfg["prior_lambda"],
            hidden_dim=cfg["hidden_dim"],
        )
        self._install_feature_capture()
        logger.info(
            f"[{self.__class__.__name__}] Q-ensemble: K={cfg['K']}, "
            f"hidden_dim={cfg['hidden_dim']}, prior_lambda={cfg['prior_lambda']}, "
            f"action_dim={self.action_dim}, d_model={self.gtrxl.d_model}"
        )

    def _q_ensemble_from_features(
        self, features: torch.Tensor, last_only: bool = False
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, d_model) — captured trunk output
            last_only: if True, slice to the last timestep before computing Q
        Returns:
            (B, K, A) if last_only else (B, T, K, A)
        """
        if last_only:
            return self.q_heads(features[:, -1, :])  # (B, K, A)
        B, T, d = features.shape
        q_flat = self.q_heads(features.reshape(B * T, d))   # (B*T, K, A)
        return q_flat.reshape(B, T, self.q_heads.K, self.q_heads.action_dim)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        out = super()._forward_train(batch, **kwargs)
        # Features were captured during super()._forward_train via the hook.
        if self._captured_features is None:
            raise RuntimeError(
                "GTrXL feature capture missed; was self.gtrxl invoked during "
                "the parent's _forward_pass? Check the forward hook."
            )
        out[COL_Q_ENSEMBLE] = self._q_ensemble_from_features(
            self._captured_features, last_only=False
        )  # (B, T, K, A)
        return out

    def compute_q_ensemble(
        self, batch: Dict[str, Any], action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the ensemble Q at a specific action, used by the M2 callback.

        Args:
            batch:  RLlib-style single-step batch (will be passed through the
                    trunk to get state_repr); shape conventions match what
                    RLlib gives _forward_inference.
            action: (B,) integer tensor in [0, action_dim).

        Returns:
            mu:  (B,)  ensemble-mean Q at the chosen action.
            var: (B,)  ensemble-variance Q at the chosen action.
        """
        # Run the trunk via the parent's _forward_pass; we only need features.
        # _forward_pass also runs policy_head/value_head, but the cost is
        # dominated by the GTrXL trunk so this is fine.
        self._captured_features = None
        with torch.no_grad():
            self._forward_pass(batch)
        if self._captured_features is None:
            raise RuntimeError("Feature capture failed during compute_q_ensemble.")
        # Use the last timestep (single-step query semantics).
        state_repr = self._captured_features[:, -1, :]
        return self.q_heads.compute_q_for_action(state_repr, action)

    def get_non_inference_attributes(self):
        # q_heads aren't needed for pure rollout inference (only V-head + policy).
        return super().get_non_inference_attributes() + ["q_heads"]


class GTrXLEnsembleGlobalRLModule(_GTrXLFeatureCapture, GTrXLGlobalRLModule):
    """
    Global-routing-agent RLModule with the V+Q hybrid critic.

    Action space here is MultiDiscrete with `nvec = [num_dc] * batch_size`
    (each routed cloudlet picks a DC). The Q-head ensemble models per-cloudlet
    *marginal* Q values: output shape (B, K, batch_size, num_dc). The
    "scalar Q for the whole routing decision" is the mean of per-cloudlet Q
    at the chosen actions; this keeps ensemble variance comparable across
    different batch sizes.
    """

    @override(TorchRLModule)
    def setup(self):
        super().setup()
        # Confirm the action space layout we depend on.
        from gymnasium import spaces  # imported here to avoid top-level cycle
        if not isinstance(self.action_space, spaces.MultiDiscrete):
            raise NotImplementedError(
                f"{self.__class__.__name__} requires a MultiDiscrete action "
                f"space; got {type(self.action_space).__name__}"
            )
        nvec = self.action_space.nvec
        if nvec.size == 0 or not (nvec == nvec[0]).all():
            raise NotImplementedError(
                f"Q-head ensemble currently assumes uniform per-cloudlet "
                f"num_dc; got nvec={nvec.tolist()}. Generalising would "
                f"require a ragged Q-output shape — leave for later."
            )
        self.crd_batch_size: int = int(nvec.size)   # number of cloudlets per step
        self.crd_num_dc: int = int(nvec[0])         # actions per cloudlet

        cfg = _read_crd_ensemble_config(self.model_config)
        self.q_heads = EnsembleQHeads(
            d_model=self.gtrxl.d_model,
            action_dim=self.crd_batch_size * self.crd_num_dc,
            K=cfg["K"],
            prior_lambda=cfg["prior_lambda"],
            hidden_dim=cfg["hidden_dim"],
        )
        self._install_feature_capture()
        logger.info(
            f"[{self.__class__.__name__}] Q-ensemble: K={cfg['K']}, "
            f"hidden_dim={cfg['hidden_dim']}, prior_lambda={cfg['prior_lambda']}, "
            f"batch_size={self.crd_batch_size}, num_dc={self.crd_num_dc}, "
            f"d_model={self.gtrxl.d_model}"
        )

    def _q_ensemble_from_features(
        self, features: torch.Tensor, last_only: bool = False
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, d_model)
        Returns:
            last_only=True:  (B, K, batch_size, num_dc)
            last_only=False: (B, T, K, batch_size, num_dc)
        """
        if last_only:
            q_flat = self.q_heads(features[:, -1, :])  # (B, K, batch*nd)
            return q_flat.view(
                -1, self.q_heads.K, self.crd_batch_size, self.crd_num_dc
            )
        B, T, d = features.shape
        q_flat = self.q_heads(features.reshape(B * T, d))  # (B*T, K, batch*nd)
        return q_flat.view(
            B, T, self.q_heads.K, self.crd_batch_size, self.crd_num_dc
        )

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        out = super()._forward_train(batch, **kwargs)
        if self._captured_features is None:
            raise RuntimeError(
                "GTrXL feature capture missed during global-agent _forward_train."
            )
        out[COL_Q_ENSEMBLE] = self._q_ensemble_from_features(
            self._captured_features, last_only=False
        )  # (B, T, K, batch_size, num_dc)
        return out

    def compute_q_ensemble(
        self, batch: Dict[str, Any], action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the ensemble Q at a per-cloudlet routing decision.

        Args:
            batch: RLlib-style single-step batch (passed through trunk).
            action: (B, batch_size) integer tensor; action[b, i] ∈ [0, num_dc)
                    is the DC chosen for cloudlet i in sample b.

        Returns:
            mu:  (B,)  ensemble-mean of the per-cloudlet-mean Q.
            var: (B,)  ensemble-variance of the same scalar.
        """
        if action.dim() != 2:
            raise ValueError(
                f"global compute_q_ensemble expects (B, batch_size) action, "
                f"got shape {tuple(action.shape)}"
            )
        if action.shape[-1] != self.crd_batch_size:
            raise ValueError(
                f"action last dim {action.shape[-1]} != batch_size "
                f"{self.crd_batch_size}"
            )

        self._captured_features = None
        with torch.no_grad():
            self._forward_pass(batch)
        if self._captured_features is None:
            raise RuntimeError("Feature capture failed during compute_q_ensemble.")

        state_repr = self._captured_features[:, -1, :]  # (B, d_model)
        q_full = self.q_heads(state_repr).view(
            -1, self.q_heads.K, self.crd_batch_size, self.crd_num_dc
        )  # (B, K, batch_size, num_dc)

        # Gather along last dim using per-cloudlet action.
        idx = action.long().unsqueeze(1).unsqueeze(-1)  # (B, 1, batch_size, 1)
        idx = idx.expand(-1, self.q_heads.K, -1, 1)     # (B, K, batch_size, 1)
        q_per_cloudlet = q_full.gather(-1, idx).squeeze(-1)  # (B, K, batch_size)

        # Aggregate per cloudlet → scalar Q for the routing decision.
        # `mean` keeps the magnitude independent of batch_size.
        q_scalar = q_per_cloudlet.mean(dim=-1)  # (B, K)
        mu = q_scalar.mean(dim=1)
        var = q_scalar.var(dim=1, unbiased=False)
        return mu, var

    def get_non_inference_attributes(self):
        return super().get_non_inference_attributes() + ["q_heads"]
