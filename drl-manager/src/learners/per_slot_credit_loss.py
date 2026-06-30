"""Tier-1 per-slot credit PPO learner.

The global routing action is MultiDiscrete([n_choices] x 128), but only ~4 of the 128 slots
carry a real cloudlet on a typical step — the other ~124 are PADDING (no cloudlet). Standard PPO
forms ONE joint log-prob = Σ over ALL 128 slots and ONE ratio = exp(ΔΣ logp). With 124 padding
slots, that joint ratio is extreme/noisy → PPO clipping bites hard → the actor is throttled and
stays ≈uniform (the long-standing "actor won't converge" symptom).

Fix (mask_padding): recompute per-slot log-probs and SUM ONLY OVER VALID SLOTS. A padded slot is
identified directly from the observation — it has batch_cloudlet_mi == 0 (verified: the nonzero-mi
count matches the Java routing_action_count exactly every step). The ratio then depends on the ~4
REAL decisions only → tame ratio → less clipping → the actor can move. The entropy bonus is likewise
restricted to valid slots (so it doesn't reward keeping padding uniform).

Gated per-module by model_config["per_slot_credit"]["enabled"] (+ "mask_padding", default True).
OFF, or on a non-MultiDiscrete module (the local Discrete scheduler), or when every slot is valid →
the masked log-prob/entropy equal the joint ones → BIT-IDENTICAL to NormalizedCriticPPOTorchLearner.
"""
from typing import Any, Dict

from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.typing import ModuleID, TensorType

from src.learners.normalized_critic_loss import (
    NormalizedCriticPPOTorchLearner,
    LEARNER_RESULTS_VF_TARGET_VAR_EMA_KEY,
    LEARNER_RESULTS_VF_LOSS_RAW_MSE_KEY,
    _DEFAULT_EMA_DECAY,
    _DEFAULT_VAR_EPS,
)

torch, _ = try_import_torch()


def _read_per_slot_config(module) -> Dict[str, Any]:
    mc = getattr(module, "model_config", None) or {}
    cfg = mc.get("per_slot_credit", {}) if isinstance(mc, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


class PerSlotCreditPPOTorchLearner(NormalizedCriticPPOTorchLearner):
    """PPO learner that masks PADDING slots out of the global router's joint log-prob/entropy."""

    @override(NormalizedCriticPPOTorchLearner)
    def compute_loss_for_module(
        self, *, module_id: ModuleID, config: PPOConfig, batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        module = self.module[module_id].unwrapped()
        ps_cfg = _read_per_slot_config(module)
        n_choices = int(ps_cfg.get("n_choices", 0)) or int(getattr(module, "num_action_choices", 0) or 0)
        n_slots = int(getattr(module, "num_batch_slots", 0) or 0)
        valid = self._valid_slot_mask(batch, n_slots) if (n_choices > 0 and n_slots > 0) else None

        if not (ps_cfg.get("enabled", False) and ps_cfg.get("mask_padding", True)) or valid is None:
            return super().compute_loss_for_module(
                module_id=module_id, config=config, batch=batch, fwd_out=fwd_out)

        # ===== identical to NormalizedCriticPPOTorchLearner.compute_loss_for_module, EXCEPT
        # ===== logp_ratio and curr_entropy are summed over VALID slots only =====
        if Columns.LOSS_MASK in batch:
            loss_mask = batch[Columns.LOSS_MASK]
            num_valid = torch.sum(loss_mask)

            def possibly_masked_mean(d):
                return torch.sum(d[loss_mask]) / num_valid
        else:
            loss_mask = None
            possibly_masked_mean = torch.mean

        actions = batch[Columns.ACTIONS]
        logp_new = self._masked_logp(fwd_out[Columns.ACTION_DIST_INPUTS], actions, n_slots, n_choices, valid)
        logp_old = self._masked_logp(batch[Columns.ACTION_DIST_INPUTS], actions, n_slots, n_choices, valid)
        logp_ratio = torch.exp(logp_new - logp_old)                                  # <-- masked
        curr_entropy = self._masked_entropy(fwd_out[Columns.ACTION_DIST_INPUTS], n_slots, n_choices, valid)  # <-- masked

        if config.use_kl_loss:
            curr_dist = module.get_train_action_dist_cls().from_logits(fwd_out[Columns.ACTION_DIST_INPUTS])
            prev_dist = module.get_exploration_action_dist_cls().from_logits(batch[Columns.ACTION_DIST_INPUTS])
            mean_kl_loss = possibly_masked_mean(prev_dist.kl(curr_dist))
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        mean_entropy = possibly_masked_mean(curr_entropy)
        surrogate_loss = torch.min(
            batch[Postprocessing.ADVANTAGES] * logp_ratio,
            batch[Postprocessing.ADVANTAGES] * torch.clamp(
                logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
        )

        if config.use_critic:
            value_fn_out = module.compute_values(batch, embeddings=fwd_out.get(Columns.EMBEDDINGS))
            norm_cfg = self._read_module_norm_config(module_id)
            targets = batch[Postprocessing.VALUE_TARGETS]
            vf_err2 = torch.pow(value_fn_out - targets, 2.0)
            if bool(norm_cfg.get("enabled", False)):
                decay = float(norm_cfg.get("ema_decay", _DEFAULT_EMA_DECAY))
                var_eps = float(norm_cfg.get("var_eps", _DEFAULT_VAR_EPS))
                var_ema = self._update_target_var_ema(module_id, self._masked_var(targets, loss_mask), decay)
                denom = max(var_ema, var_eps)
            else:
                var_ema = None
                denom = 1.0
            vf_loss = vf_err2 / denom
            vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
            mean_vf_loss = possibly_masked_mean(vf_loss_clipped)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss)
            mean_vf_raw_mse = possibly_masked_mean(vf_err2)
        else:
            z = torch.tensor(0.0, device=surrogate_loss.device)
            value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z
            mean_vf_raw_mse = None
            var_ema = None

        total_loss = possibly_masked_mean(
            -surrogate_loss
            + config.vf_loss_coeff * vf_loss_clipped
            - self.entropy_coeff_schedulers_per_module[module_id].get_current_value() * curr_entropy
        )
        if config.use_kl_loss:
            total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss

        extra_vf_metrics = {}
        if var_ema is not None:
            extra_vf_metrics[LEARNER_RESULTS_VF_TARGET_VAR_EMA_KEY] = var_ema
            extra_vf_metrics[LEARNER_RESULTS_VF_LOSS_RAW_MSE_KEY] = mean_vf_raw_mse

        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                VF_LOSS_KEY: mean_vf_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS], value_fn_out),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: mean_kl_loss,
                **extra_vf_metrics,
            },
            key=module_id,
            window=1,
        )
        return total_loss

    # ----- helpers -----
    def _valid_slot_mask(self, batch, n_slots):
        obs = batch.get(Columns.OBS)
        inner = obs.get("observation", obs) if isinstance(obs, dict) else None
        mi = inner.get("batch_cloudlet_mi") if isinstance(inner, dict) else None
        if mi is None:
            return None
        mi = mi if torch.is_tensor(mi) else torch.as_tensor(mi)
        if mi.shape[-1] != n_slots:
            return None
        return (mi > 0).to(torch.float32)

    @staticmethod
    def _reshape_logits(logits, n_slots, n_choices):
        return logits.reshape(*logits.shape[:-1], n_slots, n_choices)

    def _masked_logp(self, logits, actions, n_slots, n_choices, valid):
        lg = self._reshape_logits(logits, n_slots, n_choices)
        logsm = torch.log_softmax(lg, dim=-1)
        a = actions.reshape(*lg.shape[:-1]).long().unsqueeze(-1)
        per_slot = logsm.gather(-1, a).squeeze(-1)
        return (per_slot * valid).sum(-1)

    def _masked_entropy(self, logits, n_slots, n_choices, valid):
        lg = self._reshape_logits(logits, n_slots, n_choices)
        logsm = torch.log_softmax(lg, dim=-1)
        ent_slot = -(logsm.exp() * logsm).sum(-1)
        return (ent_slot * valid).sum(-1)
