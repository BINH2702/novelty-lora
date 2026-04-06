import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from methods.base import BaseLearner
from methods.dir_fis_lora import DirFisLoRA


class DirFisLoRADC(DirFisLoRA):
    """DirFisLoRA + lightweight decision consolidation via per-head affine calibration."""

    def __init__(self, args):
        super().__init__(args)

        self.decision_consolidation_enabled = _as_bool(
            args.get("decision_consolidation_enabled", True),
            default=True,
        )
        self.dc_epochs = max(1, int(_as_float(args.get("decision_consolidation_epochs", 1), default=1)))
        self.dc_lr = max(1e-6, _as_float(args.get("decision_consolidation_lr", 1e-2), default=1e-2))
        self.dc_weight_decay = max(0.0, _as_float(args.get("decision_consolidation_weight_decay", 0.0), default=0.0))
        self.dc_kd_weight = max(0.0, _as_float(args.get("decision_consolidation_kd_weight", 0.5), default=0.5))
        self.dc_tau = max(1e-3, _as_float(args.get("decision_consolidation_tau", 2.0), default=2.0))
        self.dc_reg = max(0.0, _as_float(args.get("decision_consolidation_reg", 1e-3), default=1e-3))
        self.dc_scale_min = max(0.05, _as_float(args.get("decision_scale_min", 0.5), default=0.5))
        self.dc_scale_max = max(self.dc_scale_min, _as_float(args.get("decision_scale_max", 1.5), default=1.5))
        self.dc_max_batches = _as_optional_int(args.get("decision_consolidation_max_batches"), default=20)

    def after_task(self):
        gate_strength = self.conflict_gate_strength if self.conflict_gate_enabled and self.count_updates > 0 else 0.0
        self.network.consolidate_task(
            self.gamma,
            historical_rank_map=self._historical_rank_snapshot,
            conflict_gate_strength=gate_strength,
            conflict_gate_floor=self.conflict_gate_floor,
            basis_update_mode=self.basis_update_mode,
            novelty_threshold=self.novelty_threshold,
            importance_aware_consolidation=self.importance_aware_consolidation,
            protected_slots=self.protected_slots,
            protected_slots_ratio=self.protected_slots_ratio,
            importance_transport=self.importance_transport,
        )
        rank_map = self.network.get_active_ranks()
        if rank_map:
            rank_values = list(rank_map.values())
            logging.info(
                "Task %s consolidated active rank => mean %.2f, min %s, max %s",
                self.cur_task,
                float(sum(rank_values) / len(rank_values)),
                int(min(rank_values)),
                int(max(rank_values)),
            )
        if self.alpha_adaptive and self.count_updates > 0:
            logging.info(
                "Task %s adaptive alpha => overlap %.4f, alpha_t %.4f",
                self.cur_task,
                self._alpha_overlap,
                self._alpha_task,
            )

        self._update_importance()
        self._decision_consolidation()

        self.count_updates += 1
        BaseLearner.after_task(self)

    def _decision_consolidation(self):
        if not self.decision_consolidation_enabled:
            return
        if self.cur_task <= 0:
            return
        if not hasattr(self, "train_loader") or self.train_loader is None:
            return

        model = self._unwrap_model()
        model.eval()
        device = self.device

        num_heads = self.cur_task + 1
        if num_heads <= 0:
            return

        head_width = model.classifier_pool[0].out_features
        total_logits = num_heads * head_width
        old_classes = self.known_classes

        scale = nn.Parameter(torch.ones(num_heads, device=device))
        bias = nn.Parameter(torch.zeros(num_heads, device=device))
        optimizer = torch.optim.Adam(
            [scale, bias],
            lr=self.dc_lr,
            weight_decay=self.dc_weight_decay,
        )

        for _ in range(self.dc_epochs):
            for batch_idx, (_, inputs, targets) in enumerate(self.train_loader):
                if self.dc_max_batches is not None and batch_idx >= self.dc_max_batches:
                    break

                inputs, targets = inputs.to(device), targets.to(device)
                with torch.no_grad():
                    base_logits = model.interface(inputs, use_buffer=True)
                if base_logits.size(1) != total_logits:
                    continue

                cal_logits = self._apply_head_affine(base_logits, scale, bias, head_width)

                loss_ce = F.cross_entropy(cal_logits, targets)
                loss = loss_ce

                if old_classes > 0 and self.dc_kd_weight > 0.0:
                    old_cal = cal_logits[:, :old_classes]
                    old_base = base_logits[:, :old_classes]
                    log_p = F.log_softmax(old_cal / self.dc_tau, dim=1)
                    q = F.softmax(old_base / self.dc_tau, dim=1)
                    loss_kd = F.kl_div(log_p, q, reduction="batchmean") * (self.dc_tau ** 2)
                    loss = loss + self.dc_kd_weight * loss_kd

                if self.dc_reg > 0.0:
                    loss = loss + self.dc_reg * ((scale - 1.0).pow(2).mean() + bias.pow(2).mean())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    scale.clamp_(self.dc_scale_min, self.dc_scale_max)

                if self.debug:
                    break

        with torch.no_grad():
            final_scale = scale.detach().clamp(self.dc_scale_min, self.dc_scale_max).cpu()
            final_bias = bias.detach().cpu()
            for head_idx in range(num_heads):
                classifier = model.classifier_pool[head_idx]
                s = final_scale[head_idx].to(classifier.weight.device, dtype=classifier.weight.dtype)
                b = final_bias[head_idx].to(classifier.bias.device, dtype=classifier.bias.dtype)
                classifier.weight.mul_(s)
                classifier.bias.mul_(s).add_(b)

        logging.info(
            "Task %s decision consolidation => heads %s, scale_mean %.4f, scale_min %.4f, scale_max %.4f",
            self.cur_task,
            num_heads,
            float(final_scale.mean().item()),
            float(final_scale.min().item()),
            float(final_scale.max().item()),
        )

    @staticmethod
    def _apply_head_affine(logits, scale, bias, head_width):
        bsz = logits.size(0)
        num_heads = scale.numel()
        view = logits.view(bsz, num_heads, head_width)
        calibrated = view * scale.view(1, num_heads, 1) + bias.view(1, num_heads, 1)
        return calibrated.reshape(bsz, num_heads * head_width)


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "f", "no", "n", "off"}:
            return False
    return default


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_optional_int(value, default=None):
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
