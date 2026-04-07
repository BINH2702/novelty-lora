import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from methods.base import BaseLearner
from methods.dir_fis_lora import DirFisLoRA


class DirFisLoRADC(DirFisLoRA):
    """DirFisLoRA + lightweight decision consolidation on the current task head."""

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
        self.dc_bias_min = _as_float(args.get("decision_bias_min", -0.05), default=-0.05)
        self.dc_bias_max = _as_float(args.get("decision_bias_max", 0.05), default=0.05)
        if self.dc_bias_min > self.dc_bias_max:
            self.dc_bias_min, self.dc_bias_max = self.dc_bias_max, self.dc_bias_min
        self.dc_max_batches = _as_optional_int(args.get("decision_consolidation_max_batches"), default=20)
        self.dc_guardrail_enabled = _as_bool(args.get("decision_guardrail_enabled", True), default=True)
        self.dc_guard_scale_delta = max(
            0.0,
            _as_float(args.get("decision_guard_max_scale_delta", 0.1), default=0.1),
        )
        self.dc_guard_bias_abs = max(
            0.0,
            _as_float(args.get("decision_guard_max_bias_abs", 0.5), default=0.5),
        )
        self.dc_guard_ce_tolerance = max(
            0.0,
            _as_float(args.get("decision_guard_ce_tolerance", 0.05), default=0.05),
        )
        self.dc_guard_eval_batches = _as_optional_int(args.get("decision_guard_eval_batches"), default=10)

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
        head_idx = self.cur_task
        head_start = head_idx * head_width
        head_end = head_start + head_width

        # Calibrate only the newly added head to avoid repeatedly modifying old heads.
        scale = nn.Parameter(torch.tensor(1.0, device=device))
        bias = nn.Parameter(torch.tensor(0.0, device=device))
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

                targets_local = targets - self.known_classes
                if targets_local.numel() == 0:
                    continue
                if targets_local.min().item() < 0 or targets_local.max().item() >= head_width:
                    continue

                current_base = base_logits[:, head_start:head_end]
                current_cal = current_base * scale + bias

                # Optimize CE only on current-head logits to avoid new-vs-old competition on new-only data.
                loss_ce = F.cross_entropy(current_cal, targets_local)
                loss = loss_ce

                if old_classes > 0 and self.dc_kd_weight > 0.0:
                    cal_logits = torch.cat(
                        [base_logits[:, :head_start], current_cal, base_logits[:, head_end:]],
                        dim=1,
                    )
                    # Preserve global decision distribution while calibrating the current head.
                    log_p = F.log_softmax(cal_logits / self.dc_tau, dim=1)
                    q = F.softmax(base_logits / self.dc_tau, dim=1)
                    loss_kd = F.kl_div(log_p, q, reduction="batchmean") * (self.dc_tau ** 2)
                    loss = loss + self.dc_kd_weight * loss_kd

                if self.dc_reg > 0.0:
                    loss = loss + self.dc_reg * ((scale - 1.0).pow(2) + bias.pow(2))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    scale.clamp_(self.dc_scale_min, self.dc_scale_max)
                    bias.clamp_(self.dc_bias_min, self.dc_bias_max)

                if self.debug:
                    break

        with torch.no_grad():
            final_scale = float(scale.detach().clamp(self.dc_scale_min, self.dc_scale_max).item())
            final_bias = float(bias.detach().clamp(self.dc_bias_min, self.dc_bias_max).item())

        proxy = self._estimate_dc_proxy(
            model=model,
            total_logits=total_logits,
            head_start=head_start,
            head_end=head_end,
            head_width=head_width,
            old_classes=old_classes,
            scale_value=final_scale,
            bias_value=final_bias,
        )

        pre_ce = float("nan")
        post_ce = float("nan")
        old_kd = float("nan")
        if proxy is not None:
            pre_ce = proxy["pre_ce"]
            post_ce = proxy["post_ce"]
            old_kd = proxy["old_kd"]

        if self.dc_guardrail_enabled:
            guard_violations = []
            if abs(final_scale - 1.0) > self.dc_guard_scale_delta:
                guard_violations.append(
                    f"|scale-1|={abs(final_scale - 1.0):.4f}>{self.dc_guard_scale_delta:.4f}"
                )
            if abs(final_bias) > self.dc_guard_bias_abs:
                guard_violations.append(f"|bias|={abs(final_bias):.4f}>{self.dc_guard_bias_abs:.4f}")
            if proxy is not None and post_ce > pre_ce * (1.0 + self.dc_guard_ce_tolerance):
                guard_violations.append(
                    f"post_ce={post_ce:.4f}>pre_ce*{(1.0 + self.dc_guard_ce_tolerance):.4f}"
                )

            if guard_violations:
                logging.warning(
                    "Task %s decision consolidation skipped => head %s, scale %.4f, bias %.4f, "
                    "pre_ce %.4f, post_ce %.4f, old_kd %.6f, reason: %s",
                    self.cur_task,
                    head_idx,
                    final_scale,
                    final_bias,
                    pre_ce,
                    post_ce,
                    old_kd,
                    "; ".join(guard_violations),
                )
                return

        with torch.no_grad():
            classifier = model.classifier_pool[head_idx]
            s = torch.as_tensor(final_scale, device=classifier.weight.device, dtype=classifier.weight.dtype)
            b = torch.as_tensor(final_bias, device=classifier.bias.device, dtype=classifier.bias.dtype)
            classifier.weight.mul_(s)
            classifier.bias.mul_(s).add_(b)

        logging.info(
            "Task %s decision consolidation => head %s, scale %.4f, bias %.4f, pre_ce %.4f, post_ce %.4f, old_kd %.6f",
            self.cur_task,
            head_idx,
            final_scale,
            final_bias,
            pre_ce,
            post_ce,
            old_kd,
        )

    def _estimate_dc_proxy(
        self,
        model,
        total_logits,
        head_start,
        head_end,
        head_width,
        old_classes,
        scale_value,
        bias_value,
    ):
        max_batches = self.dc_guard_eval_batches
        if max_batches is None:
            max_batches = self.dc_max_batches
        if max_batches is None:
            max_batches = 10

        pre_ce = 0.0
        post_ce = 0.0
        old_kd = 0.0
        used_batches = 0

        with torch.no_grad():
            for batch_idx, (_, inputs, targets) in enumerate(self.train_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                base_logits = model.interface(inputs, use_buffer=True)
                if base_logits.size(1) != total_logits:
                    continue

                targets_local = targets - self.known_classes
                if targets_local.numel() == 0:
                    continue
                if targets_local.min().item() < 0 or targets_local.max().item() >= head_width:
                    continue

                current_base = base_logits[:, head_start:head_end]
                scale_t = torch.as_tensor(scale_value, device=current_base.device, dtype=current_base.dtype)
                bias_t = torch.as_tensor(bias_value, device=current_base.device, dtype=current_base.dtype)
                current_cal = current_base * scale_t + bias_t

                pre_ce += float(F.cross_entropy(current_base, targets_local).item())
                post_ce += float(F.cross_entropy(current_cal, targets_local).item())

                if old_classes > 0 and self.dc_kd_weight > 0.0:
                    cal_logits = torch.cat(
                        [base_logits[:, :head_start], current_cal, base_logits[:, head_end:]],
                        dim=1,
                    )
                    log_p = F.log_softmax(cal_logits / self.dc_tau, dim=1)
                    q = F.softmax(base_logits / self.dc_tau, dim=1)
                    old_kd += float((F.kl_div(log_p, q, reduction="batchmean") * (self.dc_tau ** 2)).item())

                used_batches += 1
                if self.debug:
                    break

        if used_batches == 0:
            return None

        return {
            "pre_ce": pre_ce / used_batches,
            "post_ce": post_ce / used_batches,
            "old_kd": old_kd / used_batches if old_classes > 0 else 0.0,
        }


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
