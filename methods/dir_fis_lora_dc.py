import logging

import numpy as np
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
        self.dc_head_lr = max(1e-6, _as_float(args.get("decision_head_lr", self.dc_lr), default=self.dc_lr))
        self.dc_head_weight = max(
            0.0, _as_float(args.get("decision_head_retention_weight", 1.0), default=1.0)
        )
        self.dc_score_weight = max(0.0, _as_float(args.get("decision_score_weight", 1.0), default=1.0))
        self.dc_group_weight = max(0.0, _as_float(args.get("decision_group_weight", 1.0), default=1.0))
        self.dc_group_margin = _as_float(args.get("decision_group_margin", 0.05), default=0.05)
        self.dc_optimize_head_bias = _as_bool(args.get("decision_optimize_head_bias", False), default=False)
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

        # Persistent decision state H^(t): per-group temperature and bias.
        self._decision_tau = [1.0 for _ in range(self.sessions)]
        self._decision_bias = [0.0 for _ in range(self.sessions)]

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
        # Snapshot decision state and classifier heads for rollback/teacher signals.
        weight_snap = [model.classifier_pool[h].weight.detach().clone() for h in range(num_heads)]
        bias_snap = [model.classifier_pool[h].bias.detach().clone() for h in range(num_heads)]
        teacher_weights = [w.detach().clone() for w in weight_snap]
        teacher_tau = torch.tensor(self._decision_tau[:num_heads], device=device, dtype=torch.float32)
        teacher_bias = torch.tensor(self._decision_bias[:num_heads], device=device, dtype=torch.float32)

        tau = nn.Parameter(teacher_tau.clone())
        bias = nn.Parameter(teacher_bias.clone())

        param_groups = [
            {
                "params": [tau, bias],
                "lr": self.dc_lr,
                "weight_decay": self.dc_weight_decay,
            }
        ]

        head_params = []
        for h in range(num_heads):
            head_params.append(model.classifier_pool[h].weight)
            if self.dc_optimize_head_bias:
                head_params.append(model.classifier_pool[h].bias)
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": self.dc_head_lr,
                    "weight_decay": self.dc_weight_decay,
                }
            )

        optimizer = torch.optim.Adam(param_groups)

        pre_proxy = self._estimate_dc_proxy(
            model=model,
            num_heads=num_heads,
            head_start=head_start,
            head_end=head_end,
            head_width=head_width,
            old_classes=old_classes,
            classifier_weights=weight_snap,
            tau_values=teacher_tau,
            bias_values=teacher_bias,
            teacher_weights=teacher_weights,
            teacher_tau=teacher_tau,
            teacher_bias=teacher_bias,
        )

        running = {"task": 0.0, "head": 0.0, "score": 0.0, "group": 0.0, "n": 0}

        for _ in range(self.dc_epochs):
            for batch_idx, (_, inputs, targets) in enumerate(self.train_loader):
                if self.dc_max_batches is not None and batch_idx >= self.dc_max_batches:
                    break

                inputs, targets = inputs.to(device), targets.to(device)
                targets_local = targets - self.known_classes
                if targets_local.numel() == 0:
                    continue
                if targets_local.min().item() < 0 or targets_local.max().item() >= head_width:
                    continue

                with torch.no_grad():
                    features = self._extract_normalized_features(model, inputs, use_buffer=True)
                features = features.detach()

                student_weights = [model.classifier_pool[h].weight for h in range(num_heads)]
                logits = self._decision_logits_from_features(features, student_weights, tau, bias)
                if logits.size(1) != total_logits:
                    continue

                current_logits = logits[:, head_start:head_end]
                loss_task = F.cross_entropy(current_logits, targets_local)
                loss = loss_task

                loss_head = torch.tensor(0.0, device=device)
                if old_classes > 0 and self.dc_head_weight > 0.0:
                    head_terms = []
                    for h in range(num_heads - 1):
                        w_cur = F.normalize(model.classifier_pool[h].weight, dim=1)
                        w_ref = F.normalize(weight_snap[h], dim=1).to(w_cur.device, dtype=w_cur.dtype)
                        head_terms.append(F.mse_loss(w_cur, w_ref))
                    if head_terms:
                        loss_head = torch.stack(head_terms).mean()
                        loss = loss + self.dc_head_weight * loss_head

                loss_score = torch.tensor(0.0, device=device)
                if old_classes > 0 and self.dc_score_weight > 0.0:
                    with torch.no_grad():
                        teacher_logits = self._decision_logits_from_features(
                            features,
                            teacher_weights,
                            teacher_tau,
                            teacher_bias,
                        )
                    student_old = logits[:, :old_classes]
                    teacher_old = teacher_logits[:, :old_classes]
                    log_p = F.log_softmax(student_old / self.dc_tau, dim=1)
                    q = F.softmax(teacher_old / self.dc_tau, dim=1)
                    loss_score = F.kl_div(log_p, q, reduction="batchmean") * (self.dc_tau**2)
                    loss = loss + self.dc_score_weight * loss_score

                loss_group = torch.tensor(0.0, device=device)
                if old_classes > 0 and self.dc_group_weight > 0.0:
                    old_max = logits[:, :old_classes].max(dim=1).values
                    y_logit = logits.gather(1, targets.view(-1, 1)).squeeze(1)
                    margin_gap = F.relu(old_max - y_logit + self.dc_group_margin)
                    loss_group = margin_gap.pow(2).mean()
                    loss = loss + self.dc_group_weight * loss_group

                if self.dc_reg > 0.0:
                    loss = loss + self.dc_reg * ((tau - 1.0).pow(2).mean() + bias.pow(2).mean())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    tau.clamp_(self.dc_scale_min, self.dc_scale_max)
                    bias.clamp_(self.dc_bias_min, self.dc_bias_max)

                running["task"] += float(loss_task.item())
                running["head"] += float(loss_head.item())
                running["score"] += float(loss_score.item())
                running["group"] += float(loss_group.item())
                running["n"] += 1

                if self.debug:
                    break

        with torch.no_grad():
            final_tau = tau.detach().clamp(self.dc_scale_min, self.dc_scale_max)
            final_bias = bias.detach().clamp(self.dc_bias_min, self.dc_bias_max)

        post_weights = [model.classifier_pool[h].weight.detach() for h in range(num_heads)]
        post_proxy = self._estimate_dc_proxy(
            model=model,
            num_heads=num_heads,
            head_start=head_start,
            head_end=head_end,
            head_width=head_width,
            old_classes=old_classes,
            classifier_weights=post_weights,
            tau_values=final_tau,
            bias_values=final_bias,
            teacher_weights=teacher_weights,
            teacher_tau=teacher_tau,
            teacher_bias=teacher_bias,
        )

        pre_ce = float("nan") if pre_proxy is None else pre_proxy["ce"]
        post_ce = float("nan") if post_proxy is None else post_proxy["ce"]
        old_kd = float("nan") if post_proxy is None else post_proxy["old_kd"]

        if self.dc_guardrail_enabled:
            guard_violations = []
            max_tau_delta = float((final_tau - 1.0).abs().max().item())
            max_bias_abs = float(final_bias.abs().max().item())
            if max_tau_delta > self.dc_guard_scale_delta:
                guard_violations.append(f"max|tau-1|={max_tau_delta:.4f}>{self.dc_guard_scale_delta:.4f}")
            if max_bias_abs > self.dc_guard_bias_abs:
                guard_violations.append(f"max|bias|={max_bias_abs:.4f}>{self.dc_guard_bias_abs:.4f}")
            if pre_proxy is not None and post_proxy is not None and post_ce > pre_ce * (1.0 + self.dc_guard_ce_tolerance):
                guard_violations.append(
                    f"post_ce={post_ce:.4f}>pre_ce*{(1.0 + self.dc_guard_ce_tolerance):.4f}"
                )

            if guard_violations:
                with torch.no_grad():
                    for h in range(num_heads):
                        model.classifier_pool[h].weight.copy_(weight_snap[h])
                        model.classifier_pool[h].bias.copy_(bias_snap[h])
                logging.warning(
                    "Task %s decision consolidation rolled back => head %s, pre_ce %.4f, post_ce %.4f, "
                    "old_kd %.6f, reason: %s",
                    self.cur_task,
                    head_idx,
                    pre_ce,
                    post_ce,
                    old_kd,
                    "; ".join(guard_violations),
                )
                return

        for h in range(num_heads):
            self._decision_tau[h] = float(final_tau[h].item())
            self._decision_bias[h] = float(final_bias[h].item())

        denom = max(running["n"], 1)
        logging.info(
            "Task %s decision consolidation => head %s, tau_mean %.4f, tau_min %.4f, tau_max %.4f, "
            "bias_mean %.4f, pre_ce %.4f, post_ce %.4f, old_kd %.6f, "
            "L_task %.4f, L_head %.4f, L_score %.4f, L_group %.4f",
            self.cur_task,
            head_idx,
            float(final_tau.mean().item()),
            float(final_tau.min().item()),
            float(final_tau.max().item()),
            float(final_bias.mean().item()),
            pre_ce,
            post_ce,
            old_kd,
            running["task"] / denom,
            running["head"] / denom,
            running["score"] / denom,
            running["group"] / denom,
        )

    def _extract_normalized_features(self, model, inputs, use_buffer=True):
        feats = model.image_encoder(inputs, use_buffer=use_buffer)
        feats = feats[:, 0, :].reshape(feats.size(0), -1)
        return F.normalize(feats, dim=1)

    @staticmethod
    def _decision_logits_from_features(features, classifier_weights, tau_values, bias_values):
        tau = torch.as_tensor(tau_values, device=features.device, dtype=features.dtype)
        bias = torch.as_tensor(bias_values, device=features.device, dtype=features.dtype)
        logits = []
        for h, w in enumerate(classifier_weights):
            w_norm = F.normalize(w, dim=1)
            head_logits = F.linear(features, w_norm)
            head_logits = tau[h] * head_logits + bias[h]
            logits.append(head_logits)
        return torch.cat(logits, dim=1)

    def _decision_interface(self, inputs, use_buffer=True):
        model = self._unwrap_model()
        num_heads = self.cur_task + 1
        features = self._extract_normalized_features(model, inputs, use_buffer=use_buffer)
        weights = [model.classifier_pool[h].weight for h in range(num_heads)]
        tau = torch.tensor(self._decision_tau[:num_heads], device=features.device, dtype=features.dtype)
        bias = torch.tensor(self._decision_bias[:num_heads], device=features.device, dtype=features.dtype)
        return self._decision_logits_from_features(features, weights, tau, bias)

    def _estimate_dc_proxy(
        self,
        model,
        num_heads,
        head_start,
        head_end,
        head_width,
        old_classes,
        classifier_weights,
        tau_values,
        bias_values,
        teacher_weights,
        teacher_tau,
        teacher_bias,
    ):
        max_batches = self.dc_guard_eval_batches
        if max_batches is None:
            max_batches = self.dc_max_batches
        if max_batches is None:
            max_batches = 10

        ce_sum = 0.0
        kd_sum = 0.0
        used_batches = 0

        with torch.no_grad():
            for batch_idx, (_, inputs, targets) in enumerate(self.train_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets_local = targets - self.known_classes
                if targets_local.numel() == 0:
                    continue
                if targets_local.min().item() < 0 or targets_local.max().item() >= head_width:
                    continue

                features = self._extract_normalized_features(model, inputs, use_buffer=True)
                logits = self._decision_logits_from_features(features, classifier_weights, tau_values, bias_values)
                if logits.size(1) != num_heads * head_width:
                    continue

                current_logits = logits[:, head_start:head_end]
                ce_sum += float(F.cross_entropy(current_logits, targets_local).item())

                if old_classes > 0:
                    teacher_logits = self._decision_logits_from_features(
                        features,
                        teacher_weights,
                        teacher_tau,
                        teacher_bias,
                    )
                    student_old = logits[:, :old_classes]
                    teacher_old = teacher_logits[:, :old_classes]
                    log_p = F.log_softmax(student_old / self.dc_tau, dim=1)
                    q = F.softmax(teacher_old / self.dc_tau, dim=1)
                    kd_sum += float((F.kl_div(log_p, q, reduction="batchmean") * (self.dc_tau**2)).item())

                used_batches += 1
                if self.debug:
                    break

        if used_batches == 0:
            return None
        return {
            "ce": ce_sum / used_batches,
            "old_kd": kd_sum / used_batches if old_classes > 0 else 0.0,
        }

    def _test(self, test_loader):
        self.network.eval()

        y_pred, y_true = [], []
        y_pred_with_task = []
        y_task_pred, y_task_true = [], []

        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs = self._decision_interface(inputs, use_buffer=True)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)

            if self.init_cls == self.increment:
                self.class_num = self.increment
            task_ids = torch.div(predicts, self.class_num, rounding_mode="trunc")
            y_task_pred.append(task_ids.cpu())
            y_task_true.append((torch.div(targets, self.class_num, rounding_mode="trunc")).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:, : self.class_num]
            target_task_ids = torch.div(targets, self.class_num, rounding_mode="trunc")
            for idx, task_id in enumerate(target_task_ids):
                start = self.class_num * task_id
                end = self.class_num * (task_id + 1)
                outputs_with_task[idx] = outputs[idx, start:end]

            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + target_task_ids * self.class_num

            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return (
            np.concatenate(y_pred),
            np.concatenate(y_pred_with_task),
            np.concatenate(y_true),
            torch.cat(y_task_pred),
            torch.cat(y_task_true),
        )


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
