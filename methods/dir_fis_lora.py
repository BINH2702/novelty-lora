import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm

from methods.base import BaseLearner
from models.net_dir_fis_lora import Net
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency


class DirFisLoRA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self.topk = 1
        self.network = Net(args)

        self.gamma = args["gamma"]
        self.reg_weight = args["lambda"]
        self.importance_decay = args.get("importance_decay", self.gamma)
        self.basis_update_mode = str(args.get("basis_update_mode", "realized_drift")).lower()
        self.novelty_threshold = args["novelty_threshold"]
        self.grow_rank = args.get("grow_rank", 1)
        self.warmup_batches = args.get("warmup_batches", 1)
        self.basis_growth_batches = args.get("basis_growth_batches", max(self.warmup_batches, 10))

        self.reg_lambda_min = max(0.0, _as_float(args.get("reg_lambda_min", 0.2), default=0.2))
        self.reg_alpha = max(0.0, _as_float(args.get("reg_alpha", 1.0), default=1.0))
        self.reg_w_tau = max(0.0, _as_float(args.get("reg_w_tau", 0.5), default=0.5))
        self.reg_w_max = max(0.0, _as_float(args.get("reg_w_max", 5.0), default=5.0))
        self.reg_new_dir_weight = max(0.0, _as_float(args.get("reg_new_dir_weight", 0.0), default=0.0))
        self.importance_floor_frac = max(0.0, _as_float(args.get("importance_floor_frac", 0.0), default=0.0))
        self.alpha_adaptive = _as_bool(args.get("alpha_adaptive", False), default=False)
        self.alpha_probe_batches = _as_optional_int(args.get("alpha_probe_batches"), default=2) or 2
        self.alpha_probe_use_buffer = _as_bool(args.get("alpha_probe_use_buffer", False), default=False)
        self.alpha_probe_ema = min(
            0.999,
            max(0.0, _as_float(args.get("alpha_probe_ema", 0.0), default=0.0)),
        )
        self.alpha_overlap_tau = max(1e-6, _as_float(args.get("alpha_overlap_tau", 0.5), default=0.5))
        self.alpha_scale_min = max(0.0, _as_float(args.get("alpha_scale_min", 0.5), default=0.5))
        self.alpha_scale_max = max(self.alpha_scale_min, _as_float(args.get("alpha_scale_max", 2.0), default=2.0))
        self.alpha_overlap_eps = max(1e-12, _as_float(args.get("alpha_overlap_eps", 1e-12), default=1e-12))

        self.importance_aware_consolidation = _as_bool(
            args.get("importance_aware_consolidation", False),
            default=False,
        )
        self.protected_slots = max(0, int(_as_float(args.get("protected_slots", 0), default=0)))
        self.protected_slots_ratio = max(0.0, _as_float(args.get("protected_slots_ratio", 0.0), default=0.0))

        gate_flag = args.get("conflict_gate_enabled", False)
        self.conflict_gate_enabled = _as_bool(gate_flag, default=False)
        self.conflict_gate_strength = max(0.0, _as_float(args.get("conflict_gate_strength", 0.5), default=0.5))
        self.conflict_gate_floor = _as_float(args.get("conflict_gate_floor", 0.1), default=0.1)

        self.fisher_max_batches = _as_optional_int(args.get("fisher_max_batches"), default=None)

        self.count_updates = 0
        self._historical_rank_snapshot = None
        self._alpha_task = self.reg_alpha
        self._alpha_overlap = 0.0

    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        self._historical_rank_snapshot = self.network.get_active_ranks()
        if self.basis_update_mode == "realized_drift" and self.count_updates == 0:
            self._historical_rank_snapshot = {idx: 0 for idx in self._historical_rank_snapshot}

        with torch.enable_grad():
            if self.basis_update_mode == "realized_drift":
                self.network.prepare_task_drift()
            else:
                self._warmup_and_expand_basis(train_loader)
        self._alpha_task = self._compute_task_alpha(train_loader)

        encoder_params = self.network.image_encoder.parameters()
        cls_params = [p for p in self.network.classifier_pool.parameters() if p.requires_grad]

        if len(self.multiple_gpus) > 1:
            self.network = nn.DataParallel(self.network, self.multiple_gpus)

        encoder_params = {"params": encoder_params, "lr": self.lrate, "weight_decay": self.weight_decay}
        cls_params = {"params": cls_params, "lr": self.fc_lrate, "weight_decay": self.weight_decay}
        optimizer, scheduler = self.build_optimizer([encoder_params, cls_params])
        check_params_consistency(self.network, optimizer)

        self._train_function(train_loader, optimizer, scheduler)

        if len(self.multiple_gpus) > 1:
            self.network = self.network.module

    def _train_function(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self.network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes

                logits = self.network(inputs, use_buffer=True)["logits"]
                loss = F.cross_entropy(logits, targets)

                if self.count_updates > 0:
                    model_ref = self._unwrap_model()
                    loss = loss + self.reg_weight * model_ref.dirfis_regularization(
                        self.device,
                        historical_rank_map=self._historical_rank_snapshot,
                        lambda_min=self.reg_lambda_min,
                        alpha=self._alpha_task,
                        weight_power=self.reg_w_tau,
                        weight_cap=self.reg_w_max,
                        new_dir_weight=self.reg_new_dir_weight,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                if self.debug:
                    break

            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self.cur_task, epoch + 1, self.epochs, losses / max(len(train_loader), 1), train_acc
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _unwrap_model(self):
        return self.network.module if isinstance(self.network, nn.DataParallel) else self.network

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
        self.count_updates += 1
        super().after_task()

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        unfrozen_keys = [f"classifier_pool{target_suffix}"]
        if self.basis_update_mode == "realized_drift":
            unfrozen_keys.extend(
                [
                    "task_memory_k",
                    "task_memory_v",
                    "task_basis_k",
                    "task_basis_v",
                ]
            )
        else:
            unfrozen_keys.extend(
                [
                    "lora_buffer_k",
                    "lora_buffer_v",
                ]
            )
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in unfrozen_keys))

    def _warmup_and_expand_basis(self, train_loader):
        self.network.zero_grad()
        self.network.train()
        modules = list(self.network.iter_attention_modules())

        saved_flags = {}
        for name, param in self.network.named_parameters():
            saved_flags[name] = param.requires_grad
            param.requires_grad_(False)

        for module in self.network.image_encoder.modules():
            if hasattr(module, "qkv"):
                module.qkv.weight.requires_grad_(True)
                if module.qkv.bias is not None:
                    module.qkv.bias.requires_grad_(True)
        for param in self.network.classifier_pool[self.cur_task].parameters():
            param.requires_grad_(True)

        batches_used = 0
        grad_sums = [
            {"module": module, "grad_k": None, "grad_v": None, "count": 0}
            for module in modules
        ]
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            mask = (targets >= self.known_classes).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            targets = torch.index_select(targets, 0, mask) - self.known_classes

            logits = self.network(inputs, use_buffer=False)["logits"]
            loss = F.cross_entropy(logits, targets)
            loss.backward()

            for slot in grad_sums:
                module = slot["module"]
                if module.qkv.weight.grad is None:
                    continue
                grad = module.qkv.weight.grad.detach()
                grad_k = grad[module.dim : 2 * module.dim].clone()
                grad_v = grad[2 * module.dim :].clone()
                slot["grad_k"] = grad_k if slot["grad_k"] is None else slot["grad_k"] + grad_k
                slot["grad_v"] = grad_v if slot["grad_v"] is None else slot["grad_v"] + grad_v
                slot["count"] += 1

            batches_used += 1
            self.network.zero_grad()
            if self.debug or batches_used >= self.basis_growth_batches:
                break

        for name, param in self.network.named_parameters():
            param.requires_grad_(saved_flags[name])

        novelty_scores = []
        for slot in grad_sums:
            if slot["count"] == 0:
                continue
            novelty_scores.append(
                slot["module"].apply_accumulated_warmup_gradient(
                    slot["grad_k"] / slot["count"],
                    slot["grad_v"] / slot["count"],
                    self.grow_rank,
                    self.novelty_threshold,
                )
            )

        if novelty_scores:
            logging.info(
                "Task %s warm-up novelty mean %.4f",
                self.cur_task,
                float(sum(novelty_scores) / len(novelty_scores)),
            )

    def _compute_task_alpha(self, train_loader):
        if self.count_updates <= 0:
            self._alpha_overlap = 0.0
            return self.reg_alpha
        if not self.alpha_adaptive:
            self._alpha_overlap = 0.0
            return self.reg_alpha

        overlap = self._estimate_probe_overlap(train_loader)
        if self.alpha_probe_ema > 0.0:
            overlap = self.alpha_probe_ema * self._alpha_overlap + (1.0 - self.alpha_probe_ema) * overlap
        self._alpha_overlap = overlap

        scale = overlap / self.alpha_overlap_tau
        scale = float(np.clip(scale, self.alpha_scale_min, self.alpha_scale_max))
        return self.reg_alpha * scale

    def _estimate_probe_overlap(self, train_loader):
        self.network.zero_grad()
        self.network.train()
        modules = list(self.network.iter_attention_modules())

        saved_flags = {}
        for name, param in self.network.named_parameters():
            saved_flags[name] = param.requires_grad
            param.requires_grad_(False)

        for module in self.network.image_encoder.modules():
            if hasattr(module, "qkv"):
                module.qkv.weight.requires_grad_(True)
                if module.qkv.bias is not None:
                    module.qkv.bias.requires_grad_(True)
        for param in self.network.classifier_pool[self.cur_task].parameters():
            param.requires_grad_(True)

        overlap_num = 0.0
        overlap_den = 0.0
        batches_used = 0
        try:
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                if mask.numel() == 0:
                    continue
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes

                logits = self.network(inputs, use_buffer=self.alpha_probe_use_buffer)["logits"]
                loss = F.cross_entropy(logits, targets)
                loss.backward()

                for module_idx, module in enumerate(modules):
                    if module.qkv.weight.grad is None:
                        continue
                    grad = module.qkv.weight.grad.detach()
                    grad_k = grad[module.dim : 2 * module.dim]
                    grad_v = grad[2 * module.dim :]

                    historical_rank = self._historical_rank_snapshot.get(module_idx, 0)
                    historical_rank = max(0, min(int(historical_rank), int(module.active_rank)))
                    if historical_rank <= 0:
                        overlap_den += float(torch.sum(grad_k.pow(2)).item() + torch.sum(grad_v.pow(2)).item())
                        continue

                    basis_k = module.lora_basis_k[:historical_rank, :].detach()
                    basis_v = module.lora_basis_v[:historical_rank, :].detach()

                    proj_k = grad_k @ basis_k.t() @ basis_k
                    proj_v = grad_v @ basis_v.t() @ basis_v

                    overlap_num += float(torch.sum(proj_k.pow(2)).item() + torch.sum(proj_v.pow(2)).item())
                    overlap_den += float(torch.sum(grad_k.pow(2)).item() + torch.sum(grad_v.pow(2)).item())

                batches_used += 1
                self.network.zero_grad()
                if batches_used >= self.alpha_probe_batches:
                    break
        finally:
            self.network.zero_grad()
            for name, param in self.network.named_parameters():
                param.requires_grad_(saved_flags[name])

        if overlap_den <= 0.0:
            return 0.0
        overlap = overlap_num / (overlap_den + self.alpha_overlap_eps)
        return float(np.clip(overlap, 0.0, 1.0))

    def _update_importance(self):
        max_batches = self.fisher_max_batches
        if self.debug:
            max_batches = 1 if max_batches is None else min(max_batches, 1)

        fisher = FisherComputer(
            self.network,
            self.train_loader,
            self.known_classes,
            F.cross_entropy,
            self.device,
        )
        fisher_values = fisher.compute(max_batches=max_batches)
        with torch.no_grad():
            self.network.update_importance(
                fisher_values,
                self.importance_decay,
                floor_frac=self.importance_floor_frac,
            )


class FisherComputer:
    def __init__(self, network, dataloader, known_classes, criterion, device):
        self.model = network.to(device)
        self.dataloader = dataloader
        self.known_classes = known_classes
        self.criterion = criterion
        self.device = device

    def compute(self, max_batches=None):
        fisher = []
        for module in self.model.image_encoder.modules():
            if hasattr(module, "init_fisher_storage"):
                fisher.extend(module.init_fisher_storage())

        self.model.eval()
        num_samples = 0
        for i, (_, inputs, targets) in enumerate(tqdm(self.dataloader, desc="Computing Fisher")):
            if max_batches is not None and i >= max_batches:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            mask = (targets >= self.known_classes).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            targets = torch.index_select(targets, 0, mask) - self.known_classes

            self.model.zero_grad()
            logits = self.model(inputs, use_buffer=True, register_hook=True)["logits"]
            loss = self.criterion(logits, targets)
            loss.backward()

            batch_size = inputs.size(0)
            num_samples += batch_size
            idx = 0
            for module in self.model.image_encoder.modules():
                if hasattr(module, "delta_w_k_grad"):
                    fisher[idx] += (module.delta_w_k_grad.detach() ** 2) * batch_size
                    idx += 1
                    fisher[idx] += (module.delta_w_v_grad.detach() ** 2) * batch_size
                    idx += 1

        if num_samples == 0:
            return fisher
        return [item / num_samples for item in fisher]


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
