import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from methods.base import BaseLearner
from models.net_directional_lora import Net
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency


class DirectionalLoRA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self.topk = 1
        self.network = Net(args)

        self.gamma = args["gamma"]
        self.reg_weight = args["lambda"]
        self.importance_decay = args.get("importance_decay", self.gamma)
        self.novelty_threshold = args["novelty_threshold"]
        self.grow_rank = args.get("grow_rank", 1)
        self.warmup_batches = args.get("warmup_batches", 1)
        self.basis_growth_batches = args.get("basis_growth_batches", max(self.warmup_batches, 10))

        self.diag_enabled = args.get("diag_enabled", True)
        self.diag_topk = args.get("diag_topk", 3)
        self.diag_perturb_scale = args.get("diag_perturb_scale", 0.1)
        self.diag_old_new_split = args.get("diag_old_new_split", True)

        self.count_updates = 0
        self._historical_rank_snapshot = None
        self._postgrow_rank_snapshot = None
        self._pretrain_old_metrics = None

    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        self._historical_rank_snapshot = self.network.get_active_ranks()

        if self.diag_enabled and self.cur_task > 0:
            self._run_pretraining_diagnostics()

        with torch.enable_grad():
            self._warmup_and_expand_basis(train_loader)
        self._postgrow_rank_snapshot = self.network.get_active_ranks()

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

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                mask = (targets >= self.known_classes).nonzero().view(-1)
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self.known_classes

                logits = self.network(inputs, use_buffer=True)["logits"]
                loss = F.cross_entropy(logits, targets)

                if self.count_updates > 0:
                    loss = loss + self.reg_weight * self.network.directional_regularization(self.device)

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

    def after_task(self):
        if self.diag_enabled and self.cur_task > 0:
            self._run_posttraining_diagnostics()

        self._update_importance()
        self.network.consolidate_task(self.gamma)
        self.count_updates += 1
        super().after_task()

    def freeze_network(self):
        target_suffix = f".{self.cur_task}"
        unfrozen_keys = [
            f"classifier_pool{target_suffix}",
            "lora_buffer_k",
            "lora_buffer_v",
        ]
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
                "Task %s warm-up novelty mean %.4f over %s batches",
                self.cur_task,
                float(sum(novelty_scores) / len(novelty_scores)),
                batches_used,
            )

    def _update_importance(self):
        fisher = FisherComputer(
            self.network,
            self.train_loader,
            self.known_classes,
            F.cross_entropy,
            self.device,
        )
        fisher_values = fisher.compute(max_batches=None if not self.debug else 1)
        self.network.update_importance(fisher_values, self.importance_decay)

    def _run_pretraining_diagnostics(self):
        old_loader = self._build_old_task_loader()
        if old_loader is None:
            return

        baseline = self._evaluate_old_tasks(old_loader, use_buffer=False)
        self._pretrain_old_metrics = baseline

        stats = self.network.collect_direction_stats(max_rank_map=self._historical_rank_snapshot)
        if not stats:
            return

        epsilon = self._select_perturb_epsilon(stats)
        for stat in stats:
            delta = self._measure_direction_sensitivity(old_loader, stat, epsilon, baseline)
            stat.update(delta)

        fisher_weights = np.asarray([item["fisher_weight"] for item in stats], dtype=np.float64)
        energy_weights = np.asarray([item["energy_weight"] for item in stats], dtype=np.float64)
        sensitivities = np.asarray([item["loss_delta"] for item in stats], dtype=np.float64)

        payload = {
            "task": self.cur_task,
            "stage": "pretrain",
            "old_task_loss": baseline["loss"],
            "old_task_acc": baseline["acc"],
            "epsilon": epsilon,
            "num_directions": len(stats),
            "fisher_vs_sensitivity": _correlation_summary(fisher_weights, sensitivities),
            "energy_vs_sensitivity": _correlation_summary(energy_weights, sensitivities),
            "top_bottom": _top_bottom_summary(stats, self.diag_topk),
        }
        self._record_diagnostics(payload)

    def _run_posttraining_diagnostics(self):
        old_loader = self._build_old_task_loader()
        if old_loader is None or self._pretrain_old_metrics is None:
            return

        baseline_loss = self._pretrain_old_metrics["loss"]
        baseline_acc = self._pretrain_old_metrics["acc"]

        full_metrics = self._evaluate_old_tasks(old_loader, use_buffer=True)
        tracked_metrics = self._evaluate_old_tasks(old_loader, use_buffer=True, rank_limits=self._historical_rank_snapshot)

        payload = {
            "task": self.cur_task,
            "stage": "posttrain",
            "old_task_loss_before": baseline_loss,
            "old_task_loss_full": full_metrics["loss"],
            "old_task_loss_tracked_only": tracked_metrics["loss"],
            "old_task_acc_before": baseline_acc,
            "old_task_acc_full": full_metrics["acc"],
            "old_task_acc_tracked_only": tracked_metrics["acc"],
        }

        full_loss_delta = full_metrics["loss"] - baseline_loss
        tracked_loss_delta = tracked_metrics["loss"] - baseline_loss
        full_acc_drop = baseline_acc - full_metrics["acc"]
        tracked_acc_drop = baseline_acc - tracked_metrics["acc"]

        payload["fraction_forgetting_explained_loss"] = _safe_fraction(tracked_loss_delta, full_loss_delta)
        payload["fraction_forgetting_explained_acc"] = _safe_fraction(tracked_acc_drop, full_acc_drop)

        if self.diag_old_new_split and self._historical_rank_snapshot is not None and self._postgrow_rank_snapshot is not None:
            old_windows = self._build_rank_windows(self._historical_rank_snapshot, self._historical_rank_snapshot)
            new_windows = self._build_rank_windows(self._historical_rank_snapshot, self._postgrow_rank_snapshot)

            old_only_metrics = self._evaluate_old_tasks(
                old_loader,
                use_buffer=True,
                rank_windows=old_windows,
            )
            new_only_metrics = self._evaluate_old_tasks(
                old_loader,
                use_buffer=True,
                rank_windows=new_windows,
            )

            old_only_loss_delta = old_only_metrics["loss"] - baseline_loss
            new_only_loss_delta = new_only_metrics["loss"] - baseline_loss
            old_only_acc_drop = baseline_acc - old_only_metrics["acc"]
            new_only_acc_drop = baseline_acc - new_only_metrics["acc"]

            payload.update(
                {
                    "old_task_loss_old_only": old_only_metrics["loss"],
                    "old_task_loss_new_only": new_only_metrics["loss"],
                    "old_task_acc_old_only": old_only_metrics["acc"],
                    "old_task_acc_new_only": new_only_metrics["acc"],
                    "old_only_loss_delta": old_only_loss_delta,
                    "new_only_loss_delta": new_only_loss_delta,
                    "old_only_acc_drop": old_only_acc_drop,
                    "new_only_acc_drop": new_only_acc_drop,
                    "interaction_loss_delta": full_loss_delta - old_only_loss_delta - new_only_loss_delta,
                    "interaction_acc_drop": full_acc_drop - old_only_acc_drop - new_only_acc_drop,
                }
            )

        self._record_diagnostics(payload)

    def _build_old_task_loader(self):
        if self.cur_task <= 0:
            return None
        old_classes = np.arange(0, self.known_classes)
        dataset = self.data_manager.get_dataset(old_classes, source="test", mode="test")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def _evaluate_old_tasks(self, loader, use_buffer, rank_limits=None, rank_windows=None):
        self.network.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.network.interface(
                    inputs,
                    use_buffer=use_buffer,
                    rank_limits=rank_limits,
                    rank_windows=rank_windows,
                    max_task=self.cur_task - 1,
                )
                loss = F.cross_entropy(outputs, targets, reduction="sum")

                total_loss += loss.item()
                total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
                total_samples += targets.size(0)

                if self.debug:
                    break

        if total_samples == 0:
            return {"loss": 0.0, "acc": 0.0}
        return {"loss": total_loss / total_samples, "acc": 100.0 * total_correct / total_samples}

    def _build_rank_windows(self, start_ranks, end_ranks):
        windows = {}
        for layer_idx, end_rank in end_ranks.items():
            start_rank = min(start_ranks.get(layer_idx, 0), end_rank)
            windows[layer_idx] = (start_rank, end_rank)
        return windows

    def _measure_direction_sensitivity(self, loader, stat, epsilon, baseline):
        module = self.network.get_attention_module(stat["module_index"])
        tensor = module.lora_memory_k if stat["kind"] == "k" else module.lora_memory_v
        direction_index = stat["direction_index"]

        with torch.no_grad():
            original = tensor[:, direction_index].detach().clone()
            direction = module.get_direction_vector(stat["kind"], direction_index)
            tensor[:, direction_index].copy_(original + epsilon * direction)

        perturbed = self._evaluate_old_tasks(loader, use_buffer=False)

        with torch.no_grad():
            tensor[:, direction_index].copy_(original)

        return {
            "loss_delta": perturbed["loss"] - baseline["loss"],
            "acc_drop": baseline["acc"] - perturbed["acc"],
        }

    def _select_perturb_epsilon(self, stats):
        norms = [item["coeff_norm"] for item in stats if item["coeff_norm"] > 0]
        if not norms:
            return self.diag_perturb_scale
        return self.diag_perturb_scale * float(np.mean(norms))

    def _record_diagnostics(self, payload):
        logging.info("Diagnostics: %s", json.dumps(payload, sort_keys=True))
        if "logfilename" in self.args:
            path = self.args["logfilename"] + "_diagnostics.jsonl"
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


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


def _correlation_summary(weights, sensitivities):
    return {
        "pearson": _pearson(weights, sensitivities),
        "spearman": _spearman(weights, sensitivities),
    }


def _pearson(x, y):
    if len(x) < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    if len(x) < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _rankdata(x):
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _safe_fraction(numerator, denominator):
    if abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def _top_bottom_summary(stats, topk):
    ordered = sorted(stats, key=lambda item: item["fisher_weight"], reverse=True)
    k = min(topk, len(ordered))
    top = ordered[:k]
    bottom = ordered[-k:]
    return {
        "topk": k,
        "top_loss_delta_mean": float(np.mean([item["loss_delta"] for item in top])),
        "bottom_loss_delta_mean": float(np.mean([item["loss_delta"] for item in bottom])),
        "top_acc_drop_mean": float(np.mean([item["acc_drop"] for item in top])),
        "bottom_acc_drop_mean": float(np.mean([item["acc_drop"] for item in bottom])),
    }
