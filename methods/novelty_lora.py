import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm

from methods.base import BaseLearner
from models.net_novelty_lora import Net
from utils.toolkit import tensor2numpy
from utils.toolkit import print_trainable_params, check_params_consistency


class NoveltyLoRA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self.topk = 1
        self.network = Net(args)

        self.gamma = args["gamma"]
        self.ewc_weight = args["lambda"]
        self.novelty_threshold = args["novelty_threshold"]
        self.grow_rank = args.get("grow_rank", 1)
        self.warmup_batches = args.get("warmup_batches", 1)
        self.count_updates = 0

    def _train(self, train_loader):
        self.network.to(self.device)
        self.freeze_network()
        print_trainable_params(self.network)

        with torch.enable_grad():
            self._warmup_and_expand_basis(train_loader)

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
                    loss = loss + self.ewc_weight * self.network.novelty_regularization(self.device)

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
        self._update_fisher()
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
        novelty_scores = []
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

            for module in self.network.image_encoder.modules():
                if hasattr(module, "apply_warmup_gradient"):
                    novelty_scores.append(module.apply_warmup_gradient(self.grow_rank, self.novelty_threshold))

            batches_used += 1
            self.network.zero_grad()
            if self.debug or batches_used >= self.warmup_batches:
                break

        for name, param in self.network.named_parameters():
            param.requires_grad_(saved_flags[name])

        if novelty_scores:
            logging.info(
                "Task %s warm-up novelty mean %.4f",
                self.cur_task,
                float(sum(novelty_scores) / len(novelty_scores)),
            )

    def _update_fisher(self):
        fisher = FisherComputer(
            self.network,
            self.train_loader,
            self.known_classes,
            self.increment,
            F.cross_entropy,
            self.device,
        )
        fisher_values = fisher.compute(max_batches=None if not self.debug else 1)
        self.network.update_fisher(fisher_values)


class FisherComputer:
    def __init__(self, network, dataloader, known_classes, increment, criterion, device):
        self.model = network.to(device)
        self.dataloader = dataloader
        self.known_classes = known_classes
        self.increment = increment
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
