import math

import torch
import torch.nn as nn

from models.vit_ewclora import (
    VisionTransformer as BaseVisionTransformer,
    PatchEmbed,
    LayerScale,
    build_model_with_cfg,
    resolve_pretrained_cfg,
    checkpoint_filter_fn,
    Mlp,
    DropPath,
)


class AttentionDirFisLoRA(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        r=4,
        rank_budget=10,
        max_rank=20,
        task_rank=2,
        enforce_rank_budget=True,
        n_tasks=10,
    ):
        super().__init__()
        del n_tasks
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.init_rank = r
        self.rank_budget = max(rank_budget, r)
        self.max_rank = max(max_rank, self.rank_budget)
        self.task_rank = max(int(task_rank), 1)
        self.enforce_rank_budget = bool(enforce_rank_budget)
        self.active_rank = r

        self.lora_basis_k = nn.Parameter(torch.zeros(self.max_rank, dim), requires_grad=False)
        self.lora_basis_v = nn.Parameter(torch.zeros(self.max_rank, dim), requires_grad=False)
        self.lora_memory_k = nn.Parameter(torch.zeros(dim, self.max_rank), requires_grad=False)
        self.lora_memory_v = nn.Parameter(torch.zeros(dim, self.max_rank), requires_grad=False)
        self.lora_buffer_k = nn.Parameter(torch.zeros(dim, self.max_rank))
        self.lora_buffer_v = nn.Parameter(torch.zeros(dim, self.max_rank))
        self.task_memory_k = nn.Parameter(torch.zeros(dim, self.task_rank))
        self.task_memory_v = nn.Parameter(torch.zeros(dim, self.task_rank))
        self.task_basis_k = nn.Parameter(torch.zeros(self.task_rank, dim))
        self.task_basis_v = nn.Parameter(torch.zeros(self.task_rank, dim))

        self.register_buffer("importance_k", torch.zeros(self.max_rank))
        self.register_buffer("importance_v", torch.zeros(self.max_rank))
        self.register_buffer("importance_max_k", torch.zeros(self.max_rank))
        self.register_buffer("importance_max_v", torch.zeros(self.max_rank))
        self.register_buffer("fisher_k", torch.zeros(dim, dim))
        self.register_buffer("fisher_v", torch.zeros(dim, dim))
        self.use_task_drift = False

    def init_param(self):
        nn.init.kaiming_uniform_(self.lora_basis_k[: self.init_rank], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_basis_v[: self.init_rank], a=math.sqrt(5))
        self._orthonormalize_basis(self.lora_basis_k, self.init_rank)
        self._orthonormalize_basis(self.lora_basis_v, self.init_rank)
        nn.init.zeros_(self.lora_memory_k)
        nn.init.zeros_(self.lora_memory_v)
        nn.init.zeros_(self.lora_buffer_k)
        nn.init.zeros_(self.lora_buffer_v)
        self.importance_k.zero_()
        self.importance_v.zero_()
        self.importance_max_k.zero_()
        self.importance_max_v.zero_()
        self.reset_task_drift()
        self.use_task_drift = False

    def _orthonormalize_basis(self, basis, rank):
        if rank <= 0:
            return
        q, _ = torch.linalg.qr(basis[:rank].data.t(), mode="reduced")
        basis[:rank].data.copy_(q.t())

    def _sample_random_dirs(self, count, basis):
        if count <= 0:
            return torch.zeros(0, basis.size(1), device=basis.device, dtype=basis.dtype)
        rand_dirs = torch.randn(count, basis.size(1), device=basis.device, dtype=basis.dtype)
        q, _ = torch.linalg.qr(rand_dirs.t(), mode="reduced")
        return q.t()[:count]

    def activate_provisional_directions(self, grow_rank):
        add_rank = min(max(int(grow_rank), 0), self.max_rank - self.active_rank)
        if add_rank <= 0:
            return 0

        start = self.active_rank
        end = start + add_rank
        self.lora_basis_k.data[start:end].copy_(self._sample_random_dirs(add_rank, self.lora_basis_k))
        self.lora_basis_v.data[start:end].copy_(self._sample_random_dirs(add_rank, self.lora_basis_v))
        self._orthonormalize_basis(self.lora_basis_k, end)
        self._orthonormalize_basis(self.lora_basis_v, end)

        self.lora_memory_k.data[:, start:end].zero_()
        self.lora_memory_v.data[:, start:end].zero_()
        self.lora_buffer_k.data[:, start:end].zero_()
        self.lora_buffer_v.data[:, start:end].zero_()
        self.importance_k.data[start:end].zero_()
        self.importance_v.data[start:end].zero_()
        self.importance_max_k.data[start:end].zero_()
        self.importance_max_v.data[start:end].zero_()
        self.active_rank = end
        return add_rank

    def reset_task_drift(self):
        with torch.no_grad():
            nn.init.zeros_(self.task_memory_k)
            nn.init.zeros_(self.task_memory_v)
            self.task_basis_k.copy_(self._sample_random_dirs(self.task_rank, self.task_basis_k))
            self.task_basis_v.copy_(self._sample_random_dirs(self.task_rank, self.task_basis_v))
        self.use_task_drift = True

    def clear_task_drift(self):
        with torch.no_grad():
            self.task_memory_k.zero_()
            self.task_memory_v.zero_()
        self.use_task_drift = False

    def _delta_weight(self, basis, memory, buffer, use_buffer):
        rank = self.active_rank
        coeff = memory[:, :rank]
        if use_buffer:
            coeff = coeff + buffer[:, :rank]
        return coeff @ basis[:rank, :]

    def _task_delta_weight(self, memory, basis):
        if not self.use_task_drift:
            return torch.zeros(
                memory.size(0),
                basis.size(1),
                device=memory.device,
                dtype=memory.dtype,
            )
        return memory @ basis

    def _compute_novelty_and_residual(self, grad, basis, rank):
        grad_norm = torch.linalg.norm(grad)
        if grad_norm.item() == 0:
            return 0.0, grad

        residual = grad
        novelty = 1.0
        if rank > 0:
            active_basis = basis[:rank]
            proj = grad @ active_basis.t() @ active_basis
            residual = grad - proj
            novelty = (torch.linalg.norm(residual) / (grad_norm + 1e-12)).item()
        return novelty, residual

    def _extract_new_dirs(self, residual, add_rank):
        res_norm = torch.linalg.norm(residual)
        if res_norm.item() > 0:
            _, _, vh = torch.linalg.svd(residual, full_matrices=False)
            return vh[:add_rank]

        rand_dirs = torch.randn(add_rank, residual.size(1), device=residual.device, dtype=residual.dtype)
        q, _ = torch.linalg.qr(rand_dirs.t(), mode="reduced")
        return q.t()[:add_rank]

    def apply_warmup_gradient(self, grow_rank, threshold):
        if self.qkv.weight.grad is None:
            return 0.0

        grad = self.qkv.weight.grad.detach()
        grad_k = grad[self.dim : 2 * self.dim]
        grad_v = grad[2 * self.dim :]
        novelty = self.apply_accumulated_warmup_gradient(grad_k, grad_v, grow_rank, threshold)

        self.qkv.weight.grad = None
        if self.qkv.bias is not None:
            self.qkv.bias.grad = None
        return novelty

    def apply_accumulated_warmup_gradient(self, grad_k, grad_v, grow_rank, threshold):
        rank = self.active_rank
        novelty_k, residual_k = self._compute_novelty_and_residual(grad_k, self.lora_basis_k, rank)
        novelty_v, residual_v = self._compute_novelty_and_residual(grad_v, self.lora_basis_v, rank)
        novelty = 0.5 * (novelty_k + novelty_v)

        if novelty <= threshold or rank >= self.max_rank:
            return novelty

        add_rank = min(grow_rank, self.max_rank - rank)
        if add_rank <= 0:
            return novelty

        new_dirs_k = self._extract_new_dirs(residual_k, add_rank)
        new_dirs_v = self._extract_new_dirs(residual_v, add_rank)
        self.lora_basis_k.data[rank : rank + add_rank].copy_(new_dirs_k)
        self.lora_basis_v.data[rank : rank + add_rank].copy_(new_dirs_v)
        self._orthonormalize_basis(self.lora_basis_k, rank + add_rank)
        self._orthonormalize_basis(self.lora_basis_v, rank + add_rank)

        self.lora_memory_k.data[:, rank : rank + add_rank].zero_()
        self.lora_memory_v.data[:, rank : rank + add_rank].zero_()
        self.lora_buffer_k.data[:, rank : rank + add_rank].zero_()
        self.lora_buffer_v.data[:, rank : rank + add_rank].zero_()
        self.importance_k.data[rank : rank + add_rank].zero_()
        self.importance_v.data[rank : rank + add_rank].zero_()
        self.importance_max_k.data[rank : rank + add_rank].zero_()
        self.importance_max_v.data[rank : rank + add_rank].zero_()
        self.active_rank += add_rank
        return novelty

    def update_importance(self, fisher_k, fisher_v, decay, floor_frac=0.0):
        with torch.no_grad():
            self.fisher_k.copy_(fisher_k.to(self.fisher_k.device))
            self.fisher_v.copy_(fisher_v.to(self.fisher_v.device))

            rank = self.active_rank
            fisher_scores_k = self._directional_sensitivity_scores(self.fisher_k, self.lora_basis_k.detach(), rank)
            fisher_scores_v = self._directional_sensitivity_scores(self.fisher_v, self.lora_basis_v.detach(), rank)

            self.importance_k[:rank].mul_(decay).add_(fisher_scores_k, alpha=1.0 - decay)
            self.importance_v[:rank].mul_(decay).add_(fisher_scores_v, alpha=1.0 - decay)
            self.importance_max_k[:rank] = torch.maximum(self.importance_max_k[:rank], fisher_scores_k)
            self.importance_max_v[:rank] = torch.maximum(self.importance_max_v[:rank], fisher_scores_v)

            floor_frac = max(0.0, float(floor_frac))
            if floor_frac > 0.0:
                self.importance_k[:rank] = torch.maximum(
                    self.importance_k[:rank], floor_frac * self.importance_max_k[:rank]
                )
                self.importance_v[:rank] = torch.maximum(
                    self.importance_v[:rank], floor_frac * self.importance_max_v[:rank]
                )

    def _directional_sensitivity_scores(self, fisher, basis, rank):
        if rank <= 0:
            return torch.zeros(0, device=fisher.device)
        proj = torch.matmul(fisher, basis[:rank].t())
        scores = torch.sum(proj.pow(2), dim=0)
        return torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

    def regularization_loss(
        self,
        device,
        historical_rank=None,
        lambda_min=0.0,
        alpha=1.0,
        weight_power=1.0,
        weight_cap=0.0,
        new_dir_weight=0.0,
    ):
        rank = self.active_rank
        if rank == 0:
            return torch.tensor(0.0, device=device)

        historical_rank = rank if historical_rank is None else max(0, min(int(historical_rank), rank))
        lambda_min = max(0.0, float(lambda_min))
        alpha = max(0.0, float(alpha))
        weight_power = max(0.0, float(weight_power))
        weight_cap = max(0.0, float(weight_cap))
        new_dir_weight = max(0.0, float(new_dir_weight))

        if self.use_task_drift:
            task_delta_k = self._task_delta_weight(self.task_memory_k, self.task_basis_k)
            task_delta_v = self._task_delta_weight(self.task_memory_v, self.task_basis_v)
            importance_k = self.importance_k[:rank].detach().to(device).clamp(min=0.0)
            importance_v = self.importance_v[:rank].detach().to(device).clamp(min=0.0)

            penalty_k = self._branch_regularization_from_delta(
                task_delta_k,
                self.lora_basis_k[:rank, :],
                importance_k,
                historical_rank,
                lambda_min=lambda_min,
                alpha=alpha,
                weight_power=weight_power,
                weight_cap=weight_cap,
                new_dir_weight=new_dir_weight,
            )
            penalty_v = self._branch_regularization_from_delta(
                task_delta_v,
                self.lora_basis_v[:rank, :],
                importance_v,
                historical_rank,
                lambda_min=lambda_min,
                alpha=alpha,
                weight_power=weight_power,
                weight_cap=weight_cap,
                new_dir_weight=new_dir_weight,
            )
            return 0.5 * (penalty_k + penalty_v)

        coeff_k = self.lora_buffer_k[:, :rank]
        coeff_v = self.lora_buffer_v[:, :rank]
        importance_k = self.importance_k[:rank].detach().to(device).clamp(min=0.0)
        importance_v = self.importance_v[:rank].detach().to(device).clamp(min=0.0)

        penalty_k = self._branch_regularization(
            coeff_k,
            importance_k,
            historical_rank,
            lambda_min=lambda_min,
            alpha=alpha,
            weight_power=weight_power,
            weight_cap=weight_cap,
            new_dir_weight=new_dir_weight,
        )
        penalty_v = self._branch_regularization(
            coeff_v,
            importance_v,
            historical_rank,
            lambda_min=lambda_min,
            alpha=alpha,
            weight_power=weight_power,
            weight_cap=weight_cap,
            new_dir_weight=new_dir_weight,
        )
        return 0.5 * (penalty_k + penalty_v)

    def _branch_regularization_from_delta(
        self,
        delta,
        basis,
        importance,
        historical_rank,
        lambda_min,
        alpha,
        weight_power,
        weight_cap,
        new_dir_weight,
    ):
        penalty = torch.tensor(0.0, device=delta.device, dtype=delta.dtype)
        residual = delta

        if historical_rank > 0:
            old_basis = basis[:historical_rank, :]
            coeff = delta @ old_basis.t()
            energy = coeff.pow(2).sum(dim=0)
            old_importance = importance[:historical_rank]
            old_mean = torch.mean(old_importance) + 1e-12
            norm_importance = (old_importance / old_mean).pow(weight_power)
            if weight_cap > 0.0:
                norm_importance = torch.clamp(norm_importance, max=weight_cap)
            strength = lambda_min + alpha * norm_importance
            penalty = penalty + torch.sum(strength * energy)
            residual = delta - coeff @ old_basis

        if new_dir_weight > 0.0:
            penalty = penalty + new_dir_weight * torch.sum(residual.pow(2))

        return penalty

    def _branch_regularization(
        self,
        coeff,
        importance,
        historical_rank,
        lambda_min,
        alpha,
        weight_power,
        weight_cap,
        new_dir_weight,
    ):
        energy = coeff.pow(2).sum(dim=0)
        penalty = torch.tensor(0.0, device=coeff.device, dtype=coeff.dtype)

        if historical_rank > 0:
            old_importance = importance[:historical_rank]
            old_mean = torch.mean(old_importance) + 1e-12
            norm_importance = (old_importance / old_mean).pow(weight_power)
            if weight_cap > 0.0:
                norm_importance = torch.clamp(norm_importance, max=weight_cap)
            strength = lambda_min + alpha * norm_importance
            penalty = penalty + torch.sum(strength * energy[:historical_rank])

        if historical_rank < energy.numel() and new_dir_weight > 0.0:
            penalty = penalty + new_dir_weight * torch.sum(energy[historical_rank:])

        return penalty

    def _compute_new_rank_gates(
        self,
        memory,
        buffer,
        basis,
        importance,
        historical_rank,
        rank,
        conflict_gate_strength,
        conflict_gate_floor,
    ):
        new_coeff = buffer[:, historical_rank:rank]
        num_new = new_coeff.size(1)
        gates = torch.ones(num_new, device=new_coeff.device, dtype=new_coeff.dtype)

        if conflict_gate_strength <= 0.0 or historical_rank <= 0 or num_new == 0:
            return gates, None

        old_coeff = memory[:, :historical_rank]
        old_basis = basis[:historical_rank, :]
        old_importance = importance[:historical_rank].clamp(min=0.0).to(old_coeff.device)
        old_norms = torch.linalg.norm(old_coeff, dim=0)
        if torch.sum(old_importance).item() <= 0.0:
            old_importance = old_norms
        if torch.sum(old_importance).item() <= 0.0:
            return gates, None

        weights = old_importance / (torch.sum(old_importance) + 1e-12)
        old_delta = old_coeff @ old_basis
        old_atom_norms = torch.linalg.norm(old_coeff, dim=0) * torch.linalg.norm(old_basis, dim=1)
        old_scale = torch.sum(weights * old_atom_norms)
        if old_scale.item() <= 0.0:
            return gates, None

        new_basis = basis[historical_rank:rank, :]
        proj_old_on_new_basis = old_delta @ new_basis.t()
        overlap = torch.sum(new_coeff * proj_old_on_new_basis, dim=0)
        new_norms = torch.linalg.norm(new_coeff, dim=0) * torch.linalg.norm(new_basis, dim=1)
        old_norm = torch.linalg.norm(old_delta)

        cosine = overlap / (new_norms * old_norm + 1e-12)
        anti_alignment = torch.clamp(-cosine, min=0.0)
        magnitude = new_norms / (old_scale + 1e-12)
        conflict = anti_alignment * magnitude
        gates = torch.clamp(1.0 - conflict_gate_strength * conflict, min=conflict_gate_floor, max=1.0)

        stats = {
            "num_new_ranks": int(num_new),
            "gate_mean": float(gates.mean().item()),
            "gate_min": float(gates.min().item()),
            "conflict_mean": float(conflict.mean().item()),
            "conflict_max": float(conflict.max().item()),
        }
        return gates, stats

    def _compute_realized_drift_residual(self, basis, buffer, historical_rank, gamma):
        rank = self.active_rank
        delta = gamma * (buffer[:, :rank] @ basis[:rank, :])
        delta_norm = torch.linalg.norm(delta)
        if delta_norm.item() == 0.0:
            return 0.0, delta, None, 0

        residual = delta
        if historical_rank > 0:
            old_basis = basis[:historical_rank, :]
            residual = delta - delta @ old_basis.t() @ old_basis

        residual_norm = torch.linalg.norm(residual)
        novelty = (residual_norm / (delta_norm + 1e-12)).item()
        if residual_norm.item() == 0.0:
            return novelty, residual, None, 0

        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        tol = max(1e-8, 1e-6 * float(s[0].item()))
        est_rank = int(torch.count_nonzero(s > tol).item())
        if est_rank <= 0:
            est_rank = 1
        return novelty, residual, (u, s, vh), est_rank

    def _compute_task_drift_residual(self, basis, task_delta, historical_rank):
        delta_norm = torch.linalg.norm(task_delta)
        if delta_norm.item() == 0.0:
            projected = torch.zeros(
                task_delta.size(0),
                historical_rank,
                device=task_delta.device,
                dtype=task_delta.dtype,
            )
            return 0.0, projected, task_delta, None, 0

        projected = torch.zeros(
            task_delta.size(0),
            historical_rank,
            device=task_delta.device,
            dtype=task_delta.dtype,
        )
        residual = task_delta
        if historical_rank > 0:
            old_basis = basis[:historical_rank, :]
            projected = task_delta @ old_basis.t()
            residual = task_delta - projected @ old_basis

        residual_norm = torch.linalg.norm(residual)
        novelty = (residual_norm / (delta_norm + 1e-12)).item()
        if residual_norm.item() == 0.0:
            return novelty, projected, residual, None, 0

        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        tol = max(1e-8, 1e-6 * float(s[0].item()))
        est_rank = int(torch.count_nonzero(s > tol).item())
        if est_rank <= 0:
            est_rank = 1
        return novelty, projected, residual, (u, s, vh), est_rank

    def _fit_residual_to_branch(
        self,
        basis,
        memory,
        buffer,
        importance,
        importance_max,
        historical_rank,
        target_rank,
        residual_svd,
    ):
        rank = self.active_rank
        end = historical_rank + target_rank
        buffer.data[:, historical_rank:rank].zero_()
        importance.data[historical_rank:rank].zero_()
        importance_max.data[historical_rank:rank].zero_()

        if target_rank > 0 and residual_svd is not None:
            u, s, vh = residual_svd
            copy_rank = min(target_rank, vh.size(0))
            memory.data[:, historical_rank : historical_rank + copy_rank].copy_(
                u[:, :copy_rank] * s[:copy_rank].unsqueeze(0)
            )
            basis.data[historical_rank : historical_rank + copy_rank].copy_(vh[:copy_rank])
            if copy_rank < target_rank:
                memory.data[:, historical_rank + copy_rank : end].zero_()
        else:
            memory.data[:, historical_rank:end].zero_()

        if end < rank:
            basis.data[end:rank].zero_()
            memory.data[:, end:rank].zero_()
            buffer.data[:, end:rank].zero_()
            importance.data[end:rank].zero_()
            importance_max.data[end:rank].zero_()

    def _resolve_protected_slots(self, historical_rank, protected_slots, protected_slots_ratio):
        count = max(0, int(protected_slots))
        ratio = max(0.0, float(protected_slots_ratio))
        if ratio > 0.0 and historical_rank > 0:
            count = max(count, int(round(ratio * historical_rank)))
        return min(max(0, historical_rank), count)

    def _estimate_rank_from_singulars(self, singular_values):
        if singular_values is None or singular_values.numel() == 0:
            return 0
        tol = max(1e-8, 1e-6 * float(singular_values[0].item()))
        est_rank = int(torch.count_nonzero(singular_values > tol).item())
        return max(1, est_rank)

    def _transport_importance(self, old_basis, old_importance, new_basis):
        if (
            old_basis is None
            or old_importance is None
            or new_basis is None
            or old_basis.numel() == 0
            or old_importance.numel() == 0
            or new_basis.numel() == 0
        ):
            return torch.zeros(new_basis.size(0), device=new_basis.device, dtype=new_basis.dtype)
        overlap = torch.matmul(new_basis, old_basis.t())
        carried = torch.matmul(overlap.pow(2), old_importance)
        return torch.nan_to_num(carried, nan=0.0, posinf=1e6, neginf=0.0)

    def _importance_aware_branch_consolidation(
        self,
        basis,
        memory,
        buffer,
        importance,
        importance_max,
        historical_rank,
        merged_delta,
        novelty_threshold,
        protected_slots,
        importance_transport,
    ):
        device = merged_delta.device
        dtype = merged_delta.dtype
        rank = self.active_rank

        old_basis = basis[:historical_rank, :].detach().clone()
        old_importance = importance[:historical_rank].detach().clone().to(device=device, dtype=dtype).clamp(min=0.0)

        protected_slots = min(max(0, int(protected_slots)), historical_rank)
        protected_basis = torch.zeros(0, basis.size(1), device=device, dtype=dtype)
        protected_coeff = torch.zeros(merged_delta.size(0), 0, device=device, dtype=dtype)
        if protected_slots > 0 and historical_rank > 0:
            scores = old_importance
            if torch.sum(scores).item() <= 0.0:
                scores = torch.linalg.norm(memory[:, :historical_rank], dim=0).to(device=device, dtype=dtype)
            if torch.sum(scores).item() <= 0.0:
                scores = torch.ones_like(scores)
            keep_idx = torch.topk(scores, k=protected_slots, largest=True).indices.sort().values
            protected_basis = old_basis[keep_idx, :]
            protected_coeff = merged_delta @ protected_basis.t()

        protected_recon = protected_coeff @ protected_basis if protected_slots > 0 else torch.zeros_like(merged_delta)
        residual = merged_delta - protected_recon
        merged_norm = torch.linalg.norm(merged_delta)
        residual_norm = torch.linalg.norm(residual)
        novelty = (residual_norm / (merged_norm + 1e-12)).item() if merged_norm.item() > 0.0 else 0.0

        residual_keep = 0
        residual_u = None
        residual_s = None
        residual_vh = None
        residual_budget = max(self.max_rank - protected_slots, 0)
        if residual_budget > 0 and residual_norm.item() > 0.0 and novelty > novelty_threshold:
            residual_u, residual_s, residual_vh = torch.linalg.svd(residual, full_matrices=False)
            residual_keep = min(residual_budget, self._estimate_rank_from_singulars(residual_s))

        copy_rank = 0
        if residual_keep > 0 and residual_vh is not None:
            copy_rank = min(residual_keep, residual_vh.size(0))

        new_rank = protected_slots + copy_rank
        basis.data.zero_()
        memory.data.zero_()
        buffer.data.zero_()
        importance.data.zero_()
        importance_max.data.zero_()

        if protected_slots > 0:
            basis.data[:protected_slots].copy_(protected_basis)
            memory.data[:, :protected_slots].copy_(protected_coeff)

        if copy_rank > 0:
            start = protected_slots
            end = start + copy_rank
            memory.data[:, start:end].copy_(residual_u[:, :copy_rank] * residual_s[:copy_rank].unsqueeze(0))
            basis.data[start:end].copy_(residual_vh[:copy_rank])

        if new_rank > 0:
            if importance_transport:
                carried = self._transport_importance(old_basis, old_importance, basis[:new_rank, :])
            else:
                carried = torch.zeros(new_rank, device=device, dtype=dtype)
            importance.data[:new_rank].copy_(carried)
            importance_max.data[:new_rank].copy_(carried)

        if new_rank < rank:
            basis.data[new_rank:rank].zero_()
            memory.data[:, new_rank:rank].zero_()
            buffer.data[:, new_rank:rank].zero_()
            importance.data[new_rank:rank].zero_()
            importance_max.data[new_rank:rank].zero_()

        return int(new_rank)

    def consolidate_task(
        self,
        gamma,
        historical_rank=None,
        conflict_gate_strength=0.0,
        conflict_gate_floor=0.0,
        basis_update_mode="warmup_gradient",
        novelty_threshold=0.0,
        importance_aware_consolidation=False,
        protected_slots=0,
        protected_slots_ratio=0.0,
        importance_transport=True,
    ):
        with torch.no_grad():
            rank = self.active_rank
            historical_rank = rank if historical_rank is None else max(0, min(int(historical_rank), rank))
            conflict_gate_floor = max(0.0, min(float(conflict_gate_floor), 1.0))
            conflict_gate_strength = max(0.0, float(conflict_gate_strength))
            basis_update_mode = str(basis_update_mode).lower()
            novelty_threshold = max(0.0, float(novelty_threshold))
            protected_slots = self._resolve_protected_slots(
                historical_rank,
                protected_slots,
                protected_slots_ratio,
            )
            importance_aware_consolidation = bool(importance_aware_consolidation)

            if basis_update_mode == "realized_drift" and self.use_task_drift:
                task_delta_k = gamma * self._task_delta_weight(self.task_memory_k, self.task_basis_k)
                task_delta_v = gamma * self._task_delta_weight(self.task_memory_v, self.task_basis_v)

                if importance_aware_consolidation:
                    old_delta_k = torch.zeros_like(task_delta_k)
                    old_delta_v = torch.zeros_like(task_delta_v)
                    if historical_rank > 0:
                        old_delta_k = self.lora_memory_k[:, :historical_rank] @ self.lora_basis_k[:historical_rank, :]
                        old_delta_v = self.lora_memory_v[:, :historical_rank] @ self.lora_basis_v[:historical_rank, :]

                    merged_k = old_delta_k + task_delta_k
                    merged_v = old_delta_v + task_delta_v

                    new_rank_k = self._importance_aware_branch_consolidation(
                        self.lora_basis_k,
                        self.lora_memory_k,
                        self.lora_buffer_k,
                        self.importance_k,
                        self.importance_max_k,
                        historical_rank,
                        merged_k,
                        novelty_threshold,
                        protected_slots,
                        importance_transport,
                    )
                    new_rank_v = self._importance_aware_branch_consolidation(
                        self.lora_basis_v,
                        self.lora_memory_v,
                        self.lora_buffer_v,
                        self.importance_v,
                        self.importance_max_v,
                        historical_rank,
                        merged_v,
                        novelty_threshold,
                        protected_slots,
                        importance_transport,
                    )

                    self.active_rank = min(self.max_rank, max(new_rank_k, new_rank_v))
                    self.clear_task_drift()
                    if self.enforce_rank_budget and self.active_rank > self.rank_budget:
                        self._prune()
                    return

                novelty_k, projected_k, _, svd_k, est_rank_k = self._compute_task_drift_residual(
                    self.lora_basis_k,
                    task_delta_k,
                    historical_rank,
                )
                novelty_v, projected_v, _, svd_v, est_rank_v = self._compute_task_drift_residual(
                    self.lora_basis_v,
                    task_delta_v,
                    historical_rank,
                )

                if historical_rank > 0:
                    self.lora_memory_k.data[:, :historical_rank] += projected_k
                    self.lora_memory_v.data[:, :historical_rank] += projected_v

                keep_rank = 0
                available_rank = max(self.max_rank - historical_rank, 0)
                novelty = 0.5 * (novelty_k + novelty_v)
                if novelty > novelty_threshold and available_rank > 0:
                    keep_rank = min(available_rank, max(est_rank_k, est_rank_v))

                end = historical_rank + keep_rank
                if keep_rank > 0:
                    u_k, s_k, vh_k = svd_k
                    u_v, s_v, vh_v = svd_v
                    copy_rank_k = min(keep_rank, vh_k.size(0))
                    copy_rank_v = min(keep_rank, vh_v.size(0))
                    self.lora_memory_k.data[:, historical_rank : historical_rank + copy_rank_k].copy_(
                        u_k[:, :copy_rank_k] * s_k[:copy_rank_k].unsqueeze(0)
                    )
                    self.lora_basis_k.data[historical_rank : historical_rank + copy_rank_k].copy_(vh_k[:copy_rank_k])
                    if copy_rank_k < keep_rank:
                        self.lora_memory_k.data[:, historical_rank + copy_rank_k : end].zero_()
                        self.lora_basis_k.data[historical_rank + copy_rank_k : end].zero_()

                    self.lora_memory_v.data[:, historical_rank : historical_rank + copy_rank_v].copy_(
                        u_v[:, :copy_rank_v] * s_v[:copy_rank_v].unsqueeze(0)
                    )
                    self.lora_basis_v.data[historical_rank : historical_rank + copy_rank_v].copy_(vh_v[:copy_rank_v])
                    if copy_rank_v < keep_rank:
                        self.lora_memory_v.data[:, historical_rank + copy_rank_v : end].zero_()
                        self.lora_basis_v.data[historical_rank + copy_rank_v : end].zero_()

                if end < self.max_rank:
                    self.lora_basis_k.data[end:].zero_()
                    self.lora_basis_v.data[end:].zero_()
                    self.lora_memory_k.data[:, end:].zero_()
                    self.lora_memory_v.data[:, end:].zero_()
                    self.lora_buffer_k.data[:, end:].zero_()
                    self.lora_buffer_v.data[:, end:].zero_()
                    self.importance_k.data[end:].zero_()
                    self.importance_v.data[end:].zero_()
                    self.importance_max_k.data[end:].zero_()
                    self.importance_max_v.data[end:].zero_()

                self.lora_buffer_k.data.zero_()
                self.lora_buffer_v.data.zero_()
                if historical_rank < end:
                    self.importance_k.data[historical_rank:end].zero_()
                    self.importance_v.data[historical_rank:end].zero_()
                    self.importance_max_k.data[historical_rank:end].zero_()
                    self.importance_max_v.data[historical_rank:end].zero_()

                self.active_rank = end
                self.clear_task_drift()

                if self.enforce_rank_budget and self.active_rank > self.rank_budget:
                    self._prune()
                return

            if historical_rank < rank and conflict_gate_strength > 0.0:
                gates_k, _ = self._compute_new_rank_gates(
                    self.lora_memory_k,
                    self.lora_buffer_k,
                    self.lora_basis_k,
                    self.importance_k,
                    historical_rank,
                    rank,
                    conflict_gate_strength,
                    conflict_gate_floor,
                )
                gates_v, _ = self._compute_new_rank_gates(
                    self.lora_memory_v,
                    self.lora_buffer_v,
                    self.lora_basis_v,
                    self.importance_v,
                    historical_rank,
                    rank,
                    conflict_gate_strength,
                    conflict_gate_floor,
                )
                self.lora_buffer_k.data[:, historical_rank:rank].mul_(gates_k.unsqueeze(0))
                self.lora_buffer_v.data[:, historical_rank:rank].mul_(gates_v.unsqueeze(0))

            if basis_update_mode == "realized_drift":
                if importance_aware_consolidation:
                    merged_k = self._delta_weight(self.lora_basis_k, self.lora_memory_k, self.lora_buffer_k, False)
                    merged_v = self._delta_weight(self.lora_basis_v, self.lora_memory_v, self.lora_buffer_v, False)
                    merged_k = merged_k + gamma * (self.lora_buffer_k[:, :rank] @ self.lora_basis_k[:rank, :])
                    merged_v = merged_v + gamma * (self.lora_buffer_v[:, :rank] @ self.lora_basis_v[:rank, :])

                    new_rank_k = self._importance_aware_branch_consolidation(
                        self.lora_basis_k,
                        self.lora_memory_k,
                        self.lora_buffer_k,
                        self.importance_k,
                        self.importance_max_k,
                        historical_rank,
                        merged_k,
                        novelty_threshold,
                        protected_slots,
                        importance_transport,
                    )
                    new_rank_v = self._importance_aware_branch_consolidation(
                        self.lora_basis_v,
                        self.lora_memory_v,
                        self.lora_buffer_v,
                        self.importance_v,
                        self.importance_max_v,
                        historical_rank,
                        merged_v,
                        novelty_threshold,
                        protected_slots,
                        importance_transport,
                    )
                    self.active_rank = min(self.max_rank, max(new_rank_k, new_rank_v))
                    self.lora_buffer_k.data.zero_()
                    self.lora_buffer_v.data.zero_()
                    if self.enforce_rank_budget and self.active_rank > self.rank_budget:
                        self._prune()
                    return

                if historical_rank > 0:
                    self.lora_memory_k.data[:, :historical_rank] += gamma * self.lora_buffer_k.data[:, :historical_rank]
                    self.lora_memory_v.data[:, :historical_rank] += gamma * self.lora_buffer_v.data[:, :historical_rank]

                keep_rank = 0
                if historical_rank < rank:
                    novelty_k, _, svd_k, est_rank_k = self._compute_realized_drift_residual(
                        self.lora_basis_k,
                        self.lora_buffer_k,
                        historical_rank,
                        gamma,
                    )
                    novelty_v, _, svd_v, est_rank_v = self._compute_realized_drift_residual(
                        self.lora_basis_v,
                        self.lora_buffer_v,
                        historical_rank,
                        gamma,
                    )
                    novelty = 0.5 * (novelty_k + novelty_v)
                    if novelty > novelty_threshold:
                        keep_rank = min(rank - historical_rank, max(est_rank_k, est_rank_v))

                    self._fit_residual_to_branch(
                        self.lora_basis_k,
                        self.lora_memory_k,
                        self.lora_buffer_k,
                        self.importance_k,
                        self.importance_max_k,
                        historical_rank,
                        keep_rank,
                        svd_k,
                    )
                    self._fit_residual_to_branch(
                        self.lora_basis_v,
                        self.lora_memory_v,
                        self.lora_buffer_v,
                        self.importance_v,
                        self.importance_max_v,
                        historical_rank,
                        keep_rank,
                        svd_v,
                    )
                    self.active_rank = historical_rank + keep_rank
                else:
                    self.lora_buffer_k.data.zero_()
                    self.lora_buffer_v.data.zero_()
            else:
                self.lora_memory_k.data[:, :rank] += gamma * self.lora_buffer_k.data[:, :rank]
                self.lora_memory_v.data[:, :rank] += gamma * self.lora_buffer_v.data[:, :rank]
                self.lora_buffer_k.data.zero_()
                self.lora_buffer_v.data.zero_()

            if self.enforce_rank_budget and self.active_rank > self.rank_budget:
                self._prune()

    def _prune(self):
        utility_k = self.importance_k[: self.active_rank] + torch.linalg.norm(
            self.lora_memory_k[:, : self.active_rank], dim=0
        )
        utility_v = self.importance_v[: self.active_rank] + torch.linalg.norm(
            self.lora_memory_v[:, : self.active_rank], dim=0
        )
        utility = utility_k + utility_v
        keep = torch.topk(utility, k=self.rank_budget, largest=True).indices.sort().values

        self.lora_basis_k.data[: self.rank_budget].copy_(self.lora_basis_k.data[keep])
        self.lora_basis_v.data[: self.rank_budget].copy_(self.lora_basis_v.data[keep])
        self.lora_memory_k.data[:, : self.rank_budget].copy_(self.lora_memory_k.data[:, keep])
        self.lora_memory_v.data[:, : self.rank_budget].copy_(self.lora_memory_v.data[:, keep])
        self.lora_buffer_k.data[:, : self.rank_budget].copy_(self.lora_buffer_k.data[:, keep])
        self.lora_buffer_v.data[:, : self.rank_budget].copy_(self.lora_buffer_v.data[:, keep])
        self.importance_k.data[: self.rank_budget].copy_(self.importance_k.data[keep])
        self.importance_v.data[: self.rank_budget].copy_(self.importance_v.data[keep])
        self.importance_max_k.data[: self.rank_budget].copy_(self.importance_max_k.data[keep])
        self.importance_max_v.data[: self.rank_budget].copy_(self.importance_max_v.data[keep])

        self.lora_basis_k.data[self.rank_budget :].zero_()
        self.lora_basis_v.data[self.rank_budget :].zero_()
        self.lora_memory_k.data[:, self.rank_budget :].zero_()
        self.lora_memory_v.data[:, self.rank_budget :].zero_()
        self.lora_buffer_k.data[:, self.rank_budget :].zero_()
        self.lora_buffer_v.data[:, self.rank_budget :].zero_()
        self.importance_k.data[self.rank_budget :].zero_()
        self.importance_v.data[self.rank_budget :].zero_()
        self.importance_max_k.data[self.rank_budget :].zero_()
        self.importance_max_v.data[self.rank_budget :].zero_()
        self.active_rank = self.rank_budget
        self._orthonormalize_basis(self.lora_basis_k, self.active_rank)
        self._orthonormalize_basis(self.lora_basis_v, self.active_rank)

    def init_fisher_storage(self):
        return [torch.zeros_like(self.fisher_k), torch.zeros_like(self.fisher_v)]

    def save_grad(self, name):
        def hook(grad):
            setattr(self, name, grad)

        return hook

    def forward(self, x, use_buffer=True, register_hook=False):
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x)

        delta_k = self._delta_weight(
            self.lora_basis_k,
            self.lora_memory_k,
            self.lora_buffer_k,
            use_buffer and not self.use_task_drift,
        )
        delta_v = self._delta_weight(
            self.lora_basis_v,
            self.lora_memory_v,
            self.lora_buffer_v,
            use_buffer and not self.use_task_drift,
        )
        if use_buffer and self.use_task_drift:
            delta_k = delta_k + self._task_delta_weight(self.task_memory_k, self.task_basis_k)
            delta_v = delta_v + self._task_delta_weight(self.task_memory_v, self.task_basis_v)

        if use_buffer and register_hook:
            if not delta_k.requires_grad:
                delta_k = delta_k.detach().clone().requires_grad_(True)
            if not delta_v.requires_grad:
                delta_v = delta_v.detach().clone().requires_grad_(True)
            delta_k.register_hook(self.save_grad("delta_w_k_grad"))
            delta_v.register_hook(self.save_grad("delta_w_v_grad"))

        qkv[:, :, self.dim : 2 * self.dim] += x @ delta_k.t()
        qkv[:, :, 2 * self.dim :] += x @ delta_v.t()

        qkv = qkv.reshape(bsz, seq_len, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        n_tasks=10,
        r=4,
        rank_budget=10,
        max_rank=20,
        task_rank=2,
        enforce_rank_budget=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionDirFisLoRA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            n_tasks=n_tasks,
            r=r,
            rank_budget=rank_budget,
            max_rank=max_rank,
            task_rank=task_rank,
            enforce_rank_budget=enforce_rank_budget,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, use_buffer=True, register_hook=False):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), use_buffer=use_buffer, register_hook=register_hook)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(BaseVisionTransformer):
    pass
