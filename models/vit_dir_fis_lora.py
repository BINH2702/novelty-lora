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
        self.enforce_rank_budget = bool(enforce_rank_budget)
        self.active_rank = r

        self.lora_basis_k = nn.Parameter(torch.zeros(self.max_rank, dim), requires_grad=False)
        self.lora_basis_v = nn.Parameter(torch.zeros(self.max_rank, dim), requires_grad=False)
        self.lora_memory_k = nn.Parameter(torch.zeros(dim, self.max_rank), requires_grad=False)
        self.lora_memory_v = nn.Parameter(torch.zeros(dim, self.max_rank), requires_grad=False)
        self.lora_buffer_k = nn.Parameter(torch.zeros(dim, self.max_rank))
        self.lora_buffer_v = nn.Parameter(torch.zeros(dim, self.max_rank))

        self.register_buffer("importance_k", torch.zeros(self.max_rank))
        self.register_buffer("importance_v", torch.zeros(self.max_rank))
        self.register_buffer("importance_max_k", torch.zeros(self.max_rank))
        self.register_buffer("importance_max_v", torch.zeros(self.max_rank))
        self.register_buffer("fisher_k", torch.zeros(dim, dim))
        self.register_buffer("fisher_v", torch.zeros(dim, dim))

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

    def _orthonormalize_basis(self, basis, rank):
        if rank <= 0:
            return
        q, _ = torch.linalg.qr(basis[:rank].data.t(), mode="reduced")
        basis[:rank].data.copy_(q.t())

    def _delta_weight(self, basis, memory, buffer, use_buffer):
        rank = self.active_rank
        coeff = memory[:, :rank]
        if use_buffer:
            coeff = coeff + buffer[:, :rank]
        return coeff @ basis[:rank, :]

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

    def consolidate_task(
        self,
        gamma,
        historical_rank=None,
        conflict_gate_strength=0.0,
        conflict_gate_floor=0.0,
    ):
        with torch.no_grad():
            rank = self.active_rank
            historical_rank = rank if historical_rank is None else max(0, min(int(historical_rank), rank))
            conflict_gate_floor = max(0.0, min(float(conflict_gate_floor), 1.0))
            conflict_gate_strength = max(0.0, float(conflict_gate_strength))

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

        delta_k = self._delta_weight(self.lora_basis_k, self.lora_memory_k, self.lora_buffer_k, use_buffer)
        delta_v = self._delta_weight(self.lora_basis_v, self.lora_memory_v, self.lora_buffer_v, use_buffer)
        qkv[:, :, self.dim : 2 * self.dim] += x @ delta_k.t()
        qkv[:, :, 2 * self.dim :] += x @ delta_v.t()

        if use_buffer and register_hook:
            delta_k.register_hook(self.save_grad("delta_w_k_grad"))
            delta_v.register_hook(self.save_grad("delta_w_v_grad"))

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
