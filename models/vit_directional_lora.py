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


class AttentionDirectionalLoRA(nn.Module):
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
        self.active_rank = r

        self.lora_basis_k = nn.Parameter(torch.zeros(self.max_rank, dim), requires_grad=False)
        self.lora_basis_v = nn.Parameter(torch.zeros(self.max_rank, dim), requires_grad=False)
        self.lora_memory_k = nn.Parameter(torch.zeros(dim, self.max_rank), requires_grad=False)
        self.lora_memory_v = nn.Parameter(torch.zeros(dim, self.max_rank), requires_grad=False)
        self.lora_buffer_k = nn.Parameter(torch.zeros(dim, self.max_rank))
        self.lora_buffer_v = nn.Parameter(torch.zeros(dim, self.max_rank))

        self.register_buffer("importance_k", torch.zeros(self.max_rank))
        self.register_buffer("importance_v", torch.zeros(self.max_rank))
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

    def _orthonormalize_basis(self, basis, rank):
        if rank <= 0:
            return
        q, _ = torch.linalg.qr(basis[:rank].data.t(), mode="reduced")
        basis[:rank].data.copy_(q.t())

    def _delta_weight(self, basis, memory, buffer, use_buffer, rank_limit=None):
        rank = self.active_rank if rank_limit is None else min(self.active_rank, rank_limit)
        coeff = memory[:, :rank]
        if use_buffer:
            coeff = coeff + buffer[:, :rank]
        return coeff @ basis[:rank, :]

    def _grow_basis(self, basis, memory, buffer, importance, grad, grow_rank, threshold):
        grad_norm = torch.linalg.norm(grad)
        if grad_norm.item() == 0:
            return 0.0

        rank = self.active_rank
        novelty = 1.0
        residual = grad
        if rank > 0:
            active_basis = basis[:rank]
            proj = grad @ active_basis.t() @ active_basis
            residual = grad - proj
            novelty = (torch.linalg.norm(residual) / (grad_norm + 1e-12)).item()

        if novelty <= threshold or rank >= self.max_rank:
            return novelty

        add_rank = min(grow_rank, self.max_rank - rank)
        _, _, vh = torch.linalg.svd(residual, full_matrices=False)
        new_dirs = vh[:add_rank]
        basis.data[rank : rank + add_rank].copy_(new_dirs)
        self._orthonormalize_basis(basis, rank + add_rank)
        memory.data[:, rank : rank + add_rank].zero_()
        buffer.data[:, rank : rank + add_rank].zero_()
        importance.data[rank : rank + add_rank].zero_()
        self.active_rank += add_rank
        return novelty

    def apply_warmup_gradient(self, grow_rank, threshold):
        if self.qkv.weight.grad is None:
            return 0.0

        grad = self.qkv.weight.grad.detach()
        grad_k = grad[self.dim : 2 * self.dim]
        grad_v = grad[2 * self.dim :]
        novelty_k = self._grow_basis(
            self.lora_basis_k,
            self.lora_memory_k,
            self.lora_buffer_k,
            self.importance_k,
            grad_k,
            grow_rank,
            threshold,
        )
        novelty_v = self._grow_basis(
            self.lora_basis_v,
            self.lora_memory_v,
            self.lora_buffer_v,
            self.importance_v,
            grad_v,
            grow_rank,
            threshold,
        )
        self.qkv.weight.grad = None
        if self.qkv.bias is not None:
            self.qkv.bias.grad = None
        return 0.5 * (novelty_k + novelty_v)

    def consolidate_task(self, gamma):
        rank = self.active_rank
        self.lora_memory_k.data[:, :rank] += gamma * self.lora_buffer_k.data[:, :rank]
        self.lora_memory_v.data[:, :rank] += gamma * self.lora_buffer_v.data[:, :rank]
        self.lora_buffer_k.data.zero_()
        self.lora_buffer_v.data.zero_()
        if self.active_rank > self.rank_budget:
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

        self.lora_basis_k.data[self.rank_budget :].zero_()
        self.lora_basis_v.data[self.rank_budget :].zero_()
        self.lora_memory_k.data[:, self.rank_budget :].zero_()
        self.lora_memory_v.data[:, self.rank_budget :].zero_()
        self.lora_buffer_k.data[:, self.rank_budget :].zero_()
        self.lora_buffer_v.data[:, self.rank_budget :].zero_()
        self.importance_k.data[self.rank_budget :].zero_()
        self.importance_v.data[self.rank_budget :].zero_()
        self.active_rank = self.rank_budget
        self._orthonormalize_basis(self.lora_basis_k, self.active_rank)
        self._orthonormalize_basis(self.lora_basis_v, self.active_rank)

    def update_importance(self, fisher_k, fisher_v, decay):
        self.fisher_k.copy_(fisher_k.to(self.fisher_k.device))
        self.fisher_v.copy_(fisher_v.to(self.fisher_v.device))

        rank = self.active_rank
        fisher_scores_k = self._project_fisher_scores(self.fisher_k, self.lora_basis_k, rank)
        fisher_scores_v = self._project_fisher_scores(self.fisher_v, self.lora_basis_v, rank)

        self.importance_k[:rank].mul_(decay).add_(fisher_scores_k, alpha=1.0 - decay)
        self.importance_v[:rank].mul_(decay).add_(fisher_scores_v, alpha=1.0 - decay)

    def _project_fisher_scores(self, fisher, basis, rank):
        if rank <= 0:
            return torch.zeros(0, device=fisher.device)
        basis_sq = basis[:rank] ** 2
        fisher_mean = fisher.mean(dim=0, keepdim=True)
        return (basis_sq * fisher_mean).sum(dim=1)

    def regularization_loss(self, device):
        rank = self.active_rank
        if rank == 0:
            return torch.tensor(0.0, device=device)

        coeff_k = self.lora_buffer_k[:, :rank]
        coeff_v = self.lora_buffer_v[:, :rank]
        penalty_k = (coeff_k.pow(2).sum(dim=0) * self.importance_k[:rank].to(device)).sum()
        penalty_v = (coeff_v.pow(2).sum(dim=0) * self.importance_v[:rank].to(device)).sum()
        return 0.5 * (penalty_k + penalty_v)

    def init_fisher_storage(self):
        return [torch.zeros_like(self.fisher_k), torch.zeros_like(self.fisher_v)]

    def save_grad(self, name):
        def hook(grad):
            setattr(self, name, grad)
        return hook

    def get_direction_vector(self, kind, direction_index):
        tensor = self.lora_memory_k if kind == "k" else self.lora_memory_v
        vec = tensor[:, direction_index].detach()
        norm = torch.linalg.norm(vec)
        if norm.item() > 0:
            return vec / norm

        fallback = torch.zeros_like(vec)
        fallback[0] = 1.0
        return fallback

    def collect_direction_stats(self, module_index, layer_limit=None):
        rank = self.active_rank if layer_limit is None else min(self.active_rank, layer_limit)
        stats = []
        for kind, importance, memory in (
            ("k", self.importance_k, self.lora_memory_k),
            ("v", self.importance_v, self.lora_memory_v),
        ):
            for direction_index in range(rank):
                coeff = memory[:, direction_index].detach()
                stats.append(
                    {
                        "module_index": module_index,
                        "kind": kind,
                        "direction_index": direction_index,
                        "fisher_weight": float(importance[direction_index].item()),
                        "energy_weight": float(coeff.pow(2).sum().item()),
                        "coeff_norm": float(torch.linalg.norm(coeff).item()),
                    }
                )
        return stats

    def forward(self, x, use_buffer=True, register_hook=False, rank_limit=None):
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x)

        delta_k = self._delta_weight(
            self.lora_basis_k,
            self.lora_memory_k,
            self.lora_buffer_k,
            use_buffer,
            rank_limit=rank_limit,
        )
        delta_v = self._delta_weight(
            self.lora_basis_v,
            self.lora_memory_v,
            self.lora_buffer_v,
            use_buffer,
            rank_limit=rank_limit,
        )
        qkv[:, :, self.dim : 2 * self.dim] += x @ delta_k.t()
        qkv[:, :, 2 * self.dim :] += x @ delta_v.t()

        if use_buffer and register_hook:
            limit = self.active_rank if rank_limit is None else min(self.active_rank, rank_limit)
            delta_buf_k = self.lora_buffer_k[:, :limit] @ self.lora_basis_k[:limit, :]
            delta_buf_v = self.lora_buffer_v[:, :limit] @ self.lora_basis_v[:limit, :]
            delta_buf_k.register_hook(self.save_grad("delta_w_k_grad"))
            delta_buf_v.register_hook(self.save_grad("delta_w_v_grad"))

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
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionDirectionalLoRA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            n_tasks=n_tasks,
            r=r,
            rank_budget=rank_budget,
            max_rank=max_rank,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, use_buffer=True, register_hook=False, rank_limit=None):
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x), use_buffer=use_buffer, register_hook=register_hook, rank_limit=rank_limit))
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(BaseVisionTransformer):
    pass
