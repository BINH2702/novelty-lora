import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class AttentionNoveltyLoRA(nn.Module):
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

    def _delta_weight(self, basis, memory, buffer, use_buffer):
        rank = self.active_rank
        coeff = memory[:, :rank]
        if use_buffer:
            coeff = coeff + buffer[:, :rank]
        return coeff @ basis[:rank, :]

    def _grow_basis(self, basis, memory, buffer, grad, grow_rank, threshold):
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
        self.active_rank += add_rank
        return novelty

    def apply_warmup_gradient(self, grow_rank, threshold):
        if self.qkv.weight.grad is None:
            return 0.0

        grad = self.qkv.weight.grad.detach()
        grad_k = grad[self.dim : 2 * self.dim]
        grad_v = grad[2 * self.dim :]
        novelty_k = self._grow_basis(
            self.lora_basis_k, self.lora_memory_k, self.lora_buffer_k, grad_k, grow_rank, threshold
        )
        novelty_v = self._grow_basis(
            self.lora_basis_v, self.lora_memory_v, self.lora_buffer_v, grad_v, grow_rank, threshold
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
        utility_k = torch.linalg.norm(self.lora_memory_k[:, : self.active_rank], dim=0)
        utility_v = torch.linalg.norm(self.lora_memory_v[:, : self.active_rank], dim=0)
        utility = utility_k + utility_v
        keep = torch.topk(utility, k=self.rank_budget, largest=True).indices.sort().values

        self.lora_basis_k.data[: self.rank_budget].copy_(self.lora_basis_k.data[keep])
        self.lora_basis_v.data[: self.rank_budget].copy_(self.lora_basis_v.data[keep])
        self.lora_memory_k.data[:, : self.rank_budget].copy_(self.lora_memory_k.data[:, keep])
        self.lora_memory_v.data[:, : self.rank_budget].copy_(self.lora_memory_v.data[:, keep])
        self.lora_buffer_k.data[:, : self.rank_budget].copy_(self.lora_buffer_k.data[:, keep])
        self.lora_buffer_v.data[:, : self.rank_budget].copy_(self.lora_buffer_v.data[:, keep])

        self.lora_basis_k.data[self.rank_budget :].zero_()
        self.lora_basis_v.data[self.rank_budget :].zero_()
        self.lora_memory_k.data[:, self.rank_budget :].zero_()
        self.lora_memory_v.data[:, self.rank_budget :].zero_()
        self.lora_buffer_k.data[:, self.rank_budget :].zero_()
        self.lora_buffer_v.data[:, self.rank_budget :].zero_()
        self.active_rank = self.rank_budget
        self._orthonormalize_basis(self.lora_basis_k, self.active_rank)
        self._orthonormalize_basis(self.lora_basis_v, self.active_rank)

    def update_fisher(self, fisher_k, fisher_v):
        self.fisher_k.copy_(fisher_k.to(self.fisher_k.device))
        self.fisher_v.copy_(fisher_v.to(self.fisher_v.device))

    def regularization_loss(self, device):
        delta_k = self.lora_buffer_k[:, : self.active_rank] @ self.lora_basis_k[: self.active_rank, :]
        delta_v = self.lora_buffer_v[:, : self.active_rank] @ self.lora_basis_v[: self.active_rank, :]
        penalty_k = torch.sum(self.fisher_k.to(device) * (delta_k ** 2))
        penalty_v = torch.sum(self.fisher_v.to(device) * (delta_v ** 2))
        return 0.5 * (penalty_k + penalty_v)

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
            delta_buf_k = self.lora_buffer_k[:, : self.active_rank] @ self.lora_basis_k[: self.active_rank, :]
            delta_buf_v = self.lora_buffer_v[:, : self.active_rank] @ self.lora_basis_v[: self.active_rank, :]
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
        self.attn = AttentionNoveltyLoRA(
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

    def forward(self, x, use_buffer=True, register_hook=False):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), use_buffer=use_buffer, register_hook=register_hook)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(BaseVisionTransformer):
    pass
