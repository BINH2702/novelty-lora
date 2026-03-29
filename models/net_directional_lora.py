import torch
import torch.nn as nn
from functools import partial

from models.vit_directional_lora import VisionTransformer, PatchEmbed, Block, AttentionDirectionalLoRA
from models.vit_directional_lora import resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for Vision Transformer models.")

    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg["num_classes"]
    num_classes = kwargs.get("num_classes", default_num_classes)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in pretrained_cfg["url"],
        **kwargs
    )
    return model


class ViT(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        init_values=None,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        n_tasks=10,
        rank=4,
        rank_budget=10,
        max_rank=20,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            representation_size=representation_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            init_values=init_values,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=partial(block_fn, rank_budget=rank_budget, max_rank=max_rank),
            n_tasks=n_tasks,
            rank=rank,
        )

    def forward(self, x, use_buffer=True, register_hook=False, rank_limits=None, rank_windows=None):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        for blk_idx, blk in enumerate(self.blocks):
            layer_limit = None if rank_limits is None else rank_limits.get(blk_idx)
            layer_window = None if rank_windows is None else rank_windows.get(blk_idx)
            x = blk(
                x,
                use_buffer=use_buffer,
                register_hook=register_hook,
                rank_limit=layer_limit,
                rank_window=layer_window,
            )

        x = self.norm(x)
        return x


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        model_kwargs = dict(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            n_tasks=args["sessions"],
            rank=args["rank"],
            rank_budget=args.get("rank_budget", args["rank"]),
            max_rank=args.get("max_rank", args.get("rank_budget", args["rank"])),
        )

        self.image_encoder = _create_vision_transformer(
            args["load"],
            pretrained=args.get("pretrained", True),
            **model_kwargs
        )
        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList(
            [nn.Linear(768, self.class_num, bias=True) for _ in range(args["sessions"])]
        )

        for module in self.image_encoder.modules():
            if isinstance(module, AttentionDirectionalLoRA):
                module.init_param()

        self._cur_task = -1

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def forward(self, image, use_buffer=True, fc_only=False, register_hook=False, rank_limits=None, rank_windows=None):
        if fc_only:
            fc_outs = []
            for ti in range(self._cur_task + 1):
                fc_outs.append(self.classifier_pool[ti](image))
            return torch.cat(fc_outs, dim=1)

        image_features = self.image_encoder(
            image,
            use_buffer=use_buffer,
            register_hook=register_hook,
            rank_limits=rank_limits,
            rank_windows=rank_windows,
        )
        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.size(0), -1)
        logits = self.classifier_pool[self._cur_task](image_features)
        return {"logits": logits, "features": image_features}

    def interface(self, image, use_buffer=True, rank_limits=None, rank_windows=None, max_task=None):
        logits = []
        image_features = self.image_encoder(
            image,
            use_buffer=use_buffer,
            rank_limits=rank_limits,
            rank_windows=rank_windows,
        )
        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.size(0), -1)

        task_limit = self._cur_task if max_task is None else max_task
        for classifier in self.classifier_pool[: task_limit + 1]:
            logits.append(classifier(image_features))

        return torch.cat(logits, dim=1)

    def directional_regularization(
        self,
        device,
        historical_rank_map=None,
        lambda_min=0.0,
        alpha=1.0,
        weight_power=1.0,
        weight_cap=0.0,
        new_dir_weight=0.0,
    ):
        penalty = torch.tensor(0.0, device=device)
        for module_idx, module in enumerate(self.iter_attention_modules()):
            historical_rank = None if historical_rank_map is None else historical_rank_map.get(module_idx)
            penalty = penalty + module.regularization_loss(
                device,
                historical_rank=historical_rank,
                lambda_min=lambda_min,
                alpha=alpha,
                weight_power=weight_power,
                weight_cap=weight_cap,
                new_dir_weight=new_dir_weight,
            )
        return penalty

    def consolidate_task(
        self,
        gamma,
        historical_rank_map=None,
        conflict_gate_strength=0.0,
        conflict_gate_floor=0.0,
    ):
        gate_stats = []
        for module_idx, module in enumerate(self.iter_attention_modules()):
            historical_rank = None if historical_rank_map is None else historical_rank_map.get(module_idx)
            stats = module.consolidate_task(
                gamma,
                historical_rank=historical_rank,
                conflict_gate_strength=conflict_gate_strength,
                conflict_gate_floor=conflict_gate_floor,
            )
            if stats is not None:
                stats["module_index"] = module_idx
                gate_stats.append(stats)
        return gate_stats

    def update_importance(self, fisher_values, decay, floor_frac=0.0):
        idx = 0
        for module in self.iter_attention_modules():
            module.update_importance(fisher_values[idx], fisher_values[idx + 1], decay, floor_frac=floor_frac)
            idx += 2

    def collect_direction_stats(self, max_rank_map=None):
        stats = []
        for module_idx, module in enumerate(self.iter_attention_modules()):
            layer_limit = None if max_rank_map is None else max_rank_map.get(module_idx)
            stats.extend(module.collect_direction_stats(module_idx, layer_limit=layer_limit))
        return stats

    def get_active_ranks(self):
        return {idx: module.active_rank for idx, module in enumerate(self.iter_attention_modules())}

    def get_attention_module(self, module_index):
        for idx, module in enumerate(self.iter_attention_modules()):
            if idx == module_index:
                return module
        raise IndexError(f"Unknown attention module index {module_index}")

    def iter_attention_modules(self):
        for module in self.image_encoder.modules():
            if isinstance(module, AttentionDirectionalLoRA):
                yield module

    def update_fc(self, nb_classes):
        self._cur_task += 1
