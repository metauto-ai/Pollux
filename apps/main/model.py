# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)
from apps.main.modules.tokenizer import Tokenizer, TokenizerArgs
from apps.main.modules.schedulers import RectifiedFlow, SchedulerArgs
from apps.main.modules.transformer import (
    GenTransformer,
    PlanTransformer,
    PlanTransformerArgs,
    GenTransformerArgs,
)
from apps.main.modules.vae import LatentVideoVAE, LatentVideoVAEArgs
from apps.main.data import random_mask_images

logger = logging.getLogger()


@dataclass
class ModelArgs:
    gen_transformer: GenTransformerArgs = field(default_factory=GenTransformerArgs)
    plan_transformer: PlanTransformerArgs = field(default_factory=PlanTransformerArgs)
    vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    text_cfg_ratio: float = 0.1
    image_cfg_ratio: float = 0.1
    mask_patch: int = 16


class Pollux(nn.Module):
    """
    Latent Diffusion Transformer Model.
    This model integrates a VAE for latent compression, a transformer for temporal and spatial token mixing,
    and a custom scheduler for diffusion steps.
    """

    VERSION: str = "v0.6"
    DESCRIPTION: str = (
        "Latent Diffusion Transformer for VideoGen: (1) currently we only support class conditional image generation for debugging."
    )

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.compressor = LatentVideoVAE(args.vae)
        self.scheduler = RectifiedFlow(args.scheduler)
        self.gen_transformer = GenTransformer(args.gen_transformer)
        self.tokenizer = Tokenizer(model_path=args.tokenizer.model_path)

        assert args.plan_transformer.vocab_size == self.tokenizer.n_words

        self.plan_transformer = PlanTransformer(args.plan_transformer)
        self.plan_transformer.requires_grad_(False)
        self.text_seqlen = self.plan_transformer.text_seqlen
        self.text_cfg_ratio = args.text_cfg_ratio
        self.image_cfg_ratio = args.image_cfg_ratio
        self.mask_patch = args.mask_patch
        self.token_proj = nn.Linear(
            in_features=args.plan_transformer.dim,
            out_features=args.gen_transformer.dim,
            bias=False,
        )
        init_std = self.plan_transformer.dim ** (-0.5)
        nn.init.trunc_normal_(
            self.token_proj.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )

    def cap_pos_tokenize(self, batch: dict[str:any]) -> dict[str:any]:
        batch["cap_token"] = [
            self.tokenizer.encode(x, bos=True, eos=False) for x in batch["caption"]
        ]
        pad_id = self.tokenizer.pad_id
        bsz = len(batch["cap_token"])
        tokens = torch.full(
            (bsz, self.plan_transformer.text_seqlen),
            pad_id,
            dtype=torch.long,
        ).cuda()
        for k, t in enumerate(batch["cap_token"]):
            if len(t) < tokens.size(1):
                tokens[k, : len(t)] = torch.tensor(
                    t[:], dtype=torch.long, device="cuda"
                )
            else:
                tokens[k, :] = torch.tensor(
                    t[: tokens.size(1)], dtype=torch.long, device="cuda"
                )
        batch["cap_token"] = tokens
        return batch

    def cap_neg_tokenize(self, batch: dict[str:any]) -> dict[str:any]:
        bsz = len(batch["caption"])
        pad_id = self.tokenizer.pad_id
        tokens = torch.full(
            (bsz, self.plan_transformer.text_seqlen),
            pad_id,
            dtype=torch.long,
        ).cuda()
        batch["cap_token"] = tokens
        return batch

    def cap_tokenize(self, batch: dict[str:any]) -> torch.Tensor:
        if random.random() > self.text_cfg_ratio:
            return self.cap_pos_tokenize(batch)
        else:
            return self.cap_neg_tokenize(batch)

    def forward(self, batch: dict[str:any]) -> dict[str:any]:

        image = batch["image"]
        batch = self.cap_tokenize(batch)

        mask, masked_image = random_mask_images(
            image,
            mask_ratio=random.random(),
            mask_patch=self.mask_patch,
            mask_all=random.random() < self.image_cfg_ratio,
        )

        latent_masked_code = self.compressor.encode(masked_image)

        _, c, h, w = latent_masked_code.size()
        resized_mask = F.interpolate(mask, size=(h, w), mode="nearest")
        resized_mask = torch.cat([resized_mask] * c, dim=1)
        batch["masked_latent"] = torch.cat([latent_masked_code, resized_mask], dim=1)
        with torch.no_grad():
            conditional_signal, layout = self.plan_transformer(batch)

        latent_code = self.compressor.encode(image)
        conditional_signal = self.token_proj(conditional_signal)
        noised_x, t, target = self.scheduler.sample_noised_input(latent_code)
        output = self.gen_transformer(
            x=noised_x, time_steps=t, condition=conditional_signal, layout=layout
        )
        batch["prediction"] = output
        batch["target"] = target
        target = target.to(output.dtype)
        loss = F.mse_loss(output, target)

        return batch, loss

    def set_train(self):
        self.plan_transformer.train()
        self.gen_transformer.train()

    def set_eval(self):
        self.plan_transformer.eval()
        self.gen_transformer.eval()

    def init_weights(self, args: ModelArgs):
        self.gen_transformer.init_weights(args.gen_transformer.pre_trained_path)
        self.plan_transformer.init_weights(args.plan_transformer.pre_trained_path)


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: ModelArgs, vae_config: dict):
    group_plan: Tuple[int, bool] = []

    for i in range(len(vae_config.down_block_types)):
        group_plan.append((f"compressor.vae.encoder.down_blocks.{i}", False))

    for i in range(model_args.plan_transformer.n_layers):
        group_plan.append((f"plan_transformer.layers.{i}", False))

    for i in range(model_args.gen_transformer.n_layers):
        group_plan.append((f"gen_transformer.layers.{i}", False))

    group_plan.append(("gen_transformer.img_output", True))
    logger.info(f"The `group_plan` for fsdp is:\n{group_plan}")

    return group_plan


def tp_parallelize(model, tp_mesh, model_args: ModelArgs, distributed_args):

    assert model_args.plan_transformer.dim % distributed_args.tp_size == 0
    assert model_args.plan_transformer.vocab_size % distributed_args.tp_size == 0
    assert model_args.plan_transformer.n_heads % distributed_args.tp_size == 0
    assert (model_args.plan_transformer.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.plan_transformer.n_heads % (model_args.n_kv_heads or 1) == 0

    assert model_args.gen_transformer.dim % distributed_args.tp_size == 0
    assert model_args.gen_transformer.vocab_size % distributed_args.tp_size == 0
    assert model_args.gen_transformer.n_heads % distributed_args.tp_size == 0
    assert (model_args.gen_transformer.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.gen_transformer.n_heads % (model_args.n_kv_heads or 1) == 0

    main_plan = {}
    main_plan["norm"] = SequenceParallel()
    main_plan["img_output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model.gen_transformer,
        tp_mesh,
        main_plan,
    )

    # TODO: Adding plan_transformer Modules
    for layer in model.gen_transformer.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers TP
        # Feedforward layers TP
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
