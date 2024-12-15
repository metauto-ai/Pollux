# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging

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

from apps.main.modules.schedulers import RectifiedFlow, SchedulerArgs
from apps.main.modules.transformer import DiffusionTransformer, DiffusionTransformerArgs
from apps.main.modules.vae import LatentVideoVAE, LatentVideoVAEArgs


logger = logging.getLogger()


@dataclass
class ModelArgs:
    transformer: DiffusionTransformerArgs = field(
        default_factory=DiffusionTransformerArgs
    )
    vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)


class LatentDiffusionTransformer(nn.Module):
    """
    Latent Diffusion Transformer Model.
    This model integrates a VAE for latent compression, a transformer for temporal and spatial token mixing,
    and a custom scheduler for diffusion steps.
    """

    version: str = "v0.3"
    description: str = (
        "Latent Diffusion Transformer for VideoGen: (1) currently we only support image classification for debugging."
    )

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.compressor = LatentVideoVAE(args.vae)
        self.scheduler = RectifiedFlow(args.scheduler)
        self.transformer = DiffusionTransformer(args.transformer)

    def forward(self, batch: torch.Tensor) -> dict[str:any]:

        image = batch["image"]
        condition = batch["label"]
        latent_code = self.compressor.encode(image)
        noised_x, t, target = self.scheduler.sample_noised_input(latent_code)
        output = self.transformer(x=noised_x, time_steps=t, condition=condition)
        batch["prediction"] = output
        batch["target"] = target
        target = target.to(output.dtype)
        loss = F.mse_loss(output, target)

        return batch, loss

    def init_weights(self, pre_trained_path: Optional[str] = None):
        self.transformer.init_weights(pre_trained_path)


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: DiffusionTransformerArgs, vae_config: dict):
    group_plan: Tuple[int, bool] = []

    for i in range(len(vae_config.down_block_types)):
        group_plan.append((f"compressor.vae.encoder.down_blocks.{i}", False))

    for i in range(model_args.n_layers):
        group_plan.append((f"transformer.layers.{i}", False))

    group_plan.append(("transformer.img_output", True))
    logger.info(f"The `group_plan` for fsdp is:\n{group_plan}")

    return group_plan


def tp_parallelize(
    model, tp_mesh, model_args: DiffusionTransformerArgs, distributed_args
):

    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    main_plan = {}
    main_plan["norm"] = SequenceParallel()
    main_plan["img_output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model.transformer,
        tp_mesh,
        main_plan,
    )

    # TODO: Adding more for DiT specific Modules
    for layer in model.transformer.layers:
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
