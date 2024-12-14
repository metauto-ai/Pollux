# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Optional, Tuple

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
from diffusers import AutoencoderKL
from apps.main.modules.schedulers import RectifiedFlow, SchedulerArgs
from apps.main.modules.transformer import DiffusionTransformer, DiffusionTransformerArgs


@dataclass
class LatentVideoVAEArgs:
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev"
    revision: Optional[str] = None
    variant: Optional[str] = None


@dataclass
class ModelArgs:
    transformer: DiffusionTransformerArgs = field(
        default_factory=DiffusionTransformerArgs
    )
    vae: LatentVideoVAErgs = field(default_factory=LatentVideoVAEArgs)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)


class LatentVideoVAE(nn.Module):
    def __init__(self, args: LatentVideoVAEArgs):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        self.vae = vae
        self.vae = self.vae.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vae.encode(x).latent_dist.sample()
        x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(x, return_dict=False)[0]
        return image

    @torch.no_grad()
    def forward(self, x=torch.Tensor):
        x = self.encode(x)
        return self.decode(x)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()


class LatentDiffusionTransformer(nn.Module):
    """
    Latent Diffusion Transformer Model for Long Video Generation.
    This model integrates a VAE for latent compression, a transformer for temporal and spatial token mixing,
    and a custom scheduler for diffusion steps.
    """

    version: str = "v0.2"
    description: str = "Latent Diffusion Transformer for VideoGen."

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.transformer = DiffusionTransformer(args.transformer)
        self.compressor = LatentVideoVAE(args.vae)
        self.scheduler = RectifiedFlow(args.scheduler)

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

        # TODO: Mingchen: seems that currently the pipeline is work for image classification task

        return batch, loss

    def init_weights(self, pre_trained_path: Optional[str] = None):
        self.transformer.init_weights(pre_trained_path)


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: DiffusionTransformerArgs, vae_config: dict):
    group_plan: Tuple[int, bool] = []
    # Grouping and output seperately
    # group_plan.append(("tok_embeddings", False))
    for i in range(len(vae_config.down_block_types)):
        group_plan.append((f"compressor.vae.encoder.down_blocks.{i}", False))
    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"transformer.layers.{i}", False))

    group_plan.append(("transformer.img_output", True))

    return group_plan


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(
    model, tp_mesh, model_args: DiffusionTransformerArgs, distributed_args
):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    # Embedding layer tp
    main_plan = {}
    # TODO
    # main_plan["tok_embeddings"] = ColwiseParallel(
    #     input_layouts=Replicate(), output_layouts=Shard(1)
    # )
    main_plan["norm"] = SequenceParallel()
    main_plan["img_output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model.transformer,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    # TODO Adding more for DiT specific Modules
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

        # Feedforward layers tp
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
