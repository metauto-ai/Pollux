# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Tuple, Optional
import logging
import random
import torch
from torch import nn
import torch.nn.functional as F
from apps.MMTransformer.modules.gen_transformer import ModelArgs as PolluxGenArgs
from apps.MMTransformer.modules.gen_transformer import LatentPollux_Gen

from apps.main.modules.vae import build_vae, LatentVideoVAEArgs
from apps.main.modules.text_encoder import CLIP, CLIPArgs

logger = logging.getLogger()


@dataclass
class ModelArgs:
    gen: PolluxGenArgs = field(default_factory=PolluxGenArgs)
    gen_weight: float = 1.0
    with_vae: bool = False
    vae_args: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    pre_trained_weight: Optional[str] = None
    text_encoder: CLIPArgs = field(default_factory=CLIPArgs)


class Latent_Pollux(nn.Module):
    VERSION: str = "v1.0"
    DESCRIPTION: str = "Latent ImageGen"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gen_model = LatentPollux_Gen(args.gen)
        if args.with_vae:
            self.compressor = build_vae(args.vae_args)
        self.gen_weight = args.gen_weight
        self.text_encoder = CLIP(args.text_encoder)

    def forward(self, batch: dict[str:any]) -> dict[str:any]:
        batch["text_embedding"] = self.text_encoder(batch)
        if hasattr(self, "compressor"):
            batch["gen_latent_code"] = self.compressor.encode(batch["image"])
        with torch.set_grad_enabled(self.gen_model.is_train):
            gen_out_put, gen_loss = self.gen_model(batch)
        return gen_out_put, gen_loss

    def set_train(self):
        self.gen_model.train()

    def set_eval(self):
        self.gen_model.eval()

    def init_weights(self, args: ModelArgs):
        if args.pre_trained_weight:
            args.gen.gen_transformer.pre_trained_path = None
            self.gen_model.init_weights(args=args.gen)
            logger.info(f"Loading pre-trained weights from {args.pre_trained_weight}")
            pre_trained_state_dict = torch.load(args.pre_trained_weight)
            if "model" in pre_trained_state_dict:
                pre_trained_state_dict = pre_trained_state_dict["model"]
            self.load_state_dict(pre_trained_state_dict)
        else:
            self.gen_model.init_weights(args=args.gen)


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: ModelArgs):
    group_plan: Tuple[int, bool] = []
    if model_args.with_vae:
        for i in range(4):  # Specific for Hunyuan's VAE
            group_plan.append((f"compressor.vae.encoder.down_blocks.{i}", False))
    for i in range(model_args.gen.gen_transformer.n_layers):
        group_plan.append((f"gen_model.gen_transformer.layers.{i}", False))
    group_plan.append(("gen_model.gen_transformer.img_output", True))
    logger.info(f"The `group_plan` for fsdp is:\n{group_plan}")

    return group_plan


def tp_parallelize(model, tp_mesh, model_args: ModelArgs, distributed_args):
    pass
