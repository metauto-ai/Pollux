# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Tuple, Optional
import logging
import random
import torch
from torch import nn
import torch.nn.functional as F
from apps.main.modules.gen_transformer import ModelArgs as PolluxGenArgs
from apps.main.modules.gen_transformer import LatentPollux_Gen

from apps.main.modules.plan_transformer import ModelArgs as PolluxPlanArgs
from apps.main.modules.plan_transformer import Latent_Pollux_Plan

from apps.main.modules.vae import build_vae, LatentVideoVAEArgs

logger = logging.getLogger()


@dataclass
class ModelArgs:
    gen: PolluxGenArgs = field(default_factory=PolluxGenArgs)
    plan: PolluxPlanArgs = field(default_factory=PolluxPlanArgs)
    plan_weight: float = 1.0
    gen_weight: float = 1.0
    with_vae: bool = False
    vae_args: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    pre_trained_weight: Optional[str] = None


class Latent_Pollux(nn.Module):
    VERSION: str = "v1.0"
    DESCRIPTION: str = "Latent ImageGen"

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert (
            args.plan.llm.dim == args.gen.gen_transformer.plan_transformer_dim
        ), "Plan and Gen transformer dimensions must match"

        self.plan_model = Latent_Pollux_Plan(args.plan)
        self.gen_model = LatentPollux_Gen(args.gen)
        if args.with_vae:
            self.compressor = build_vae(args.vae_args)
        self.plan_weight = args.plan_weight
        self.gen_weight = args.gen_weight

    def forward(self, batch: dict[str:any]) -> dict[str:any]:
        if hasattr(self, "compressor"):
            print("batch:" ,batch["image"].shape)
            print("image_cond batch:", batch["image_cond"].shape)
            batch["gen_latent_code"] = self.compressor.encode(batch["image"])
            batch["plan_latent_code"] = self.compressor.encode(batch["image_cond"])
        with torch.set_grad_enabled(self.plan_model.is_train):
            plan_output, plan_loss = self.plan_model(batch)
        with torch.set_grad_enabled(self.gen_model.is_train):
            gen_out_put, gen_loss = self.gen_model(plan_output)
        if self.plan_model.is_train and self.gen_model.is_train:
            loss = self.plan_weight * plan_loss + self.gen_weight * gen_loss
        elif self.plan_model.is_train:
            loss = plan_loss
        elif self.gen_model.is_train:
            loss = gen_loss
        else:
            raise ValueError("Both plan and gen models are in eval mode")
        return gen_out_put, loss

    def set_train(self):
        self.plan_model.train()
        self.gen_model.train()

    def set_eval(self):
        self.plan_model.eval()
        self.gen_model.eval()

    def init_weights(self, args: ModelArgs):
        if args.pre_trained_weight:
            args.plan.llm.pre_trained_path = None
            args.gen.gen_transformer.pre_trained_path = None
            self.plan_model.init_weights(args=args.plan)
            self.gen_model.init_weights(args=args.gen)
            logger.info(f"Loading pre-trained weights from {args.pre_trained_weight}")
            pre_trained_state_dict = torch.load(args.pre_trained_weight)
            if "model" in pre_trained_state_dict:
                pre_trained_state_dict = pre_trained_state_dict["model"]
            model_dict = self.state_dict()

            # Filter out non-matching keys
            filtered_dict = {
                k: v for k, v in pre_trained_state_dict.items() if k in model_dict
            }
            not_loaded_keys = set(pre_trained_state_dict.keys()) - set(
                filtered_dict.keys()
            )
            logger.warning(
                f"Keys not loaded from pre-trained weights: {not_loaded_keys}"
            )
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
        else:
            self.plan_model.init_weights(args=args.plan)
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
    for i in range(model_args.plan.llm.n_layers):
        group_plan.append((f"plan_model.llm.layers.{i}", False))
    group_plan.append(("plan_model.latent_head", True))
    for i in range(model_args.gen.gen_transformer.n_layers):
        group_plan.append((f"gen_model.gen_transformer.layers.{i}", False))
    group_plan.append(("gen_model.gen_transformer.img_output", True))
    logger.info(f"The `group_plan` for fsdp is:\n{group_plan}")

    return group_plan


def tp_parallelize(model, tp_mesh, model_args: ModelArgs, distributed_args):
    pass
