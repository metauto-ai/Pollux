# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import logging
import random
import torch
from torch import nn
import torch.nn.functional as F
from .modules.transformer import TransformerArgs
from .modules.transformer import DiffusionTransformer

from .modules.vae import HunyuanVideoVAE, VideoVAEArgs
from .modules.text_encoder import CLIP, CLIPArgs
from .modules.schedulers import RectifiedFlow, SchedulerArgs

logger = logging.getLogger()


@dataclass
class ModelArgs:
    diffusion_model: TransformerArgs = field(default_factory=TransformerArgs)
    with_vae: bool = False
    vae_args: VideoVAEArgs = field(default_factory=VideoVAEArgs)
    pre_trained_weight: Optional[str] = None
    text_encoder: CLIPArgs = field(default_factory=CLIPArgs)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)
    text_cfg_ratio: float = 0.1


class Castor(nn.Module):
    VERSION: str = "v1.0"
    DESCRIPTION: str = "Latent ImageGen"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.diffusion_transformer = DiffusionTransformer(args.diffusion_model)
        if args.with_vae:
            self.compressor = HunyuanVideoVAE(args.vae_args)
        self.text_encoder = CLIP(args.text_encoder)
        self.scheduler = RectifiedFlow(args.scheduler)
        self.text_cfg_ratio = args.text_cfg_ratio

    def forward(self, batch: dict[str:any]) -> dict[str:any]:
        if "text_embedding" not in batch:
            batch["text_embedding"] = self.text_encoder(batch)
        conditional_signal = batch["text_embedding"]
        if random.random() <= self.text_cfg_ratio:
            conditional_signal = self.diffusion_transformer.negative_token.repeat(
                conditional_signal.size(0), conditional_signal.size(1), 1
            )
        if hasattr(self, "compressor"):
            batch["latent_code"] = self.compressor.encode(batch["image"])
        latent_code = batch["latent_code"]
        noised_x, t, target = self.scheduler.sample_noised_input(latent_code)
        output = self.diffusion_transformer(
            x=noised_x, time_steps=t, condition=conditional_signal
        )
        batch["prediction"] = output
        batch["target"] = target
        target = target.to(output.dtype)
        loss = F.mse_loss(output, target)
        return batch, loss

    def set_train(self):
        self.diffusion_transformer.train()

    def set_eval(self):
        self.diffusion_transformer.eval()

    def init_weights(self, args: ModelArgs):
        if args.pre_trained_weight:
            args.diffusion_model.pre_trained_path = None
            self.diffusion_transformer.init_weights(args=args.diffusion_model)
            logger.info(f"Loading pre-trained weights from {args.pre_trained_weight}")
            pre_trained_state_dict = torch.load(args.pre_trained_weight)
            if "model" in pre_trained_state_dict:
                pre_trained_state_dict = pre_trained_state_dict["model"]
            self.load_state_dict(pre_trained_state_dict)
        else:
            self.diffusion_transformer.init_weights(
                pre_trained_path=args.diffusion_model.pre_trained_path
            )
    
    def get_checkpointing_wrap_module_list(self) -> List[nn.Module]:
        return list(self.diffusion_transformer.layers)


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: ModelArgs):
    group_plan: Tuple[int, bool] = []
    if model_args.with_vae:
        for i in range(4):  # Specific for Hunyuan's VAE
            group_plan.append((f"compressor.vae.encoder.down_blocks.{i}", False))
    for i in range(model_args.diffusion_model.n_layers):
        group_plan.append((f"diffusion_transformer.layers.{i}", False))
    group_plan.append(("diffusion_transformer.img_output", True))
    logger.info(f"The `group_plan` for fsdp is:\n{group_plan}")

    return group_plan


def tp_parallelize(model, tp_mesh, model_args: ModelArgs, distributed_args):
    pass


def build_2B_Castor():
    diffusion_transformer = TransformerArgs(
        dim=2048,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        n_heads=32,
        n_kv_heads=8,
        n_layers=24,
        time_step_dim=2048,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        tmb_size=256,
        gen_seqlen=16,
        condition_seqlen=128,
        norm_eps=1e-5,
        condition_dim=512,
    )
    vae_args = VideoVAEArgs(
        pretrained_model_name_or_path="/mnt/pollux/checkpoints/HunyuanVideo/vae",
        enable_tiling=False,
        enable_slicing=False,
    )
    text_encoder = CLIPArgs()
    scheduler = SchedulerArgs(
        num_train_timesteps=1000,
        base_image_seq_len=256,
        base_shift=0.5,
        max_image_seq_len=4096,
        max_shift=1.15,
        shift=1.0,  # need consider 3.0 or 1.0
        weighting_scheme="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        use_dynamic_shifting=True,
    )
    model_arg = ModelArgs(
        with_vae=False,
        vae_args=vae_args,
        text_encoder=text_encoder,
        scheduler=scheduler,
        text_cfg_ratio=1.0,
        diffusion_model=diffusion_transformer,
    )
    return Castor(model_arg)


if __name__ == "__main__":
    model = build_2B_Castor()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Params: {total_params / 1e9:.2f}B")
    print(f"Trainable Params: {trainable_params / 1e9:.2f}B")
