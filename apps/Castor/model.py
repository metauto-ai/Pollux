# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .modules.schedulers import RectifiedFlow, SchedulerArgs
from .modules.text_encoder import TextEncoderArgs, create_text_encoder
from .modules.transformer import DiffusionTransformer, TransformerArgs
from .modules.vae import VideoVAEArgs, create_vae
from .modules.vision_encoder import VisionEncoderArgs, create_vision_encoder

logger = logging.getLogger()


@dataclass
class ModelArgs:
    diffusion_model: TransformerArgs = field(default_factory=TransformerArgs)
    with_vae: bool = False
    vae_args: VideoVAEArgs = field(default_factory=VideoVAEArgs)
    vision_encoder_alignment: bool = False
    vision_encoder_alignment_factor: float = 0.5
    vision_encoder_args: VisionEncoderArgs = field(default_factory=VisionEncoderArgs)
    pre_trained_weight: Optional[str] = None
    text_encoder: TextEncoderArgs = field(default_factory=TextEncoderArgs)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)
    text_cfg_ratio: float = 0.1


@dataclass
class CastorModelOutputs:
    batch: dict[str:any]
    loss: torch.Tensor
    target_loss: torch.Tensor
    align_loss: Optional[torch.Tensor] = None
    forward_timings: Optional[dict[str:float]] = None

class AlignmentProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, encoder_dim: int):
        super(AlignmentProjection, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, encoder_dim),
            )
        )
        
    def forward(self, x):
        x = self.proj(x)
        return x


class Castor(nn.Module):
    VERSION: str = "v1.0"
    DESCRIPTION: str = "Latent ImageGen"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.diffusion_transformer = DiffusionTransformer(args.diffusion_model)
        if args.with_vae:
            self.compressor = create_vae(args.vae_args)
        if args.vision_encoder_alignment:
            self.vision_encoder = create_vision_encoder(args.vision_encoder_args)
            self.vision_encoder_proj = AlignmentProjection(
                args.diffusion_model.dim, args.vision_encoder_args.projection_hidden_dim, self.vision_encoder.dim)

        self.text_encoder = create_text_encoder(args.text_encoder)
        self.scheduler = RectifiedFlow(args.scheduler)
        self.text_cfg_ratio = args.text_cfg_ratio

    def forward(self, batch: dict[str:any]) -> dict[str:any]:
        # Latent extraction timing
        if hasattr(self, "compressor"):
            batch["latent_code"] = self.compressor.extract_latents(batch)

        # Vision encoder timing
        if hasattr(self, "vision_encoder"):
            batch["vision_encoder_target"] = self.vision_encoder.extract_image_representations(batch)

        # Text embedding timing
        if "text_embedding" not in batch:
            batch["text_embedding"], batch["attention_mask"] = self.text_encoder(batch)
        
        conditional_signal, conditional_mask = batch["text_embedding"], batch["attention_mask"]

        if random.random() <= self.text_cfg_ratio:
            conditional_signal = self.diffusion_transformer.negative_token.repeat(
                conditional_signal.size(0), conditional_signal.size(1), 1
            )
            conditional_mask = torch.ones_like(
                conditional_mask, dtype=conditional_signal.dtype
            )
        
        latent_code = batch["latent_code"]
        
        noised_x, t, target = self.scheduler.sample_noised_input(latent_code)
        
        # Diffusion transformer timing
        output = self.diffusion_transformer(
            x=noised_x,
            time_steps=t,
            condition=conditional_signal,
            condition_mask=conditional_mask,
        )
        
        batch["prediction"] = output.output
        batch["target"] = target
        
        # Loss calculation timing
        target_loss = self.mse_loss(output.output, batch["target"])
        
        align_loss = None
        if hasattr(self, "vision_encoder"):
            vision_encoder_pred = self.vision_encoder_proj(output.align_hidden_state)
            align_loss = self.consine_loss_with_features(
                vision_encoder_pred, output.cond_l, output.img_size, batch["vision_encoder_target"])
        
        # Synchronize once at the end and calculate all timings
        torch.cuda.synchronize()
        
        return CastorModelOutputs(
            batch=batch,
            loss=(target_loss + self.args.vision_encoder_alignment_factor * align_loss) if align_loss is not None else target_loss,
            target_loss=target_loss,
            align_loss=align_loss,
        )

    def mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(target, list) or isinstance(output, list):
            loss_list = [F.mse_loss(o, t.to(o.dtype)) for o, t in zip(output, target)]
            return torch.mean(torch.stack(loss_list))
        else:
            target = target.to(output.dtype)
            return F.mse_loss(output, target)

    def consine_loss_with_features(self, x: torch.Tensor, cond_l: torch.Tensor, 
                                   img_size: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pH = pW = self.diffusion_transformer.patch_size
        H, W = img_size
        use_dynamic_res = isinstance(H, list) and isinstance(W, list)
        
        def _cosine_loss(x, t):
            return 1 - F.cosine_similarity(x, t.to(x.dtype), dim=-1).mean()

        if use_dynamic_res:
            # Extract features for each image based on its resolution
            return torch.stack([
                _cosine_loss(
                    x[i, cond_l[i]:cond_l[i] + (_H // pH) * (_W // pW)], 
                    target[i]
                )
                for i, (_H, _W) in enumerate(zip(H, W))
            ]).mean()
        else:
            # Handle condition length consistently whether it's a list or scalar
            offset = max(cond_l) if isinstance(cond_l, list) else cond_l
            feature_length = (H // pH) * (W // pW)
            img_features = x[:, offset:offset + feature_length]
            return _cosine_loss(img_features, target)

    def set_train(self):
        self.diffusion_transformer.train()

    def set_eval(self):
        self.diffusion_transformer.eval()

    def init_weights(self, args: ModelArgs):
        if args.pre_trained_weight:
            self.diffusion_transformer.init_weights()
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
    # if model_args.with_vae:
    #     for i in range(4):  # Specific for Hunyuan's VAE
    #         group_plan.append((f"compressor.vae.encoder.down_blocks.{i}", False))
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
    text_encoder = TextEncoderArgs()
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
