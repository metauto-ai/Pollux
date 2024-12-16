import torch
import time
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
from apps.main.model import LatentDiffusionTransformer, ModelArgs
from apps.Simple_DiT.generate import (
    GeneratorArgs,
    load_consolidated_model,
    randn_tensor,
)
from apps.Simple_DiT.schedulers import RectFlow, retrieve_timesteps, calculate_shift
from lingua.args import dataclass_from_dict
import logging

logger = logging.getLogger()


class LatentGenerator(nn.Module):
    def __init__(self, cfg: GeneratorArgs, model: nn.Module):
        super().__init__()
        self.model = model
        self.vae_scale_factor = (
            2 ** (len(self.model.compressor.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.resolution = cfg.resolution // self.vae_scale_factor
        self.device = cfg.device
        self.guidance_scale = cfg.guidance_scale
        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]
        self.in_channel = model.transformer.in_channels
        self.sigma = cfg.sigma
        self.scheduler = model.scheduler.scheduler
        self.num_inference_steps = cfg.inference_steps

    def prepare_latent(self, context: torch.Tensor):
        bsz = context.size(0)
        latent_size = (bsz, self.in_channel, self.resolution, self.resolution)
        latents = randn_tensor(latent_size, device=context.device, dtype=self.dtype)
        return latents

    def prepare_negative_context(self, context):
        return torch.tensor([self.model.transformer.num_classes] * context.size(0)).to(
            context.device
        )

    def return_seq_len(self):
        return (self.resolution // self.model.transformer.patch_size) ** 2

    @torch.no_grad()
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        latent = self.prepare_latent(context)
        image_seq_len = self.return_seq_len()
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        sigmas = (
            np.linspace(1.0, 1 / self.num_inference_steps, self.num_inference_steps)
            if self.sigma is None
            else self.sigma
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            self.num_inference_steps,
            context.device,
            sigmas=sigmas,
            mu=mu,
        )
        negative_context = self.prepare_negative_context(context)
        context = torch.cat([context, negative_context])

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latent] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model.transformer(
                x=latent_model_input, time_steps=timestep, condition=context
            )
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latent = self.scheduler.step(noise_pred, t, latent, return_dict=False)[0]

        latent = (
            latent / self.model.compressor.vae.config.scaling_factor
        ) + self.model.compressor.vae.config.shift_factor
        image = self.model.compressor.decode(latent)
        return image


def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load(cfg.config)
    gen_cfg = dataclass_from_dict(GeneratorArgs, cfg, strict=False)
    print(cfg)

    model, _ = load_consolidated_model(
        cfg.ckpt_dir, model_cls=LatentDiffusionTransformer, model_args_cls=ModelArgs
    )

    generator = LatentGenerator(gen_cfg, model)

    context = torch.tensor(
        [207, 360, 387, 231, 245, 234, 256, 476, 173, 238, 237, 978]
    ).cuda()
    # Start generation
    start_time = time.time()
    samples = generator(context)
    end_time = time.time()

    # Calculate tokens per second
    save_image(samples, "sample.png", nrow=3, normalize=True, value_range=(-1, 1))
    logger.info(f"inference time is {end_time-start_time} seconds")


if __name__ == "__main__":
    main()
