import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from apps.Castor.model import Castor, ModelArgs
from apps.Castor.modules.vae import (BaseLatentVideoVAE, VideoVAEArgs,
                                     create_vae)
from lingua.args import dataclass_from_dict
from lingua.checkpoint import (CONSOLIDATE_FOLDER, CONSOLIDATE_NAME,
                               consolidate_checkpoints)
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

from .modules.schedulers import calculate_shift, retrieve_timesteps

logger = logging.getLogger()


@dataclass
class GeneratorArgs:
    guidance_scale: float = 2.0
    resolution: int = 256
    cond_resolution: int = 256
    in_channel: int = 3
    show_progress: bool = False
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"
    sigma: Optional[float] = None
    inference_steps: int = 25
    vae_scale_factor: float = 8.0
    tvae: VideoVAEArgs = field(default_factory=VideoVAEArgs)


class LatentGenerator(nn.Module):
    def __init__(
        self,
        cfg: GeneratorArgs,
        model: nn.Module,
        tvae: BaseLatentVideoVAE,
    ):
        super().__init__()
        self.model = model
        self.vae_scale_factor = cfg.vae_scale_factor
        self.resolution = int(cfg.resolution // self.vae_scale_factor)
        self.cond_resolution = int(cfg.cond_resolution // self.vae_scale_factor)
        self.device = cfg.device
        self.guidance_scale = cfg.guidance_scale
        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]
        self.in_channel = model.diffusion_transformer.in_channels
        self.sigma = cfg.sigma
        self.scheduler = model.scheduler.scheduler
        self.num_inference_steps = cfg.inference_steps
        self.tvae = tvae

    def prepare_latent(self, context, device):
        bsz = len(context["caption"])
        latent_size = (bsz, self.in_channel, self.resolution, self.resolution)
        latents = randn_tensor(latent_size, device=device, dtype=self.dtype)
        return latents

    def return_seq_len(self):
        return (self.resolution // self.model.diffusion_transformer.patch_size) ** 2

    @torch.no_grad()
    def forward(self, context: Dict[str, Any]) -> torch.Tensor:
        cur_device = next(self.model.parameters()).device
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
            cur_device,
            sigmas=sigmas,
            mu=mu,
        )
        latent = self.prepare_latent(context, device=cur_device)
        context['caption'] = context['caption'] + ["" for _ in context['caption']]
        context, context_mask = self.model.text_encoder(context)
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latent] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model.diffusion_transformer(
                x=latent_model_input,
                time_steps=timestep,
                condition=context,
                condition_mask=context_mask,
            ).output
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latent = self.scheduler.step(noise_pred, t, latent, return_dict=False)[0]

        # latent = latent / self.model.compressor.vae.config.scaling_factor
        image = self.tvae.decode(latent)
        return image


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU. from https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/torch_utils.py
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = (
            generator.device.type
            if not isinstance(generator, list)
            else generator[0].device.type
        )
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents


def load_consolidated_model(
    consolidated_path,
    model_cls,
    model_args_cls,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    model = model_cls(model_args)
    consolidated_path = consolidate_checkpoints(ckpt_path)
    st_dict = torch.load(Path(consolidated_path) / CONSOLIDATE_NAME, weights_only=True)
    model.load_state_dict(st_dict["model"])
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, config


def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load(cfg.config)
    gen_cfg = dataclass_from_dict(GeneratorArgs, cfg.generator, strict=False)
    print(gen_cfg)
    pollux, _ = load_consolidated_model(
        cfg.ckpt_dir, model_cls=Castor, model_args_cls=ModelArgs
    )
    tvae = create_vae(gen_cfg.tvae)
    generator = LatentGenerator(gen_cfg, pollux, tvae).cuda()
    print("Model loaded successfully")
    context = {
        "caption": [
            # Short, simple descriptions
            "A red rose in full bloom against a black background.",
            "A happy young man sitting on a piece of cloud, reading a book.",
            "A sleeping cat curled up on a windowsill.",
            "Fresh snow falling in a forest.",
            "Hot air balloons floating in a clear blue sky.",
            
            # Medium length, more detailed
            "A cozy coffee shop interior with vintage furniture, warm lighting, and the aroma of freshly ground beans wafting through the air.",
            "An ancient temple hidden in a misty mountain valley, its weathered stone walls covered in flowering vines.",
            "A bustling night market in Tokyo, neon signs reflecting off wet streets as people hurry past food stalls.",
            "A sea turtle glides gracefully through crystal-clear turquoise water above a school of small fish, with sunlight reflecting off the surface.",
            "A petri dish with a bamboo forest growing within it that has tiny red pandas running around.",
            
            # Technical/scientific
            "A lone astronaut floats in space, gazing at a swirling black hole surrounded by vibrant landscapes, rivers and clouds below.",
            "Microscopic view of CRISPR gene editing in action, with precisely rendered molecular structures.",
            "A topographical map of an alien planet's surface, complete with elevation data and geological formations.",
            
            # Artistic/abstract
            "An impressionist painting of music made visible, with colorful swirls representing different instruments in an orchestra.",
            "A surreal landscape where books grow like trees and their pages flutter like leaves in the wind.",
            "Geometric patterns inspired by Islamic art transforming into modern digital glitch aesthetics.",
            
            # Long, elaborate narratives
            "A photorealistic scene of a centuries-old lighthouse perched on a weathered cliff face during a violent storm at dusk, waves crashing against rocks while lightning illuminates dark clouds.",
            "In a gravity-defying scene, a library exists where books float upward instead of falling down, and readers walk on the ceiling while elderly librarians glide through the air using umbrellas.",
            "A vast colony of king penguins densely populates a rocky shore, with numerous penguins standing closely together against a backdrop of ocean waves.",
            "A picturesque riverside scene featuring a medieval castle atop a hill, surrounded by vibrant autumn foliage, with colorful homes lining the waterfront and several boats docked along the river.",
            "A fantastical underwater city where bioluminescent creatures provide light, and buildings are grown from living coral, with merfolk going about their daily lives among floating gardens."
        ]
    }
    # Start generation
    start_time = time.time()
    samples = generator(context)
    end_time = time.time()

    # Calculate tokens per second
    save_image(samples, "sample.png", nrow=3, normalize=True, value_range=(-1, 1))
    logger.info(f"inference time is {end_time-start_time} seconds")


if __name__ == "__main__":
    main()
