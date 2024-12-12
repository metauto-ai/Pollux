# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import List, Optional

import torch
from torch import nn
from omegaconf import OmegaConf
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
from apps.Simple_DiT.transformer import DiTransformer, DiTransformerArgs
from apps.Simple_DiT.schedulers import RectFlow, SchedulerArgs, retrieve_timesteps
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME, consolidate_checkpoints, CONSOLIDATE_FOLDER

from torchvision.utils import save_image

import logging

logger = logging.getLogger()


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
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


@dataclass
class GeneratorArgs:
    guidance_scale: float = 2.0
    resolution: int = 256
    in_channel: int = 3
    show_progress: bool = False
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"
    sigma: Optional[float] = None
    inference_steps: int = 25


class SimpleGenerator(nn.Module):
    def __init__(self,cfg:GeneratorArgs, model:nn.Module, scheduler:RectFlow):
        super().__init__()
        self.model = model
        self.resolution = cfg.resolution
        self.device = cfg.device
        self.guidance_scale = cfg.guidance_scale
        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]
        self.in_channel = cfg.in_channel
        self.sigma = cfg.sigma
        self.scheduler = scheduler.scheduler
        self.num_inference_steps = cfg.inference_steps
    def prepare_latent(self, context:torch.Tensor):
        bsz = context.size(0)
        latent_size = (bsz,self.in_channel,self.resolution,self.resolution)
        latents = randn_tensor(latent_size, device=context.device, dtype=self.dtype)
        return latents
    def prepare_negative_context(self,context):
        return torch.tensor([self.model.num_classes]*context.size(0)).to(context.device)
    @torch.no_grad()
    def forward(self,context:torch.Tensor)->torch.Tensor:
        timesteps, _ = retrieve_timesteps(self.scheduler, self.num_inference_steps, context.device, sigmas=self.sigma)
        latent = self.prepare_latent(context)
        negative_context = self.prepare_negative_context(context)
        context = torch.cat([context,negative_context])
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latent] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model(x=latent_model_input,time_steps=timestep,context=context)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latent = self.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
        return latent


def load_consolidated_model(
    consolidated_path,
    model_cls=DiTransformer,
    model_args_cls=DiTransformerArgs,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    model = model_cls(model_args)
    consolidate_checkpoints(ckpt_path)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_FOLDER / CONSOLIDATE_NAME, weights_only=True)
    model.load_state_dict(st_dict["model"])
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, config


def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load(cfg.config)
    gen_cfg = dataclass_from_dict(
        GeneratorArgs, cfg, strict=False
    )
    scheduler_cfg = dataclass_from_dict(
        SchedulerArgs, cfg, strict=False
    )
    print(cfg)

    model,_ = load_consolidated_model(cfg.ckpt_dir)
    scheduler = RectFlow(scheduler_cfg)
    generator = SimpleGenerator(gen_cfg, model, scheduler)


    context = torch.tensor([207, 360, 387, 231, 245, 234, 256, 476, 173, 238, 237, 978]).cuda()
    # Start generation
    start_time = time.time()
    samples = generator(context)
    end_time = time.time()

    # Calculate tokens per second
    save_image(samples, "sample.png", nrow=3, normalize=True, value_range=(-1, 1))
    logger.info(f"inference time is {end_time-start_time} seconds")


if __name__ == "__main__":
    main()
