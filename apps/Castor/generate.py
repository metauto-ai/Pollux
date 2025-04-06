import torch
import time
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
from apps.Castor.model import Castor, ModelArgs
from typing import List, Optional, Tuple, Union, Dict, Any
from .modules.schedulers import retrieve_timesteps, calculate_shift
from lingua.args import dataclass_from_dict
import logging
from pathlib import Path
from lingua.checkpoint import (
    CONSOLIDATE_NAME,
    consolidate_checkpoints,
    CONSOLIDATE_FOLDER,
)
from apps.Castor.modules.vae import BaseLatentVideoVAE, create_vae, VideoVAEArgs

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
        pos_conditional_signal, pos_conditional_mask = self.model.text_encoder(context)
        negative_conditional_signal = (
            self.model.diffusion_transformer.negative_token.repeat(
                pos_conditional_signal.size(0), pos_conditional_signal.size(1), 1
            )
        )
        negative_conditional_mask = torch.ones_like(pos_conditional_mask, dtype=pos_conditional_mask.dtype)
        context = torch.cat(
            [
                pos_conditional_signal,
                negative_conditional_signal,
            ]
        )
        context_mask = torch.cat(
            [
                pos_conditional_mask,
                negative_conditional_mask,
            ]
        )
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latent] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model.diffusion_transformer(
                x=latent_model_input,
                time_steps=timestep,
                condition=context,
                condition_mask=context_mask,
            )
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
            "A charming old building with a terracotta roof and white-washed walls stands prominently in a quaint European town. The building features several windows adorned with brown shutters, and a line of freshly laundered clothes hangs from a clothesline outside the second-floor window, adding a touch of domesticity to the scene. The sunlight casts gentle shadows, highlighting the textures of the aged walls and the rustic charm of the setting. The overall composition evokes a sense of nostalgia and simplicity, reminiscent of a bygone era, with the clear blue sky serving as a serene backdrop.",
            "Statue of Pharaonic Queen Cleopatra inside the Egyptian temple done from Limestone, full-body, Pottery utensils and sand dunes cover walls and rocks, painting in watercolor,",
            "On a wooden surface, a can of Positive Beverage Tropical Berry, with its vibrant blue and white label featuring a cheerful illustration of tropical fruits, sits next to a wooden rolling pin and a pile of flour. The can is surrounded by a large, golden-brown gingerbread dough with cutouts of star and gingerbread man shapes, creating a festive and cozy atmosphere. The scene is bathed in soft, natural light, highlighting the textures of the dough and the rustic wooden elements, evoking a warm and inviting holiday spirit.",
            "A serene photograph capturing the golden reflection of the sun on a vast expanse of water. The sun is positioned at the top center, casting a brilliant, shimmering trail of light across the rippling surface. The water is textured with gentle waves, creating a rhythmic pattern that leads the eye towards the horizon. The entire scene is bathed in warm, golden hues, enhancing the tranquil and meditative atmosphere. High contrast, natural lighting, golden hour, photorealistic, expansive composition, reflective surface, peaceful, visually harmonious.",
            "An angry duck doing heavy weightlifting at the gym.",
            "Mona Lisa in winter",
            "A corgi.",
            "graffiti of a panda with snow goggles snowboarding on a street wall",
            "A group of three teddy bears in suit in an office celebrating the birthday of their friend. There is a pizza cake on the desk",
            "A photo of a smiling person with snow goggles on holding a snowboard",
            "Photo of a bear catching salmon",
            "an astronaut rides a pig through in the forest. next to a river, with clouds in the sky",
            "Hot air balloons and flowers, collage art, photorealism, muted colors, 3D shading beautiful eldritch, mixed media, vaporous",
            "A close-up photo of a baby sloth holding a treasure chest. A warm, golden light emanates from within the chest, casting a soft glow on the sloth's fur and the surrounding rainforest foliage.",
            "A photo of a cat playing chess.",
            "A cloud dragon flying over mountains, its body swirling with the wind",
            "a clock on a desk, ghibli style",
            "Temple in ruins, epic, forest, stairs, columns, cinematic, detailed, atmospheric, epic, concept art, matte painting, background, mist, photo-realistic, concept art, volumetric light, cinematic epic, 8k",
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
