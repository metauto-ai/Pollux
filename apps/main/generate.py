import torch
import time
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
from apps.main.model import Latent_Pollux, ModelArgs
from typing import List, Optional, Tuple, Union, Dict, Any
from apps.main.modules.schedulers import retrieve_timesteps, calculate_shift
from lingua.args import dataclass_from_dict
import logging
from pathlib import Path
from lingua.checkpoint import (
    CONSOLIDATE_NAME,
    consolidate_checkpoints,
    CONSOLIDATE_FOLDER,
)
import PIL
from apps.main.modules.vae import BaseLatentVideoVAE, build_vae, LatentVideoVAEArgs
from apps.main.modules.preprocess import center_crop_arr
import torchvision.transforms as transforms
import requests
from io import BytesIO

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
    tvae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)


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
        self.in_channel = model.gen_model.gen_transformer.in_channels
        self.sigma = cfg.sigma
        self.scheduler = model.gen_model.scheduler.scheduler
        self.num_inference_steps = cfg.inference_steps
        self.tvae = tvae
        self.model.gen_model.gen_transformer.token_drop_ratio = 0
        self.transform = transforms.Compose(
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        )

    def preprocess_cond_image(self, image):
        image = center_crop_arr(image, self.cond_resolution)
        return self.transform(image)

    def prepare_latent(self, context, device):
        bsz = len(context["caption"])
        latent_size = (bsz, self.in_channel, self.resolution, self.resolution)
        latents = randn_tensor(latent_size, device=device, dtype=self.dtype)
        return latents

    def prepare_cond_latent(self, context, device):
        bsz = len(context["caption"])
        if "image_cond" in context:
            if isinstance(context["image_cond"], PIL.Image.Image):
                context["image_cond"] = self.preprocess_cond_image(
                    context["image_cond"]
                )
            latents = self.tvae.encode(context["image_cond"])
        else:
            latent_size = (
                bsz,
                self.in_channel,
                self.cond_resolution,
                self.cond_resolution,
            )
            latents = randn_tensor(latent_size, device=device, dtype=self.dtype)
        return latents

    @torch.no_grad()
    def prepare_negative_context(self, context):
        conditional_signal = self.model.gen_model.negative_token.repeat(
            context["positive_text_embedding"].size(0),
            context["positive_text_embedding"].size(1),
            1,
        )
        context["negative_text_embedding"] = conditional_signal.to(self.dtype)
        return context

    @torch.no_grad()
    def prepare_positive_context(self, context):
        context["positive_text_embedding"] = self.model.plan_model.encode(context).to(
            self.dtype
        )
        return context

    def return_seq_len(self):
        return (self.resolution // self.model.gen_model.gen_transformer.patch_size) ** 2

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
        context["plan_latent_code"] = self.prepare_cond_latent(
            context, device=cur_device
        )
        context = self.prepare_positive_context(context)
        context = self.prepare_negative_context(context)
        negative_conditional_signal = self.model.gen_model.token_proj(
            context["negative_text_embedding"]
        )
        pos_conditional_signal = self.model.gen_model.token_proj(
            context["positive_text_embedding"]
        )

        context = torch.cat([pos_conditional_signal, negative_conditional_signal])
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latent] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model.gen_model.gen_transformer(
                x=latent_model_input,
                time_steps=timestep,
                condition=context,
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
        cfg.ckpt_dir, model_cls=Latent_Pollux, model_args_cls=ModelArgs
    )
    tvae = build_vae(gen_cfg.tvae)
    generator = LatentGenerator(gen_cfg, pollux, tvae).cuda()
    print("Model loaded successfully")
    context = {
        "caption": [
            "A charming old building with a terracotta roof and white-washed walls stands prominently in a quaint European town. The building features several windows adorned with brown shutters, and a line of freshly laundered clothes hangs from a clothesline outside the second-floor window, adding a touch of domesticity to the scene. The sunlight casts gentle shadows, highlighting the textures of the aged walls and the rustic charm of the setting. The overall composition evokes a sense of nostalgia and simplicity, reminiscent of a bygone era, with the clear blue sky serving as a serene backdrop.",
            "Statue of Pharaonic Queen Cleopatra inside the Egyptian temple done from Limestone, full-body, Pottery utensils and sand dunes cover walls and rocks, painting in watercolor,",
            "On a wooden surface, a can of Positive Beverage Tropical Berry, with its vibrant blue and white label featuring a cheerful illustration of tropical fruits, sits next to a wooden rolling pin and a pile of flour. The can is surrounded by a large, golden-brown gingerbread dough with cutouts of star and gingerbread man shapes, creating a festive and cozy atmosphere. The scene is bathed in soft, natural light, highlighting the textures of the dough and the rustic wooden elements, evoking a warm and inviting holiday spirit.",
        ]
    }
    # Start generation
    start_time = time.time()
    samples = generator(context)
    end_time = time.time()

    # Calculate tokens per second
    save_image(samples, "sample.png", nrow=3, normalize=True, value_range=(-1, 1))
    logger.info(f"inference time is {end_time-start_time} seconds")


def save_images_to_grid(
    images, grid_size=(3, 3), image_size=256, output_path="output_grid.jpg"
):
    if not images:
        print("No images to save.")
        return

    w = h = image_size
    grid_w, grid_h = grid_size

    # Create a blank canvas
    grid_image = PIL.Image.new("RGB", (grid_w * w, grid_h * h))

    for idx, img in enumerate(images):
        img = center_crop_arr(img, image_size)
        x_offset = (idx % grid_w) * w
        y_offset = (idx // grid_w) * h
        grid_image.paste(img, (x_offset, y_offset))

    grid_image.save(output_path)
    print(f"Grid image saved at {output_path}")


def main_inpaint():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load(cfg.config)
    gen_cfg = dataclass_from_dict(GeneratorArgs, cfg.generator, strict=False)
    print(gen_cfg)
    context = {
        "image_cond": [
            "https://ayon.blob.core.windows.net/pexelimages/10954867.jpg",
            "https://ayon.blob.core.windows.net/pexelimages/13776712.jpg",
            "https://ayon.blob.core.windows.net/pexelimages/9281812.jpg",
            "https://ayon.blob.core.windows.net/pexelimages/28111771.jpg",
            "https://ayon.blob.core.windows.net/pexelimages/27948601.jpg",
            "https://ayon.blob.core.windows.net/pexelimages/7647233.jpg",
            "https://ayon.blob.core.windows.net/pexelimages/9709424.jpg",
            "https://ayon.blob.core.windows.net/pexelimages/10423579.jpg",
            "https://ayon.blob.core.windows.net/pexelimages/27536622.jpg",
        ],
        "caption": [
            "An empty asphalt road stretches into the distance, flanked by a dense forest of deciduous trees, some of which are bare while others retain a few yellow leaves, creating a picturesque autumnal scene. The road is smooth and wet, reflecting the soft, diffused light of a partly cloudy day, with the sky transitioning from a pale blue to a lighter hue near the horizon. The composition captures a serene and tranquil mood, with the trees standing tall and proud on either side of the road, their branches reaching out as if to embrace the traveler. The overall scene evokes a sense of calm and solitude, inviting the viewer to imagine a peaceful journey through the forest.",
            "A daring individual, dressed in a black shirt and gray pants, is captured mid-stride on a vibrant tightrope suspended high above the ground, with a mix of red, blue, and green ropes adding a splash of color against the clear blue sky. The performer's right arm is raised in triumph, while his left leg is extended forward, showcasing a moment of balance and skill. The scene is set against a backdrop of fluffy white clouds, creating a striking contrast between the solidity of the tightrope and the ethereal softness of the sky. The overall mood is one of exhilaration and triumph, with the bright sunlight highlighting the performer's silhouette and the taut lines of the ropes.",
            "Under a brooding, overcast sky, a bustling city street at night is captured in a moment of quiet transition. The scene is dominated by a high-rise building with numerous illuminated windows, casting a soft glow against the dark sky. To the left, a smaller building displays a sign reading CLOP in bold letters, while to the right, a modern structure with a lit sign reading Xce stands tall. The street is wet from recent rain, reflecting the city lights and creating a shimmering effect. A car with its headlights on navigates the intersection, while another car follows closely behind. A few pedestrians, some wearing face masks, are visible, walking along the sidewalk. The atmosphere is one of urban life, with",
            "A woman with long, dark hair is sitting cross-legged on a bed, smiling warmly at the camera. She is wearing a white blouse with delicate lace details and beige pants, and she has a necklace with a pendant. The bed is covered with a white sheet, and the headboard is black with a simple design. The room has a muted, gray wall, and the lighting is soft, creating a cozy and intimate atmosphere. The composition is balanced, with the woman centered and the background providing a simple, uncluttered backdrop that emphasizes her relaxed and joyful expression.",
            "A black-and-white photograph captures a solemn moment in a church, focusing on a bride and groom standing before the altar. The bride, with her back turned, wears a flowing gown and has a rose tattoo on her shoulder blade, while the groom stands attentively beside her. Above them, a large, ornate clock-like structure adorns the wall, featuring a statue of a saint in a central position, surrounded by rays of light. The altar is adorned with a white cloth and flanked by two candle stands holding lit candles. The composition is framed by an arched alcove, creating a sense of intimacy and reverence.",
            "A man with curly hair is standing in front of a wall, wearing a white long-sleeve shirt with a blue undershirt that has white text on it. He is smiling and appears to be fixing something on the wall, possibly an electrical outlet, with a focused expression. The wall is covered in a light-colored material, and there are white wires hanging from it. The lighting is soft and natural, casting a gentle glow on the man and the wall, creating a warm and inviting atmosphere."
            "A man is sitting on a blue couch, smiling and holding a plant in his hands, possibly trimming it. He is wearing a white t-shirt and has a beard. In front of him, there is a table with a potted plant, a pair of scissors, and a white container with soil. Behind him, there is a wooden cabinet with a plant on top, and a few chairs are placed near the wall. The room has a warm and cozy atmosphere, with natural light coming from the right side.",
            "A group of seven individuals, all wearing glasses, are playfully posing for a photograph in a minimalist setting. They are leaning against two white walls, creating a frame with their bodies, and their heads are peeking out from the gap between the walls. The individuals are smiling and appear to be in a joyful mood. The background is a simple, pebbled floor, and the lighting is bright, casting soft shadows on the walls. The composition is symmetrical, with the subjects evenly spaced and centered, creating a harmonious and balanced visual effect.",
            "A meticulously crafted hedge maze sprawls across a lush landscape, its vibrant green bushes forming a labyrinthine pattern that invites exploration. In the background, a charming house with a steeply pitched roof and a mix of brick and timber cladding stands as a picturesque focal point. The house's windows reflect the soft, natural light of a clear day, while the surrounding trees, including a cherry blossom tree in delicate pink bloom, add to the serene and picturesque setting. The overall composition is reminiscent of a classical garden, blending elements of nature and human artistry in a harmonious and inviting scene.",
        ],
    }
    images_list = []
    for url in context["image_cond"]:
        response = requests.get(url, timeout=10)  # Add timeout for reliability
        response.raise_for_status()  # Raise error for bad status codes
        image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
        images_list.append(image)
    context["image_cond"] = images_list
    # Start generation
    save_images_to_grid(images_list)
    pollux, _ = load_consolidated_model(
        cfg.ckpt_dir, model_cls=Latent_Pollux, model_args_cls=ModelArgs
    )
    tvae = build_vae(gen_cfg.tvae)
    generator = LatentGenerator(gen_cfg, pollux, tvae).cuda()
    print("Model loaded successfully")
    start_time = time.time()
    samples = generator(context)
    end_time = time.time()

    # Calculate tokens per second
    save_image(samples, "sample.png", nrow=3, normalize=True, value_range=(-1, 1))
    logger.info(f"inference time is {end_time-start_time} seconds")


if __name__ == "__main__":
    # main()
    main_inpaint()
