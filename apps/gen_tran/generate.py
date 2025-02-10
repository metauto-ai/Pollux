import torch
import time
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
from apps.gen_tran.model import Pollux, LatentPollux, ModelArgs
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
from apps.main.modules.tokenizer import Tokenizer, TokenizerArgs
from apps.main.modules.text_encoder import LLAMATransformerArgs, LLAMA3
from apps.main.modules.vae import BaseLatentVideoVAE, build_vae, LatentVideoVAEArgs

logger = logging.getLogger()


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
    vae_scale_factor: float = 8.0
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    text_encoder: LLAMATransformerArgs = field(default_factory=LLAMATransformerArgs)
    tvae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)


class LatentGenerator(nn.Module):
    def __init__(
        self,
        cfg: GeneratorArgs,
        model: nn.Module,
        tokenizer: Tokenizer,
        text_encoder: LLAMA3,
        tvae: BaseLatentVideoVAE,
    ):
        super().__init__()
        self.model = model
        self.vae_scale_factor = cfg.vae_scale_factor
        self.resolution = int(cfg.resolution // self.vae_scale_factor)
        self.device = cfg.device
        self.guidance_scale = cfg.guidance_scale
        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]
        self.in_channel = model.gen_transformer.in_channels
        self.sigma = cfg.sigma
        self.scheduler = model.scheduler.scheduler
        self.num_inference_steps = cfg.inference_steps
        self.tvae = tvae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def prepare_latent(self, context, device):
        bsz = len(context["caption"])
        latent_size = (bsz, self.in_channel, self.resolution, self.resolution)
        latents = randn_tensor(latent_size, device=device, dtype=self.dtype)
        return latents

    @torch.no_grad()
    def prepare_negative_context(self, context):
        conditional_signal = self.model.negative_token.repeat(
            context["positive_text_embedding"].size(0),
            context["positive_text_embedding"].size(1),
            1,
        )
        context["negative_text_embedding"] = conditional_signal.to(self.dtype)
        return context

    @torch.no_grad()
    def prepare_positive_context(self, context):
        context["cap_token"] = []
        for x in context["caption"]:
            if not isinstance(x, str):
                logger.warning(f"Expected string but got {type(x)}: {x}")
                context["cap_token"].append(
                    self.tokenizer.encode("", bos=True, eos=False)
                )
            else:
                context["cap_token"].append(
                    self.tokenizer.encode(x, bos=True, eos=False)
                )

        pad_id = self.tokenizer.pad_id
        bsz = len(context["cap_token"])
        tokens = torch.full(
            (bsz, self.text_encoder.text_seqlen),
            pad_id,
            dtype=torch.long,
        ).cuda()
        for k, t in enumerate(context["cap_token"]):
            if len(t) < tokens.size(1):
                tokens[k, : len(t)] = torch.tensor(
                    t[:], dtype=torch.long, device="cuda"
                )
            else:
                tokens[k, :] = torch.tensor(
                    t[: tokens.size(1)], dtype=torch.long, device="cuda"
                )
        context["cap_token"] = tokens.cuda()
        context["positive_text_embedding"] = self.text_encoder(context).to(self.dtype)
        return context

    def return_seq_len(self):
        return (self.resolution // self.model.gen_transformer.patch_size) ** 2

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
        context = self.prepare_positive_context(context)
        context = self.prepare_negative_context(context)
        negative_conditional_signal = self.model.token_proj(
            context["negative_text_embedding"]
        )
        pos_conditional_signal = self.model.token_proj(
            context["positive_text_embedding"]
        )

        context = torch.cat([pos_conditional_signal, negative_conditional_signal])
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latent] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model.gen_transformer(
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
    text_encoder = LLAMA3(gen_cfg.text_encoder)
    diffusion_model, _ = load_consolidated_model(
        cfg.ckpt_dir, model_cls=LatentPollux, model_args_cls=ModelArgs
    )
    tokenizer = Tokenizer(model_path=gen_cfg.tokenizer.model_path)
    tvae = build_vae(gen_cfg.tvae)
    generator = LatentGenerator(
        gen_cfg, diffusion_model, tokenizer, text_encoder, tvae
    ).cuda()
    # param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
    #     gen_cfg.dtype
    # ]
    # for param in generator.parameters():
    #     param.data = param.data.to(dtype=param_dtype)
    print("Model loaded successfully")
    context = {
        "caption": [
            "In a modern urban setting, a large blue planter box with the words MediaCityUK prominently displayed on its sides stands as a focal point in a paved outdoor area. The planter box is filled with lush green foliage, adding a touch of nature to the otherwise concrete and steel environment. Behind the planter, a row of bicycles is neatly parked, indicating a possible bike-sharing scheme. The scene is set against a backdrop of high-rise buildings with glass facades, reflecting the cloudy sky above. The overall mood is one of contemporary urbanity, with a hint of tranquility provided by the greenery and the peaceful atmosphere of the area.",
            "In a modern urban setting, a large blue planter box with the words MediaCityUK prominently displayed on its sides stands as a focal point in a paved outdoor area. The planter box is filled with lush green foliage, adding a touch of nature to the otherwise concrete and steel environment. Behind the planter, a row of bicycles is neatly parked, indicating a possible bike-sharing scheme. The scene is set against a backdrop of high-rise buildings with glass facades, reflecting the cloudy sky above. The overall mood is one of contemporary urbanity, with a hint of tranquility provided by the greenery and the peaceful atmosphere of the area.",
            "In a modern urban setting, a large blue planter box with the words MediaCityUK prominently displayed on its sides stands as a focal point in a paved outdoor area. The planter box is filled with lush green foliage, adding a touch of nature to the otherwise concrete and steel environment. Behind the planter, a row of bicycles is neatly parked, indicating a possible bike-sharing scheme. The scene is set against a backdrop of high-rise buildings with glass facades, reflecting the cloudy sky above. The overall mood is one of contemporary urbanity, with a hint of tranquility provided by the greenery and the peaceful atmosphere of the area.",
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
