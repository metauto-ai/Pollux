from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from diffusers import AutoencoderKL

@dataclass
class LatentVideoVAEArgs:
    pretrained_model_name_or_path: str = "black-forest-labs/FLUX.1-dev"
    revision: Optional[str] = None
    variant: Optional[str] = None


class LatentVideoVAE(nn.Module):
    def __init__(self, args: LatentVideoVAEArgs):
        super().__init__()
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        self.vae = vae
        self.vae = self.vae.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vae.encode(x).latent_dist.sample()
        x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(x, return_dict=False)[0]
        return image

    @torch.no_grad()
    def forward(self, x=torch.Tensor):
        x = self.encode(x)
        return self.decode(x)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()
