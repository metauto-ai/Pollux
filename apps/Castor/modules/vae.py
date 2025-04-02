from dataclasses import dataclass, field
from typing import Optional, Literal, Type, Dict, List, Tuple, TypeVar, Generic
import logging
import torch
from torch import nn
from diffusers import AutoencoderKLHunyuanVideo

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class VideoVAEArgs:
    pretrained_model_name_or_path: str = "tencent/HunyuanVideo"
    revision: Optional[str] = None
    variant: Optional[str] = None
    enable_tiling: bool = True
    enable_slicing: bool = True

class BaseLatentVideoVAE(nn.Module):
    def __init__(self, args: VideoVAEArgs):
        super().__init__()
        self.cfg = args

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def enable_vae_slicing(self):
        logger.warning(
            f"Useless func call, {self.cfg.model_name} TVAE model doesn't support slicing !"
        )
        pass

    def disable_vae_slicing(self):
        logger.warning(
            f"Useless func call, {self.cfg.model_name} TVAE model doesn't support slicing !"
        )
        pass

    def enable_vae_tiling(self):
        logger.warning(
            f"Useless func call, {self.cfg.model_name} TVAE model doesn't support tiling !"
        )
        pass

    def disable_vae_tiling(self):
        logger.warning(
            f"Useless func call, {self.cfg.model_name} TVAE model doesn't support tiling !"
        )
        pass

class HunyuanVideoVAE(BaseLatentVideoVAE):
    def __init__(self, args: VideoVAEArgs):
        super().__init__(args)
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            revision=self.cfg.revision,
            variant=self.cfg.variant,
        ).requires_grad_(False)

        # Configure tiling and slicing
        if self.cfg.enable_slicing:
            vae.enable_slicing()
        else:
            vae.disable_slicing()

        if self.cfg.enable_tiling:
            vae.enable_tiling()
        else:
            vae.disable_tiling()
        self.vae = vae

    # TODO: jinjie: we are using video vae for BCHW image generation, so below code is tricky
    # we need to refactor our dataloader once video gen training begins
    # only feed vae with 5d tensor BCTHW
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.vae.dtype)
        if x.ndim == 4:  # Check if the input tensor is 4D, BCHW, image tensor
            x = x.unsqueeze(2)  # Add a temporal dimension (T=1) for video vae
        x = self.vae.encode(x).latent_dist.sample()
        if x.ndim == 5 and x.shape[2] == 1:  # Check if T=1
            x = x.squeeze(2)  # Remove the temporal dimension at index 2
        x = x * self.vae.config.scaling_factor
        return x  # return 4d image tensor now

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:  # Check if the input tensor is 4D, BCHW, image tensor
            x = x.unsqueeze(2)  # Add a temporal dimension (T=1) for video vae
        x = x.to(self.vae.dtype)
        x = x / self.vae.config.scaling_factor
        x = self.vae.decode(x).sample
        if x.ndim == 5 and x.shape[2] == 1:  # Check if T=1
            x = x.squeeze(2)  # Remove the temporal dimension at index 2
        return x  # return 4d image tensor now

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
