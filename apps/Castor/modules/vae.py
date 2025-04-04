from dataclasses import dataclass, field
from typing import Optional, Literal, Type, Dict, List, Tuple, TypeVar, Generic
import logging
import torch
from torch import nn
from diffusers import AutoencoderKLHunyuanVideo, AutoencoderKL
from cosmos_tokenizer.image_lib import ImageTokenizer


logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class VideoVAEArgs:
    vae_type: str = "hunyuan"  # can be "hunyuan" or "flux"
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


class FluxVAE(BaseLatentVideoVAE):
    def __init__(self, args: VideoVAEArgs):
        super().__init__(args)
        self.vae = AutoencoderKL.from_pretrained(self.cfg.pretrained_model_name_or_path).requires_grad_(False)
        self.scale = 0.3611
        self.shift = 0.1159

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (self.vae.encode(x.to(self.vae.dtype)).latent_dist.mode() - self.shift) * self.scale
        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # Reverse the scaling and shifting from encode method
        x = (x / self.scale + self.shift).to(self.vae.dtype)
        # Use the VAE's decode method and get the sample
        decoded = self.vae.decode(x).sample
        return decoded

    @torch.no_grad()
    def forward(self, x=torch.Tensor):
        x = self.encode(x)
        return self.decode(x)
    

class COSMOSContinuousVAE(BaseLatentVideoVAE):
    def __init__(self, args: VideoVAEArgs):
        super().__init__(args)
        """
        Initialize the encoder and decoder for Continuous VAE.
        Checks model type and returns the initialized VAE instance.
        """
        self.vae = ImageTokenizer(
            checkpoint_enc=f"{self.cfg.pretrained_model_name_or_path}/encoder.jit",
            checkpoint_dec=f"{self.cfg.pretrained_model_name_or_path}/decoder.jit",
        ).requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input frames into latent representations.
        """
        (latent,) = self.vae.encode(x.to(self.vae.dtype))
        return latent

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representations back into reconstructed frames.
        """
        x = self.vae.decode(x.to(self.vae.dtype))
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode and then decode.
        """
        x = self.encode(x)
        return self.decode(x)
    

def create_vae(args: VideoVAEArgs) -> BaseLatentVideoVAE:
    """
    Create a VAE based on the type specified in the args.
    """
    if args.vae_type.lower() == "hunyuan":
        return HunyuanVideoVAE(args)
    elif args.vae_type.lower() == "flux":
        return FluxVAE(args)
    elif args.vae_type.lower() == "cosmos":
        return COSMOSContinuousVAE(args)
    else:
        raise ValueError(f"Unknown VAE type: {args.vae_type}")
