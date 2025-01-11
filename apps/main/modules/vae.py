from dataclasses import dataclass, field
from typing import Optional, Literal, Type, Dict, List, Tuple, TypeVar, Generic
import logging
import torch
from torch import nn
from diffusers import AutoencoderKLHunyuanVideo
import os
import re
import numpy as np
from torchvision import transforms
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from cosmos_tokenizer.networks.configs import (
    discrete_video as discrete_video_dict,
)
from cosmos_tokenizer.networks.configs import (
    continuous_video as continuous_video_dict,
)
import types

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class LatentVideoVAEArgs:
    model_name: str = "COSMOS-DV"  # ["Hunyuan", "COSMOS-DV", "COSMOS-CV"]
    pretrained_model_name_or_path: str = "tencent/HunyuanVideo"
    revision: Optional[str] = None
    variant: Optional[str] = None
    model_dtype: str = "bfloat16"  # "bfloat16"
    enable_tiling: bool = True
    enable_slicing: bool = True
    device: str = "cpu"


class BaseLatentVideoVAE(nn.Module):
    def __init__(self, args: LatentVideoVAEArgs):
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
    def __init__(self, args: LatentVideoVAEArgs):
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


def _assert_cosmos_model_type(
    model_path: str, expected_type: Literal["DV", "CV"]
) -> str:
    """
    Asserts that the provided model_path matches the expected model type.
    """
    match = re.search(r"Cosmos-Tokenizer-(DV|CV)", model_path, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"No valid model type found in model_path: {model_path}. Expected 'Cosmos-Tokenizer-DV' or 'Cosmos-Tokenizer-CV'."
        )

    model_type = match.group(1)  # Extract the matched type (DV or CV)
    if model_type.upper() != expected_type.upper():
        raise ValueError(
            f"Model type mismatch: Expected '{expected_type.upper()}' but found '{model_type.upper()}' in model_path: {model_path}."
        )
    return model_type


# * For COSMOS, create have a dummy
class COSMOSDiscreteVAE(BaseLatentVideoVAE):

    def __init__(self, args: LatentVideoVAEArgs):
        """
        Initialize the encoder and decoder for discrete VAE.
        Checks model type and returns the initialized VAE instance.
        """
        super().__init__(args)
        cfg = self.cfg
        model_type = _assert_cosmos_model_type(cfg.pretrained_model_name_or_path, "DV")
        logger.info(f"COSMOSDiscreteVAE initialized with type: {model_type}")

        # vae = CausalVideoTokenizer(
        #     checkpoint=f"{cfg.pretrained_model_name_or_path}/autoencoder.jit"
        # )
        # return vae
        # Create a "vae" object to hold the encoder and decoder
        encoder_config = discrete_video_dict
        encoder_config.update(dict(spatial_compression=16))
        encoder_config.update(dict(temporal_compression=8))
        self.vae = CausalVideoTokenizer(
            checkpoint_enc=f"{cfg.pretrained_model_name_or_path}/encoder.jit",
            checkpoint_dec=f"{cfg.pretrained_model_name_or_path}/decoder.jit",
            tokenizer_config=discrete_video_dict,
            dtype=args.model_dtype,
            device=args.device,
        )
        self.vae._enc_model.quantizer.dtype = getattr(torch, args.model_dtype)
        self.vae._dec_model.inv_quant.quantizer.dtype = getattr(torch, args.model_dtype)

    @torch.no_grad()
    def encode(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input frames into discrete indices.

        return indices with 4d shape, BTHW;
        """
        dtype = next(self.vae.parameters()).dtype
        frames_tensor = frames_tensor.to(dtype)
        if (
            frames_tensor.ndim == 4
        ):  # Check if the input tensor is 4D, BCHW, image tensor
            frames_tensor = frames_tensor.unsqueeze(
                2
            )  # Add a temporal dimension (T=1) for video vae
        indices, codes = self.vae.encode(frames_tensor)
        return indices, codes

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes the discrete indices back into reconstructed frames.
        """
        x = self.vae.decode(indices)
        if x.ndim == 5 and x.shape[2] == 1:  # Check if T=1
            x = x.squeeze(2)  # Remove the temporal dimension at index 2
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode and then decode.
        """
        indices, codes = self.encode(x)
        return self.decode(indices)


class COSMOSContinuousVAE(BaseLatentVideoVAE):
    def __init__(self, args: LatentVideoVAEArgs):
        super().__init__(args)
        """
        Initialize the encoder and decoder for Continuous VAE.
        Checks model type and returns the initialized VAE instance.
        """
        cfg = self.cfg
        model_type = _assert_cosmos_model_type(cfg.pretrained_model_name_or_path, "CV")
        logger.info(f"COSMOSContinuousVAE initialized with type: {model_type}")
        self.vae = types.SimpleNamespace()
        encoder_config = continuous_video_dict
        encoder_config.update(dict(spatial_compression=16))
        encoder_config.update(dict(temporal_compression=8))
        self.vae = CausalVideoTokenizer(
            checkpoint_enc=f"{cfg.pretrained_model_name_or_path}/encoder.jit",
            checkpoint_dec=f"{cfg.pretrained_model_name_or_path}/decoder.jit",
            tokenizer_config=continuous_video_dict,
            dtype=args.model_dtype,
            device="cpu",
        )

    @torch.no_grad()
    def encode(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input frames into latent representations.
        """
        dtype = next(self.vae.parameters()).dtype
        frames_tensor = frames_tensor.to(dtype)
        if (
            frames_tensor.ndim == 4
        ):  # Check if the input tensor is 4D, BCHW, image tensor
            frames_tensor = frames_tensor.unsqueeze(
                2
            )  # Add a temporal dimension (T=1) for video vae
        (latent,) = self.vae.encode(frames_tensor)
        return latent

    @torch.no_grad()
    def decode(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representations back into reconstructed frames.
        """
        dtype = next(self.vae.parameters()).dtype
        encoded_tensor = encoded_tensor.to(dtype)
        x = self.vae.decode(encoded_tensor)
        if x.ndim == 5 and x.shape[2] == 1:  # Check if T=1
            x = x.squeeze(2)  # Remove the temporal dimension at index 2
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode and then decode.
        """
        x = self.encode(x)
        return self.decode(x)


T = TypeVar("T", bound=BaseLatentVideoVAE)


# LatentVideoVAE class with registry and instantiation
class LatentVideoVAE(Generic[T]):
    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register_vae(cls, name: str, vae_class: Type[T]):
        cls._registry[name] = vae_class

    def __init__(self, args: LatentVideoVAEArgs, **kwargs):
        name = args.model_name
        if name not in self._registry:
            raise ValueError(
                f"VAE '{name}' is not registered. Available options: {list(self._registry.keys())}"
            )
        self.vae: T = self._registry[name](args, **kwargs)  # Instantiate the VAE class

    def __getattr__(self, attr):
        """
        Delegate attribute and method access to the actual internal VAE instance.
        """
        if hasattr(self.vae, attr):
            return getattr(self.vae, attr)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )


# Register VAE classes
LatentVideoVAE.register_vae("Hunyuan", HunyuanVideoVAE)
LatentVideoVAE.register_vae("COSMOS-DV", COSMOSDiscreteVAE)
LatentVideoVAE.register_vae("COSMOS-CV", COSMOSContinuousVAE)


"""
# Example usage
args = {"vae": "VAE1", "param1": 42, "param2": "example"}  # Example args
compressor = LatentVideoVAE(args["vae"], param1=args["param1"], param2=args["param2"])
print(compressor.forward("data"))  # Output: VAE1 processing data with 42, example

args = {"vae": "VAE2", "param3": 3.14}  # Example args for VAE2
compressor = LatentVideoVAE(args["vae"], param3=args["param3"])
print(compressor.forward("data"))  # Output: VAE2 processing data with 3.14
"""
