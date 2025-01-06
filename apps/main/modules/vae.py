from dataclasses import dataclass, field
from typing import Optional, Literal, Type, Dict
import logging
import torch
from torch import nn
from diffusers import AutoencoderKLHunyuanVideo

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class LatentVideoVAEArgs:
    model_name: Literal["Hunyuan", "COSMOS"] = "Hunyuan"  # Default value is "Hunyuan"
    pretrained_model_name_or_path: str = "tencent/HunyuanVideo"
    revision: Optional[str] = None
    variant: Optional[str] = None
    model_dtype: str = "bf16"
    enable_tiling: bool = True
    enable_slicing: bool = True




class BaseLatentVideoVAE(nn.Module):
    def __init__(self, args: LatentVideoVAEArgs):
        super().__init__()
        self.cfg=args
        
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        logger.warning(f"Useless func call, {self.cfg.model_name} TVAE model doesn't support slicing !")
        pass

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        logger.warning(f"Useless func call, {self.cfg.model_name} TVAE model doesn't support slicing !")
        pass

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        logger.warning(f"Useless func call, {self.cfg.model_name} TVAE model doesn't support tiling !")
        pass

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        logger.warning(f"Useless func call, {self.cfg.model_name} TVAE model doesn't support tiling !")
        pass




class HunyuanVideoVAE(nn.Module):
    def __init__(self, args: LatentVideoVAEArgs):
        super().__init__()
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
        )
        self.vae = vae
        self.vae = self.vae.requires_grad_(False)

        if args.enable_slicing:
            self.vae.enable_slicing()
        else:
            self.vae.disable_slicing()

        if args.enable_tiling:
            self.vae.enable_tiling()
        else:
            self.vae.disable_tiling()

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

class COSMOSVideoVAE:
    def __init__(self):
        pass


# LatentVideoVAE class with registry and instantiation
class LatentVideoVAE:
    _registry: Dict[str, Type[BaseLatentVideoVAE]] = {}

    @classmethod
    def register_vae(cls, name: str, vae_class: Type[BaseLatentVideoVAE]):
        cls._registry[name] = vae_class

    def __init__(self, args: LatentVideoVAEArgs, **kwargs):
        name=args.model_name
        if name not in self._registry:
            raise ValueError(f"VAE '{name}' is not registered. Available options: {list(self._registry.keys())}")
        self.vae = self._registry[name](**kwargs)  # Instantiate the selected VAE class

    def __getattr__(self, attr):
        """
        Delegate attribute and method access to the actual internal VAE instance.
        """
        if hasattr(self._vae, attr):
            return getattr(self._vae, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")



# Register VAE classes
LatentVideoVAE.register_vae("Hunyuan",  HunyuanVideoVAE)
LatentVideoVAE.register_vae("COSMOS", COSMOSVideoVAE)

# Example usage
args = {"vae": "VAE1", "param1": 42, "param2": "example"}  # Example args
compressor = LatentVideoVAE(args["vae"], param1=args["param1"], param2=args["param2"])
print(compressor.forward("data"))  # Output: VAE1 processing data with 42, example

args = {"vae": "VAE2", "param3": 3.14}  # Example args for VAE2
compressor = LatentVideoVAE(args["vae"], param3=args["param3"])
print(compressor.forward("data"))  # Output: VAE2 processing data with 3.14