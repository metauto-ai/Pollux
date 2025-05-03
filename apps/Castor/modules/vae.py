import logging
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Literal, Optional, Tuple, Type, TypeVar, Union

import torch
from diffusers import AutoencoderKL, AutoencoderKLHunyuanVideo
from torch import nn
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
    dtype: str = "bfloat16"


class BaseLatentVideoVAE:
    def __init__(self, args: VideoVAEArgs):
        self.cfg = args
        self.dtype = dict(fp32=torch.float32, bfloat16=torch.bfloat16)[args.dtype]
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def enable_vae_slicing(self):
        logger.warning(
            f"Useless func call, {self.cfg.vae_type} TVAE model doesn't support slicing !"
        )
        pass

    def disable_vae_slicing(self):
        logger.warning(
            f"Useless func call, {self.cfg.vae_type} TVAE model doesn't support slicing !"
        )
        pass

    def enable_vae_tiling(self):
        logger.warning(
            f"Useless func call, {self.cfg.vae_type} TVAE model doesn't support tiling !"
        )
        pass

    def disable_vae_tiling(self):
        logger.warning(
            f"Useless func call, {self.cfg.vae_type} TVAE model doesn't support tiling !"
        )
        pass

    def extract_latents(self, batch: dict[str:any]) -> Union[torch.Tensor, List[torch.Tensor]]:
        images = batch["image"]
        if isinstance(images, torch.Tensor):
            # Handle the case where input is already a batched tensor
            return self.encode(images)
        elif isinstance(images, list):
            # Group images by resolution (H, W) while preserving original index
            grouped_images: Dict[Tuple[int, int], List[Tuple[int, torch.Tensor]]] = {}
            for i, img in enumerate(images):
                # Assuming BCHW or CHW format, get H and W
                resolution = (img.shape[-2], img.shape[-1])
                if resolution not in grouped_images:
                    grouped_images[resolution] = []
                grouped_images[resolution].append((i, img))

            # Encode batches for each resolution group
            results = [None] * len(images) # Pre-allocate results list to maintain order
            for resolution, indexed_tensors in grouped_images.items():
                indices = [item[0] for item in indexed_tensors]
                tensors = [item[1] for item in indexed_tensors]

                # Stack tensors into a batch
                input_batch = torch.stack(tensors, dim=0)

                # Encode the batch
                latent_batch = self.encode(input_batch)

                # Place individual latents back into the results list at their original indices
                for i, latent in enumerate(latent_batch):
                    original_index = indices[i]
                    results[original_index] = latent

            return results
        else:
            raise TypeError(f"Unsupported type for batch['image']: {type(images)}")


class HunyuanVideoVAE(BaseLatentVideoVAE):
    def __init__(self, args: VideoVAEArgs):
        super().__init__(args)
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            revision=self.cfg.revision,
            variant=self.cfg.variant,
            torch_dtype=self.dtype,
            device_map="auto"
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
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=self.dtype,
        ).requires_grad_(False)
        if torch.cuda.is_available():
            self.vae = self.vae.to(torch.cuda.current_device())

        self.scale = 0.3611
        self.shift = 0.1159

    @torch.compile
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.vae.device, dtype=self.vae.dtype)
        x = (
            self.vae.encode(x).latent_dist.mode() - self.shift
        ) * self.scale
        return x

    @torch.compile
    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.vae.device, dtype=self.vae.dtype)
        # Reverse the scaling and shifting from encode method
        x = (x / self.scale + self.shift) # .to(self.vae.dtype) # dtype conversion happens inside decode
        # Use the VAE's decode method and get the sample
        decoded = self.vae.decode(x).sample
        return decoded

    @torch.compile
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
            device="cuda",
            dtype=args.dtype
        ).requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input frames into latent representations.
        """
        (latent,) = self.vae.encode(x.to(self.vae._dtype))
        return latent

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representations back into reconstructed frames.
        """
        x = self.vae.decode(x.to(self.vae._dtype))
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
