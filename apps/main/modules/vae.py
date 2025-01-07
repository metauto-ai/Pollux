from dataclasses import dataclass, field
from typing import Optional, Literal, Type, Dict, List, Tuple
import logging
import torch
from torch import nn
from diffusers import AutoencoderKLHunyuanVideo
import os
import re
import imageio
import numpy as np
from torchvision import transforms
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

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


# CosmosVAE Class
class COSMOSVideoVAE:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype

        os.makedirs(self.config.output_base, exist_ok=True)

        # Determine if the model is discrete (DV) or continuous (CV)
        self.is_discrete = self._check_model_type(config.model_path)

        
        self.encoder = CausalVideoTokenizer(
            checkpoint_enc=f'{self.config.model_path}/encoder.jit'
        ).to(self.device)
        self.decoder = CausalVideoTokenizer(
            checkpoint_dec=f'{self.config.model_path}/decoder.jit'
        ).to(self.device)

        model_type = "Discrete (DV)" if self.is_discrete else "Continuous (CV)"
        print(f"CosmosVAE loaded successfully from {self.config.model_path}. Model type: {model_type}")

        self.transform = transforms.ToTensor()

    @staticmethod
    def _check_model_type(model_path: str) -> bool:
        """
        Determine the model type based on the model_path using regex.
        Returns True if the model is discrete (DV), otherwise False (continuous CV).
        """
        if re.search(r"Cosmos-Tokenizer-DV", model_path, re.IGNORECASE):
            return True
        elif re.search(r"Cosmos-Tokenizer-CV", model_path, re.IGNORECASE):
            return False
        else:
            raise ValueError(f"Unable to determine model type from model_path: {model_path}")

    def preprocess_videos(self, video_paths: List[str]) -> Tuple[torch.Tensor, List[float], List[int], List[Tuple[int, int]]]:
        batch_frames = []
        fps_list, num_frames_list, resolutions = [], [], []

        for video_path in video_paths:
            video_reader = imageio.get_reader(video_path, "ffmpeg")
            meta_data = video_reader.get_meta_data()
            fps = meta_data.get('fps', 30)

            frames = [self.transform(frame) for frame in video_reader]
            video_reader.close()

            if not frames:
                raise ValueError(f"No frames found in video: {video_path}")

            num_frames = len(frames)
            resolution = frames[0].shape[1], frames[0].shape[2]

            fps_list.append(fps)
            num_frames_list.append(num_frames)
            resolutions.append(resolution)

            frames_tensor = torch.stack(frames).to(self.device).permute(1, 0, 2, 3)
            batch_frames.append(frames_tensor)

        batch_tensor = torch.stack(batch_frames).to(self.dtype)
        return batch_tensor, fps_list, num_frames_list, resolutions

    def encode(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input frames using the appropriate encoding mechanism.
        Returns latent or indices depending on the model type.
        """
        if self.is_discrete:
            indices, codes = self.encoder.encode(frames_tensor)
            return indices
        else:
            latent, = self.encoder.encode(frames_tensor)
            return latent

    def decode(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent or indices back into reconstructed frames.
        """
        if self.is_discrete:
            decoded_frames = self.decoder.decode(encoded_tensor)
        else:
            decoded_frames = self.decoder.decode(encoded_tensor)
        return decoded_frames

    def save_videos(self, tensor: torch.Tensor, output_paths: List[str], fps_list: List[float],
                    num_frames_list: List[int], resolutions: List[Tuple[int, int]]):
        tensor = tensor.to(dtype=torch.float32)
        for i, (fps, num_frames, resolution, output_path) in enumerate(zip(fps_list, num_frames_list, resolutions, output_paths)):
            frames = tensor[i].permute(1, 2, 3, 0).cpu().numpy()
            frames = np.clip(frames, 0, 1) * 255
            frames = frames.astype(np.uint8)

            num_output_frames = frames.shape[0]
            assert num_output_frames == num_frames, (
                f"Frame count mismatch: input {num_frames} vs output {num_output_frames}")

            output_resolution = frames.shape[1], frames.shape[2]
            assert output_resolution == resolution, (
                f"Resolution mismatch: input {resolution} vs output {output_resolution}")

            writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            print(f"Saved video to {output_path} with {num_output_frames} frames at {output_resolution} resolution and {fps} fps.")

    def reconstruct_videos(self, video_paths: List[str], output_paths: List[str]):
        batch_size = self.config.batch_size
        for dataset_name, custom_size in self.config.custom_batch_size.items():
            if any(dataset_name in path for path in video_paths):
                batch_size = custom_size
                break

        for i in range(0, len(video_paths), batch_size):
            batch_video_paths = video_paths[i:i + batch_size]
            batch_output_paths = output_paths[i:i + batch_size]

            frames_tensor, fps_list, num_frames_list, resolutions = self.preprocess_videos(batch_video_paths)
            encoded = self.encode(frames_tensor)
            decoded = self.decode(encoded)
            self.save_videos(decoded, batch_output_paths, fps_list, num_frames_list, resolutions)
            del frames_tensor, encoded, decoded
            torch.cuda.empty_cache()


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

"""
# Example usage
args = {"vae": "VAE1", "param1": 42, "param2": "example"}  # Example args
compressor = LatentVideoVAE(args["vae"], param1=args["param1"], param2=args["param2"])
print(compressor.forward("data"))  # Output: VAE1 processing data with 42, example

args = {"vae": "VAE2", "param3": 3.14}  # Example args for VAE2
compressor = LatentVideoVAE(args["vae"], param3=args["param3"])
print(compressor.forward("data"))  # Output: VAE2 processing data with 3.14
"""