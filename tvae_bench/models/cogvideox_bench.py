import os
# Set Hugging Face home directory
os.environ["HF_HOME"] = "/jfs/jinjie/huggingface"


import time
from datetime import datetime, timedelta
from tqdm import tqdm
import torch
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms

from typing import Dict,Type
import numpy as np

# Configuration Class
class Config:

    def __init__(self,
                 model_class: Type,
                 model_path: str,
                 device: str = "cuda",
                 dtype: str = "float16",
                 batch_size: int = 1,
                 custom_batch_size: Dict[str, int] = None,
                 source_base: str = "../resources/videos/",
                 output_base: str = "./output_videos/"):
        self.model_class = model_class
        self.model_path = model_path
        self.device = device
        self.dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.batch_size = batch_size
        self.custom_batch_size = custom_batch_size or {}
        self.source_base = source_base
        self.output_base = output_base


# GeneralAutoEncoderKL Class
class GeneralAutoEncoderKL:

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype

        os.makedirs(self.config.output_base, exist_ok=True)

        self.model = self.config.model_class.from_pretrained(
            self.config.model_path, torch_dtype=self.dtype).to(self.device)

        print(f"Model loaded successfully from {self.config.model_path}.")

        self.model.enable_slicing()
        self.model.enable_tiling()
        # self.model.disable_slicing()
        # self.model.disable_tiling()

        self.transform = transforms.ToTensor()

    def encode(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded_frames = self.model.encode(frames_tensor)[0].sample()
        return encoded_frames

    def decode(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            decoded_frames = self.model.decode(encoded_tensor).sample
        return decoded_frames


def format_timedelta(td):
    """Formats a timedelta object into HH:MM:SS string."""
    seconds = int(td.total_seconds())
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Processing Script
if __name__ == "__main__":

    data_root = "/jfs/jinjie"

    config = Config(model_class=AutoencoderKLCogVideoX,
                    # model_path="THUDM/CogVideoX-2b",
                    model_path=
                    f"{data_root}/huggingface/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/vae",
                    device="cuda:0",
                    dtype="bfloat16",
                    batch_size=10,
                    custom_batch_size={
                        'imagenet_val': 1,
                        'textocr': 1,
                        'bridgedata_v2': 1,
                        'panda_70m': 1,
                        'real10k': 1
                    },  # * variable resolution
                    source_base=f"{data_root}/data/vae_eval_bench/processed_gt_v3",
                    output_base=f"{data_root}/data/vae_eval_bench/model_recon/cogvideox")

    autoencoder = GeneralAutoEncoderKL(config)

    B, C, T, H, W = 12, 3, 8*3+1, 360, 640  # Example shape
    # B, C, T, H, W = 12, 3, 8*3+1, 
    num_runs = 30

    encoding_times = []
    decoding_times = []
    total_times = []
    peak_memory = []

    for _ in tqdm(range(num_runs)):
        # Generate random input tensor
        input_tensor = torch.randn(B, C, T, H, W).to(
            autoencoder.device, dtype=autoencoder.dtype)

        # Encoding
        start_time = time.time()
        encoded = autoencoder.encode(input_tensor)
        encoding_time = time.time() - start_time
        encoding_times.append(encoding_time)

        # Decoding
        start_time = time.time()
        decoded = autoencoder.decode(encoded)
        decoding_time = time.time() - start_time
        decoding_times.append(decoding_time)

        total_time = encoding_time + decoding_time
        total_times.append(total_time)

        # Memory usage
        peak_memory.append(torch.cuda.max_memory_allocated())
        
        # Latent tensor shape
        latent_shape = encoded.shape

        del input_tensor, encoded, decoded
        torch.cuda.empty_cache()

    # Calculate average times
    avg_encoding_time = np.mean(encoding_times)
    avg_decoding_time = np.mean(decoding_times)
    avg_total_time = np.mean(total_times)

    # Calculate per-image times
    NUM_FRAME=B*T
    
    per_image_encoding_time = avg_encoding_time / NUM_FRAME
    per_image_decoding_time = avg_decoding_time / NUM_FRAME
    per_image_total_time = avg_total_time / NUM_FRAME

    # Peak memory usage
    peak_memory_usage = np.max(peak_memory)

    

    print(f"Average encoding time per image: {per_image_encoding_time:.4f} seconds")
    print(f"Average decoding time per image: {per_image_decoding_time:.4f} seconds")
    print(
        f"Average total reconstruction time per image: {per_image_total_time:.4f} seconds"
    )
    
    
    print(f"Average encoding frame per seconde: {1/per_image_encoding_time:.4f} frames")
    print(f"Average decoding frame per second: {1/per_image_decoding_time:.4f} frames")
    print(
        f"Average frames per second: {(1/per_image_total_time):.4f} frames"
    )
    
    print(f"Peak GPU memory usage: {peak_memory_usage / (1024 ** 3):.2f} GB")
    print(f"Latent tensor shape: {latent_shape}")
    
    # 73819 MB; 7493 MB;