import os
import torch
import imageio
import numpy as np
from torchvision import transforms
from typing import Tuple, List, Dict, Type
import logging
from tqdm import tqdm

logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

# Configuration Class
class Config:
    def __init__(
        self,
        model_class: Type,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 1,
        custom_batch_size: Dict[str, int] = None,
        source_base: str = "../resources/videos/",
        output_base: str = "./output_videos/"
    ):
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
            self.config.model_path,
            torch_dtype=self.dtype
        ).to(self.device)

        print(f"Model loaded successfully from {self.config.model_path}.")

        self.model.enable_slicing()
        self.model.enable_tiling()
        # self.model.disable_slicing()
        # self.model.disable_tiling()
        self.transform = transforms.ToTensor()

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
        with torch.no_grad():
            encoded_frames = self.model.encode(frames_tensor)[0].sample()
        return encoded_frames

    def decode(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            decoded_frames = self.model.decode(encoded_tensor).sample
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


# * Testing code on H100 server
import os
import time
from datetime import datetime, timedelta

# Set Hugging Face home directory
os.environ["HF_HOME"] = "/jfs/jinjie/huggingface"
def format_timedelta(td):
    """Formats a timedelta object into HH:MM:SS string."""
    seconds = int(td.total_seconds())
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

from diffusers import AutoencoderKLHunyuanVideo


# Processing Script
if __name__ == "__main__":
    
    data_root="/jfs/jinjie"
    
    config = Config(
        model_class=AutoencoderKLHunyuanVideo,
        # model_path="THUDM/CogVideoX-2b",
        # model_path="/home/maij/.cache/huggingface/hub/models--THUDM--CogVideoX-5b/snapshots/8d6ea3f817438460b25595a120f109b88d5fdfad/vae",
        model_path=f"{data_root}/huggingface/hub/models--tencent--HunyuanVideo/snapshots/2a15b5574ee77888e51ae6f593b2ceed8ce813e5/vae",
        device="cuda:2",
        dtype="bfloat16", 
        batch_size=10,
        custom_batch_size={'imagenet_val': 1, 'textocr': 1, 'bridgedata_v2':1, 'panda_70m':1, 'real10k':1}, # * variable resolution
        source_base=f"{data_root}/data/vae_eval_bench/processed_gt_v3", 
        output_base=f"{data_root}/data/vae_eval_bench/model_recon/hunyuan",
        # source_base="/mnt/data/jinjie/data/vae_eval_bench/processed_gt_v3",
        # output_base="/mnt/data/jinjie/data/vae_eval_bench/model_recon/cogvideox"
    )
    
    autoencoder = GeneralAutoEncoderKL(config)

    B, C, T, H, W = 12, 3, 8*3+1, 360, 640  # Example shape
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
