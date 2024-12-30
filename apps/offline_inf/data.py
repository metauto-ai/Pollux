import os
import logging
import datasets
from apps.main.utils.mongodb_data_load import MONGODB_URI
from apps.main.utils.imagenet_classes import IMAGENET2012_CLASSES
from pymongo import MongoClient
from typing import Dict, Any
from pathlib import Path
import uuid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import pickle
import torch
import time
from safetensors.torch import save_file, load_file
import numpy as np
import glob

logger = logging.getLogger()

class AverageMeter:
    def __init__(self):
        self.total_time = 0.0
        self.total_samples = 0

    def update(self, time_spent, num_samples):
        self.total_time += time_spent
        self.total_samples += num_samples

    def info(self):
        avg_time_per_sample = self.total_time / self.total_samples if self.total_samples > 0 else float('inf')
        samples_per_sec = self.total_samples / self.total_time if self.total_time > 0 else 0.0
        return f"Avg time/sample: {avg_time_per_sample:.6f}s, Samples/sec: {samples_per_sec:.2f}"

    def conclude(self, meter_name):
        logger.info(f"{meter_name}: {self.info()}")

class StorageMeter:
    def __init__(self):
        self.total_size_mb = 0.0
        self.total_samples = 0

    def update(self, size_bytes, num_samples):
        size_mb = size_bytes / (1024 * 1024)  # Convert bytes to MB
        self.total_size_mb += size_mb
        self.total_samples += num_samples

    def info(self):
        avg_size_per_sample_mb = self.total_size_mb / self.total_samples if self.total_samples > 0 else float('inf')
        return f"Total size: {self.total_size_mb:.2f} MB, Total samples: {self.total_samples}, Avg size/sample: {avg_size_per_sample_mb:.2f} MB"

    def conclude(self, meter_name):
        logger.info(f"{meter_name}: {self.info()}")
        

def save_tensor(
    tensors: torch.Tensor,
    output_dir: str,
    batch: Dict[str, Any],
    prefix: str = "latent",
    save_format: str = ".pkl",
    save_meter: AverageMeter = None,
    storage_meter: AverageMeter = None,
):
    # Validate save format
    supported_formats = [".pkl", ".pt", ".npy", ".safetensors"]
    if save_format not in supported_formats:
        raise ValueError(f"Unsupported save format: {save_format}. Supported formats: {supported_formats}")
    
    # Input should be B x ... Tensor
    output_dir = os.path.join(output_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each tensor image to PIL format and save
    tensor_paths = []
    total_size_bytes = 0
    start_time = time.time()

    for i in range(len(tensors)):
        tensor = tensors[i].cpu()
        tensor_path = os.path.join(output_dir, f"{prefix}_{str(batch['_id'][i])}{save_format}")

        if save_format == ".pkl":
            with open(tensor_path, "wb") as f:
                pickle.dump(tensor, f)
        elif save_format == ".pt":
            torch.save(tensor, tensor_path)
        elif save_format == ".npy":
            np.save(tensor_path, tensor.numpy())
        elif save_format == ".safetensors":
            save_file({"tensor": tensor}, tensor_path)

        # Get file size
        total_size_bytes += os.path.getsize(tensor_path)
        tensor_paths.append(tensor_path)
        
        tensor_paths.append(tensor_path)

    save_time = time.time() - start_time
    if save_meter:
        save_meter.update(save_time, len(tensors))
    if storage_meter:
        storage_meter.update(total_size_bytes, len(tensors))
        
    batch[f"{prefix}_tensor_path"] = tensor_paths
    logger.warning(f"Saved {len(tensors)} tensors to {output_dir}")
    return batch


def benchmark_loading(dump_dir, prefix_mapping, save_format):
    logger.info("Benchmarking tensor loading...")
    load_meters = {key: AverageMeter() for key in prefix_mapping.keys()}

    for key, prefix in prefix_mapping.items():
        tensor_dir = os.path.join(dump_dir, prefix)
        tensor_files = glob.glob(os.path.join(tensor_dir, f"*{save_format}"))
        
        for tensor_file in tensor_files:
            start_time = time.time()
            if save_format == ".pkl":
                with open(tensor_file, "rb") as f:
                    tensor = pickle.load(f)
            elif save_format == ".pt":
                tensor = torch.load(tensor_file)
            elif save_format == ".npy":
                tensor = torch.from_numpy(np.load(tensor_file))
            elif save_format == ".safetensors":
                tensor = load_file(tensor_file)["tensor"]
            tensor = tensor.cuda()  # Move to GPU
            load_time = time.time() - start_time
            load_meters[key].update(load_time, 1)
    
    for name, meter in load_meters.items():
        meter.conclude(f"Loading ({name})")

def transform_dict(input_dict):
    keys = list(input_dict.keys())
    values = list(zip(*input_dict.values()))
    return [{key: value for key, value in zip(keys, v)} for v in values]


def remove_tensors(input_dict):
    return_dict = {}
    for k, values in input_dict.items():
        if not isinstance(values[0], torch.Tensor):
            return_dict[k] = values
    return return_dict
