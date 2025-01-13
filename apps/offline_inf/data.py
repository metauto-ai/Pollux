import os
import logging
import datasets
from apps.main.utils.mongodb_data_load import MONGODB_URI
from apps.main.utils.imagenet_classes import IMAGENET2012_CLASSES
from pymongo import MongoClient
from typing import Dict, Any, List, Optional
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
import multiprocessing
import torch.utils.benchmark as benchmark
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from io import BytesIO
import boto3

config = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    max_pool_connections=200,  # Increase pool size (default is 10)
)
s3 = boto3.client(
    "s3",
    aws_access_key_id="AKIA47CRZU7STC4XUXER",
    aws_secret_access_key="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",
    region_name="us-east-1",
    config=config,
)

logger = logging.getLogger()


class AverageMeter:
    def __init__(self):
        self.total_time = 0.0
        self.total_samples = 0

    def update(self, time_spent, num_samples):
        self.total_time += time_spent
        self.total_samples += num_samples

    def info(self):
        avg_time_per_sample = (
            self.total_time / self.total_samples
            if self.total_samples > 0
            else float("inf")
        )
        samples_per_sec = (
            self.total_samples / self.total_time if self.total_time > 0 else 0.0
        )
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
        avg_size_per_sample_mb = (
            self.total_size_mb / self.total_samples
            if self.total_samples > 0
            else float("inf")
        )
        return f"Total size: {self.total_size_mb:.2f} MB, Total samples: {self.total_samples}, Avg size/sample: {avg_size_per_sample_mb:.2f} MB"

    def conclude(self, meter_name):
        logger.info(f"{meter_name}: {self.info()}")


def save_unit_func(
    idx,
    tensors,
    tensor_paths,
    total_size_bytes,
    output_dir,
    prefix,
    id_list,
    save_format,
):
    tensor = tensors[idx]
    tensor_path = os.path.join(output_dir, f"{prefix}_{id_list[idx]}{save_format}")
    if save_format == ".pkl":
        with open(tensor_path, "wb") as f:
            pickle.dump(tensor, f)
    elif save_format == ".pt":
        torch.save(tensor, tensor_path)
    elif save_format == ".npy":
        np.save(tensor_path, tensor.numpy())
    elif save_format == ".safetensors":
        save_file({"tensor": tensor}, tensor_path)
    tensor_paths[idx] = tensor_path
    total_size_bytes[idx] = os.path.getsize(tensor_path)


def save_tensor(
    tensors: torch.Tensor,
    output_dir: str,
    batch: Dict[str, Any],
    prefix: str = "latent",
    save_format: str = ".pkl",
    save_meter: AverageMeter = None,
    storage_meter: AverageMeter = None,
    num_workers: int = 8,
):
    tensors = tensors.cpu()
    # Validate save format
    supported_formats = [".pkl", ".pt", ".npy", ".safetensors"]
    if save_format not in supported_formats:
        raise ValueError(
            f"Unsupported save format: {save_format}. Supported formats: {supported_formats}"
        )

    # Input should be B x ... Tensor
    output_dir = os.path.join(output_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)

    # Convert each tensor image to PIL format and save
    start_time = time.time()

    manager = multiprocessing.Manager()
    tensor_paths = manager.list([None] * len(tensors))
    total_size_bytes = manager.list([0] * len(tensors))
    id_list = batch["_id"]
    args = [
        (
            idx,
            tensors,
            tensor_paths,
            total_size_bytes,
            output_dir,
            prefix,
            id_list,
            save_format,
        )
        for idx in range(len(tensors))
    ]
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(save_unit_func, args)
    total_size_bytes = np.sum(total_size_bytes)

    save_time = time.time() - start_time
    if save_meter:
        save_meter.update(save_time, len(tensors))
    if storage_meter:
        storage_meter.update(total_size_bytes, len(tensors))

    batch.update({f"{prefix}_tensor_path": tensor_paths})
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


def benchmark_url_loading(data_loader):
    start_time = time.perf_counter()  # Use time.perf_counter() for precise timing
    iterator = iter(data_loader)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    logger.info(f"Time taken to create dataloader iterator: {elapsed_time:.4f} seconds")

    timer = benchmark.Timer(
        stmt="batch = next(iterator)", globals={"iterator": iterator}
    )
    result = timer.timeit(20)  # Number of batches
    logger.info("Average batch fetching time: %.6f seconds", result.mean)


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


def save_parquet(
    data: Dict[str, Any],
    output_dir: str,
    prefix: str = "data",
    save_meter: AverageMeter = None,
    storage_meter: StorageMeter = None,
):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # Convert the dictionary to a PyArrow Table
    df = pd.DataFrame(data)

    # Define the output file path
    parquet_path = os.path.join(output_dir, f"{prefix}.parquet")

    # Save the table as a Parquet file
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    save_time = time.time() - start_time
    if save_meter:
        save_meter.update(save_time, 1)
    total_size_bytes = os.path.getsize(parquet_path)
    if storage_meter:
        storage_meter.update(total_size_bytes, 1)
    logger.warning(f"Saved data to {parquet_path}")
    return parquet_path


def upload_parquet(
    data: Dict[str, Any],
    output_dir: str,
    prefix: str = "data",
    bucket_name: str = "haozhe",
    save_meter: AverageMeter = None,
    storage_meter: StorageMeter = None,
):
    start_time = time.time()

    # Convert the dictionary to a PyArrow Table
    df = pd.DataFrame(data)

    # Define the output file path
    parquet_path = f"{output_dir}/{prefix}.parquet"
    buffer = BytesIO()
    df.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
    buffer.seek(0)
    s3.upload_fileobj(buffer, bucket_name, parquet_path)
    save_time = time.time() - start_time
    if save_meter:
        save_meter.update(save_time, 1)
    total_size_bytes = os.path.getsize(parquet_path)
    if storage_meter:
        storage_meter.update(total_size_bytes, 1)
    logger.warning(f"Saved data to {parquet_path}")
    return parquet_path
