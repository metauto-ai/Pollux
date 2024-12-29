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

logger = logging.getLogger()


def save_tensor(
    tensors: torch.Tensor,
    output_dir: str,
    batch: Dict[str, Any],
    prefix: str = "latent",
):
    # Input should be B x ... Tensor
    os.makedirs(output_dir, exist_ok=True)

    # Convert each tensor image to PIL format and save
    tensor_paths = []
    for i in range(len(tensors)):
        tensors[i].cpu()
        tensor_path = os.path.join(output_dir, f"{prefix}_{str(batch['_id'][i])}.pkl")
        with open(tensor_path, "wb") as f:
            pickle.dump(tensors, f)
        tensor_paths.append(tensor_path)
    batch[f"{prefix}_tensor_path"] = tensor_paths
    logger.warning(f"Saved {len(tensors)} tensor to {output_dir}")
    return batch


def transform_dict(input_dict):
    keys = list(input_dict.keys())
    values = list(zip(*input_dict.values()))
    return [{key: value for key, value in zip(keys, v)} for v in values]


def remove_tensors(input_dict):
    return {
        k: [v for v in values if not isinstance(v, torch.Tensor)]
        for k, values in input_dict.items()
    }
