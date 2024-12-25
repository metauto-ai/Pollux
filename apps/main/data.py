import os
import logging

import datasets
import numpy as np
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, TypedDict, Final
import io
import os
import requests
from urllib.parse import quote_plus
from pymongo import MongoClient
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms

from apps.main.modules.preprocessing import (
    center_crop_arr,
    random_mask_images,
    generate_random_text,
)
from apps.main.utils.imagenet_classes import IMAGENET2012_CLASSES
from apps.main.utils.mongo_db_initial import MONGODB_URI


# Configure logging
logger = logging.getLogger()

EXISTED_DATA = {
# Put the dataset name and the repo name here if we cached for fast hit.
"ILSVRC/imagenet-1k": "imagenet-1k",
}

# Huggingface data hook
def HF_DATA_LOADER(data_name: str, cache_dir: str) -> str:

    if data_name in EXISTED_DATA.keys() and os.path.exists(os.path.join(cache_dir, data_name.split("/")[-1])):
        data = datasets.load_dataset(path=data_name, cache_dir=os.path.join(cache_dir, data_name.split("/")[-1]))
        logger.info(f"Dataset \"{data_name}\" loaded from local cache.")
    else:
        data = datasets.load_dataset(path=data_name, cache_dir=os.path.join(cache_dir, data_name.split("/")[-1]))
        logger.info(f"Dataset \"{data_name}\" downloaded from huggingface.")
        import stat
        os.chmod(os.path.join(cache_dir, data_name.split("/")[-1]), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    return data

@dataclass
class DataArgs:
    data_name: str = "ILSVRC/imagenet-1k"
    batch_size: int = 2
    num_workers: int = 8
    image_size: int = 256
    split: str = "train"
    root_dir: Optional[str] = None


# Instantiate the dataset and dataloader
def create_dummy_dataloader(
    batch_size=32,
    num_samples=1000,
    num_classes=10,
    image_size=(3, 64, 64),
    word_count=256,
):
    """
    Create a dataloader for the diffusion model.
    Args:
        batch_size (int): Batch size for the dataloader.
        num_samples (int): Number of samples in the dataset.
        num_classes (int): Number of classes in the dataset.
        image_size (tuple): Size of the generated images (C, H, W).
    Returns:
        DataLoader: A DataLoader object for the dummy dataset.
    """

    dataset = DiffusionDummyDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        word_count=word_count,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_imagenet_dataloader(
    shard_id: int,
    num_shards: int,
    args: DataArgs,
) -> DataLoader:


    data = HF_DATA_LOADER(data_name="ILSVRC/imagenet-1k", cache_dir=args.root_dir)

    train_data = data[args.split]
    data_pipeline = DataPipeline(args)
    train_data.set_transform(data_pipeline)
    logger.warning(
        f"Read Data with Total Shard: {num_shards} Current Index: {shard_id} Split: {args.split}"
    )
    train_data = train_data.shard(num_shards=num_shards, index=shard_id)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.num_workers
    )
    return train_loader


def create_dataloader(
    shard_id: int,
    num_shards: int,
    args: DataArgs,
):
    if args.data_name == "ILSVRC/imagenet-1k":
        return create_imagenet_dataloader(
            shard_id,
            num_shards,
            args,
        )
    if args.data_name == "dummy":
        return create_dummy_dataloader(
            batch_size=args.batch_size,
            image_size=(3, args.image_size, args.image_size),
            word_count=32,
        )


# Dummy Dataset Class
class DiffusionDummyDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        num_classes=10,
        image_size=(3, 64, 64),
        word_count=256,
    ):
        """
        Initialize the dummy dataset.
        Args:
            num_samples (int): Number of samples in the dataset.
            num_classes (int): Number of distinct classes.
            image_size (tuple): Shape of the dummy images (C, H, W).
            word_count: Number of the word in this caption
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.word_count = word_count

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a random sample.
        Returns:
            class_idx (int): A random class index.
            time_step (int): A random time step for the diffusion process.
            image (torch.Tensor): A random image tensor.
        """
        class_idx = np.random.randint(0, self.num_classes)
        image = torch.randn(self.image_size)  # Random image tensor
        caption = generate_random_text()
        batch = {"label": class_idx, "caption": caption, "image": image}
        return batch


class DataPipeline(nn.Module):
    def __init__(self, args: DataArgs) -> nn.Module:
        super().__init__()
        self.normalize_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        self.image_size = args.image_size
        self.cls_list = list(IMAGENET2012_CLASSES.items())

    def transform(self, x: Image) -> torch.Tensor:
        x = center_crop_arr(x, self.image_size)
        return self.normalize_transform(x)

    def forward(self, data: Dict) -> Dict:
        processed_image = []
        for image in data["image"]:
            try:
                processed_image.append(self.transform(image))
            except Exception as e:
                # logger.warning(f'Error to process the image: {e}')
                pass
        if len(data["image"]) != len(processed_image):
            dup_num = len(data["image"]) - len(processed_image)
            processed_image.extend([processed_image[-1]] * dup_num)
            for k in data.keys():
                if k != "image":
                    data[k][-dup_num:] = [data[k][-dup_num - 1]] * dup_num
        data["image"] = processed_image
        caption_list = []
        for label in data["label"]:
            _, cap = self.cls_list[label]
            caption_list.append(cap)
        data["caption"] = caption_list
        return data


class BaseMongoDBDataset(Dataset):
    """
    with BaseMongoDBDataset(
        collection_name="unsplash_images",
        query={"aesthetic_score": {"$gt": 5.5}},
        shard_idx=0,
        num_shards=8,
    ) as data_set:
        print(len(data_set.data))
        print(data_set[0])

    """

    def __init__(
        self,
        num_shards,
        shard_idx,
        collection_name: str,
        query: dict[str, Any],
    ) -> None:
        super().__init__()
        assert shard_idx >= 0 and shard_idx < num_shards, "Invalid shard index"
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client["world_model"]
        self.collection_name = collection_name
        self.query = query
        self.collection = self.db[collection_name]
        self.data = list(self.collection.find(query))
        shard_size = len(self.data) // num_shards
        remainder = len(self.data) % num_shards
        start = shard_idx * shard_size + min(shard_idx, remainder)
        end = start + shard_size + (1 if shard_idx < remainder else 0)

        self.data = self.data[start:end]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
