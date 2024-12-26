import os
import logging
import datasets
import numpy as np
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, TypedDict, Final
from pymongo import MongoClient
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from apps.main.modules.preprocessing import ImageProcessing
from apps.main.utils.hf_data_load import HFDataLoad
from apps.main.utils.dummy_data_load import DummyDataLoad
from apps.main.utils.mongodb_data_load import MongoDBDataLoad


logger = logging.getLogger()

@dataclass
class DataArgs:
    id: str = 0
    data_name: str = "ILSVRC/imagenet-1k"  # Supported values: "ILSVRC/imagenet-1k", "dummy", "NucluesIMG-100M"
    task: str = "class_to_image"
    batch_size: int = 12
    num_workers: int = 8
    image_size: int = 256
    split: str = "train"
    root_dir: Optional[str] = None  # For local/huggingface datasets
    cache_dir: Optional[str] = None  # Cache directory for datasets
    mongo_uri: Optional[str] = None  # MongoDB URI for NucluesIMG-100M
    usage: Optional[str] = None  # "class_to_image", "image_generation"
    source: Optional[str] = None  # "huggingface", "mongodb", "local"
    stage: Optional[str] = None  # "preliminary", "pretraining", "posttraining"
    use: bool = False  # Whether to use this dataset


class AutoDataLoader:
    def __init__(self, shard_id: int, num_shards: int, train_stage: str, data_config: List[DataArgs]):

        self.shard_id = shard_id
        self.num_shards = num_shards
        self.train_stage = train_stage
        self.data_config = data_config

    def create_dataloader(self) -> DataLoader:


        for dataset_config in self.data_config:
            if dataset_config.stage == self.train_stage and dataset_config.use:
                if dataset_config.source == "huggingface":
                    return self._create_imagenet_dataloader(dataset_config)
                elif dataset_config.source == "local":
                    return self._create_dummy_dataloader(dataset_config)
                elif dataset_config.source == "mongodb":
                    return self._create_mongodb_dataloader(dataset_config)

        raise ValueError(f"No dataset configured for stage {self.train_stage} with `use: True`.")

    def _create_imagenet_dataloader(self, args: DataArgs) -> DataLoader:

        data = HFDataLoad(data_name=args.data_name, cache_dir=args.cache_dir)
        train_data = data[args.split]
        data_pipeline = ImageProcessing(args)
        train_data.set_transform(data_pipeline)
        logger.warning(
            f"Read Data with Total Shard: {self.num_shards} Current Index: {self.shard_id} Split: {args.split}"
        )
        train_data = train_data.shard(num_shards=self.num_shards, index=self.shard_id)
        return DataLoader(
            train_data, batch_size=args.batch_size, num_workers=args.num_workers
        )

    def _create_dummy_dataloader(self, args: DataArgs) -> DataLoader:

        dataset = DummyDataLoad(
            num_samples=1000, 
            num_classes=10,   
            image_size=(3, args.image_size, args.image_size),
            word_count=32,    
        )
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    def _create_mongodb_dataloader(self, args: DataArgs) -> DataLoader:

        dataset = MongoDBDataLoad(
            mongo_uri=args.mongo_uri,
            collection_name=args.data_name,
            query={"aesthetic_score": {"$gt": 5.5}},
            shard_id=self.shard_id,
            num_shards=self.num_shards,
        )
        return DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
        )
