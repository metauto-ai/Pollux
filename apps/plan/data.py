import os
import logging
import random

import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, TypedDict, Final, Tuple

import datasets
import numpy as np
from PIL import Image
from pymongo import MongoClient
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from apps.main.modules.preprocess import ImageProcessing
from apps.main.utils.hf_data_load import HFDataLoad
from apps.main.utils.dummy_data_load import DummyDataLoad
from apps.main.utils.mongodb_data_load import (
    MongoDBDataLoad,
    MongoDBImageNetDataLoad,
    MongoDBCC12MDataLoad,
    MongoDBParquetDataLoad,
)
from apps.main.utils.sampler import StatefulDistributedSampler

logger = logging.getLogger()

"""
Deterministic distributed dataloader

    we disbale different random augmentations being applied 
    by each worker by comment out workder_id now
    
    if each dataloader worker should apply different transform randomly,
    set  seed + workder_id
"""


def worker_init(workder_id, seed):
    torch.manual_seed(seed)  # + worker_id)
    np.random.seed(seed)  # + worker_id)
    random.seed(seed)  # + worker_id)


@dataclass
class DataArgs:
    # * TODO: Jinjie: we should have seperate configs for data args and dataloader args
    id: str = 0
    data_name: str = (
        "ILSVRC/imagenet-1k"  # Supported values: "ILSVRC/imagenet-1k", "dummy", "NucluesIMG-100M"
    )
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
    query: Optional[Dict[str, Any]] = field(
        default_factory=dict
    )  # MongoDB query  # MongoDB query
    retries: Optional[int] = 3  # Number of retries for MongoDB connection
    use: bool = False  # Whether to use this dataset
    partition_key: str = "key"  # MongoDB partition key
    prefetch_factor: int = 2  # Prefetch factor for dataloader


class AutoDataLoader:
    def __init__(
        self,
        shard_id: int,
        num_shards: int,
        train_stage: str,
        data_config: List[DataArgs],
        # * following args should only be used by dataloader and sampler
        shuffle: Optional[bool] = False,
        pin_memory: Optional[bool] = True,
        drop_last: Optional[bool] = True,
        seed: Optional[int] = 1024,
    ):

        self.shard_id = shard_id
        self.num_shards = num_shards
        self.train_stage = train_stage
        self.data_config = data_config
        # * used by dataloader and sampler
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed = seed

    def create_dataloader(self) -> DataLoader:
        for dataset_config in self.data_config:
            logger.info(
                f"Initializing dataloader for dataset: {dataset_config.data_name}"
            )
            try:
                if dataset_config.source == "huggingface":
                    return self._create_imagenet_dataloader(dataset_config)
                elif dataset_config.source == "local":
                    return self._create_dummy_dataloader(dataset_config)
                elif dataset_config.source == "mongodb":
                    return self._create_mongodb_dataloader(dataset_config)
                else:
                    raise ValueError(
                        f"Unsupported data source: {dataset_config.source}"
                    )
            except Exception as e:
                # NOTE: we need to through the error to the upper level if the dataloader is failed to load
                logger.error(f"Error initializing dataloader: {str(e)}")
                raise

        raise ValueError(
            f"No dataset configured for stage {self.train_stage} with `use: True`."
        )

    def _create_imagenet_dataloader(
        self, args: DataArgs
    ) -> Tuple[DataLoader, StatefulDistributedSampler]:

        data = HFDataLoad(data_name=args.data_name, cache_dir=args.root_dir)
        train_data = data[args.split]
        data_pipeline = ImageProcessing(args)
        train_data.set_transform(data_pipeline)
        logger.warning(
            f"Read Data with Total Shard: {self.num_shards} Current Index: {self.shard_id} Split: {args.split}"
        )
        train_data = train_data.shard(num_shards=self.num_shards, index=self.shard_id)

        self.dataset = train_data
        return self._warp_dataloader_with_stateful_sampler(args, train_data)

    def _create_dummy_dataloader(
        self, args: DataArgs
    ) -> Tuple[DataLoader, StatefulDistributedSampler]:

        dataset = DummyDataLoad(
            num_samples=1000,
            num_classes=10,
            image_size=(3, args.image_size, args.image_size),
            word_count=32,
        )

        self.dataset = dataset
        return self._warp_dataloader_with_stateful_sampler(args, dataset)

    def _create_mongodb_dataloader(
        self, args: DataArgs
    ) -> Tuple[DataLoader, StatefulDistributedSampler]:
        if args.data_name == "imagenet-1k":
            dataset = MongoDBImageNetDataLoad(
                num_shards=self.num_shards,
                shard_idx=self.shard_id,
                collection_name=args.data_name,
                temporal_cache_name=args.data_name,
                partition_key=args.partition_key,
                args=args,
            )
        elif args.data_name == "cc12m":
            dataset = MongoDBCC12MDataLoad(
                collection_name=args.data_name,
                query=args.query,
                shard_idx=self.shard_id,
                num_shards=self.num_shards,
                temporal_cache_name=args.data_name,
                extract_field={
                    "s3url": "image",
                },
                partition_key=args.partition_key,
                args=args,
            )
        elif args.data_name == "cc12m_llama3bf128_hunyuanr256_test1m":
            dataset = MongoDBParquetDataLoad(
                collection_name=args.data_name,
                query=args.query,
                shard_idx=self.shard_id,
                num_shards=self.num_shards,
                temporal_cache_name=args.data_name,
                extract_field={
                    "parquet_size": "sample_num",
                    "parquet_path": "path",
                },
                mapping_field={
                    "HunyuanVideo_latent_code": "latent_code",
                    "LLAMA3_3B_text_embedding": "text_embedding",
                },
                partition_key=args.partition_key,
            )
        else:
            dataset = MongoDBDataLoad(
                collection_name=args.data_name,
                query=args.query,
                shard_idx=self.shard_id,
                num_shards=self.num_shards,
                temporal_cache_name=args.data_name,
                partition_key=args.partition_key,
            )

        dataset.set_local_partition()
        if hasattr(dataset, "set_mapping"):
            dataset.set_mapping()
        self.dataset = dataset
        return self._warp_dataloader_with_stateful_sampler(args, dataset)

    def _warp_dataloader_with_stateful_sampler(
        self, args: DataArgs, dataset: Dataset
    ) -> Tuple[DataLoader, StatefulDistributedSampler]:
        sampler = StatefulDistributedSampler(
            dataset,
            batch_size=args.batch_size,
            num_replicas=self.num_shards,
            rank=self.shard_id,
            shuffle=self.shuffle,
        )
        return (
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                worker_init_fn=partial(worker_init, seed=self.seed),
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                num_workers=args.num_workers,
                persistent_workers=True if args.num_workers > 0 else False,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            ),
            sampler,
        )