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
    MongoDBImageDataLoad,
    MongoDBParquetDataLoad,
    MongoDBCaptionDataLoad,
)
from apps.main.utils.sampler import StatefulDistributedSampler
from apps.main.utils.mongodb_data_load import DictTensorBatchIterator

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
class DataLoaderArgs:
    prefetch_factor: int = 2  # Prefetch factor for dataloader
    batch_size: int = 12
    num_workers: int = 8
    seed: int = 1024
    shuffle: bool = False
    pin_memory: Optional[bool] = True
    drop_last: Optional[bool] = True


@dataclass
class DataArgs:
    id: str = 0
    data_name: str = (
        "ILSVRC/imagenet-1k"  # the same as the dataset name in Huggingface or the collection name in MongoDB
    )
    task: str = "class_to_image"
    split: str = "train"
    source: Optional[str] = None  # "huggingface", "mongodb"
    stage: Optional[str] = (
        None  # "preliminary", "pretraining", "posttraining" and so on
    )
    use: bool = False  # Whether to use this dataset

    # * Image specific args
    image_size: int = 256
    condition_image_size: int = 256

    # * Huggingface specific args
    root_dir: Optional[str] = None  # For local/huggingface datasets

    # * MongoDB specific args
    query: Optional[Dict[str, Any]] = field(default_factory=dict)  # MongoDB query
    retries: Optional[int] = 3  # Number of retries for MongoDB connection
    partition_key: str = "key"  # MongoDB partition key
    extract_field: Optional[Dict[str, str]] = field(
        default_factory=dict
    )  # which media url field is used to extract from MongoDB it is a dataset specific field
    mapping_field: Optional[Dict[str, str]] = field(
        default_factory=dict
    )  # which parquet column name field is used in parquet data. it is a dataset specific field

    # * DataLoader specific args
    dataloader: DataLoaderArgs = field(default_factory=DataLoaderArgs)


class AutoDataLoader:
    def __init__(
        self,
        shard_id: int,
        num_shards: int,
        train_stage: str,
        data_config: List[DataArgs],
        # * following args should only be used by dataloader and sampler
    ):

        self.shard_id = shard_id
        self.num_shards = num_shards
        self.train_stage = train_stage
        self.data_config = data_config

    def create_dataloader(self) -> DataLoader:
        for dataset_config in self.data_config:
            logger.info(
                f"Initializing dataloader for dataset: {dataset_config.data_name}"
            )
            try:
                if dataset_config.source == "mongodb":
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

    def _create_mongodb_dataloader(
        self, args: DataArgs
    ) -> Tuple[DataLoader, StatefulDistributedSampler]:
        record_batch_size = args.dataloader.batch_size
        if args.data_name in ["cc12m", "bucket-256-1", "bucket-hq", "bucket-256-2"]:
            dataset = MongoDBImageDataLoad(
                collection_name=args.data_name,
                query=args.query,
                shard_idx=self.shard_id,
                num_shards=self.num_shards,
                extract_field=args.extract_field,  # {"s3url": "image",}
                partition_key=args.partition_key,
                args=args,
            )
        elif args.data_name in [
            "cc12m_l3bf128_hr256",
            "cc12m_aethetics_6_5_llama3bf128_hunyuanr256_s3",
            "bucket-256-parquet",
        ]:
            dataset = MongoDBParquetDataLoad(
                collection_name=args.data_name,
                query=args.query,
                shard_idx=self.shard_id,
                num_shards=self.num_shards,
                extract_field=args.extract_field,  # {"parquet_size": "sample_num","parquet_path": "path",}
                mapping_field=args.mapping_field,  # {"HunyuanVideo_latent_code": "latent_code","LLAMA3_3B_text_embedding": "text_embedding",},
                partition_key=args.partition_key,
            )
            record_batch_size = args.dataloader.batch_size
            args.dataloader.batch_size = 1
        elif args.data_name in ["gen-eval"]:
            dataset = MongoDBCaptionDataLoad(
                collection_name=args.data_name,
                query=args.query,
                shard_idx=self.shard_id,
                num_shards=self.num_shards,
                mapping_field=args.mapping_field,
                partition_key=args.partition_key,
            )
        else:
            raise ValueError(f"Unsupported MongoDB dataset: {args.data_name}")

        dataset.set_local_partition()
        self.dataset = dataset
        data_loader, sampler = self._warp_dataloader_with_stateful_sampler(
            args, dataset
        )
        args.dataloader.batch_size = record_batch_size
        return data_loader, sampler

    def _warp_dataloader_with_stateful_sampler(
        self, args: DataArgs, dataset: Dataset
    ) -> Tuple[DataLoader, StatefulDistributedSampler]:
        dataloader_args = args.dataloader
        sampler = StatefulDistributedSampler(
            dataset,
            batch_size=dataloader_args.batch_size,
            num_replicas=self.num_shards,
            rank=self.shard_id,
            shuffle=dataloader_args.shuffle,
        )
        return (
            DataLoader(
                dataset,
                batch_size=dataloader_args.batch_size,
                sampler=sampler,
                worker_init_fn=partial(worker_init, seed=dataloader_args.seed),
                drop_last=dataloader_args.drop_last,
                pin_memory=dataloader_args.pin_memory,
                num_workers=dataloader_args.num_workers,
                persistent_workers=True if dataloader_args.num_workers > 0 else False,
                prefetch_factor=(
                    dataloader_args.prefetch_factor
                    if dataloader_args.num_workers > 0
                    else None
                ),
            ),
            sampler,
        )
