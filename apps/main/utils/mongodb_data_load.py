import io
import os
import requests
import logging
from typing import Any
import time
import pandas as pd

import tempfile
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Final
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from bson import json_util, ObjectId
import pandas as pd
import pyarrow.parquet as pq
import bisect
from apps.main.modules.preprocess import ImageProcessing
import time
import random
import numpy as np
import torch

logging.getLogger("pymongo").setLevel(logging.WARNING)


load_dotenv()

# Iniitialize
MONGODB_URI: Final[str] = os.environ["MONGODB_URI"]
MONGODB_USER: Final[str] = os.environ["MONGODB_USER"]
MONGODB_PASSWORD: Final[str] = os.environ["MONGODB_PASSWORD"]
encoded_user = quote_plus(MONGODB_USER)
encoded_password = quote_plus(MONGODB_PASSWORD)
MONGODB_URI = f"mongodb+srv://{encoded_user}:{encoded_password}@{MONGODB_URI}"
LOCAL_TEMP_DIR: Final[str] = "/dev/shm"


# TODO: Add the logic of MongoDB data loading here
class MongoDBDataLoad(Dataset):
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
        partition_key: str,
        query: dict[str, Any],
    ) -> None:
        super().__init__()
        assert shard_idx >= 0 and shard_idx < num_shards, "Invalid shard index"
        self.collection_name = collection_name
        self.query = query
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        self.data = None
        self.partition_key = partition_key

    def set_local_partition(self):
        """
        we need a partition key to split the data into multiple shards.
        Generate the partion key:
        db["imagenet-1k"].updateMany(
          {}, // Match all documents
          [
            {
              $set: {
                key: { $floor: { $multiply: [ { $rand: {} }, 1000000 ] } }
              }
            }
          ]
        );
        """
        logging.info("Data partition begins!")
        client = MongoClient(MONGODB_URI)
        db = client["world_model"]
        collection = db[self.collection_name]
        self.query.update(
            {
                "$expr": {
                    "$eq": [
                        {
                            "$mod": [
                                {"$toInt": f"${self.partition_key}"},
                                self.num_shards,  # Total number of shards
                            ]
                        },
                        self.shard_idx,  # Current shard index
                    ]
                }
            }
        )
        logging.info(f"Query: {self.query}")

        start_time = time.time()  # Record the start time
        # * download the sub table head for this shard gpu
        # self.data = list(collection.find(self.query))
        self.data = pd.DataFrame(list(collection.find(self.query))).reset_index()
        end_time = time.time()  # Record the end time
        # Calculate the duration in seconds
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        logging.info(
            f"Data Index retrieval from MongoDB completed in {int(minutes)} minutes and {seconds:.2f} seconds."
        )

        client.close()

    def __len__(self) -> int:
        return self.data.index.max()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


class MongoDBImageNetDataLoad(MongoDBDataLoad):
    def __init__(
        self,
        num_shards,
        shard_idx,
        collection_name,
        partition_key,
        args,
    ) -> None:
        if args.split == "train":
            query = {"split": "train"}
        elif args.split == "validation":
            query = {"split": "validation"}
        else:
            raise ValueError(f"Invalid split: {args.split}")
        super().__init__(
            num_shards=num_shards,
            shard_idx=shard_idx,
            collection_name=collection_name,
            query=query,
            partition_key=partition_key,
        )
        self.image_processing = ImageProcessing(args)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cur_data = self.data.iloc[idx]
        image_url = cur_data["image"]
        image = Image.open(image_url)
        return {
            "image": self.image_processing.transform(image),
            "label": cur_data["label"],
            "caption": cur_data["caption"],
            "_id": str(cur_data["_id"]),
        }


class MongoDBCC12MDataLoad(MongoDBDataLoad):
    def __init__(
        self,
        num_shards,
        shard_idx,
        query,
        collection_name,
        extract_field,
        partition_key,
        args,
    ) -> None:
        super().__init__(
            num_shards=num_shards,
            shard_idx=shard_idx,
            collection_name=collection_name,
            query=query,
            partition_key=partition_key,
        )
        self.image_processing = ImageProcessing(args)
        self.extract_field = extract_field
        self.retries = args.retries
        self.place_holder_image = Image.new("RGB", (args.image_size, args.image_size))

    def __getitem__(self, idx: int) -> dict[str, Any]:

        # sample = self.data[idx]
        # for pd data
        sample = self.data.iloc[idx]  # Use iloc for row access in DataFrame
        return_sample = {}
        return_sample["_id"] = str(sample["_id"])
        return_sample["caption"] = sample["caption"]
        for k, v in self.extract_field.items():
            imageUrl = sample[k]
            for attempt in range(self.retries):
                try:
                    response = requests.get(imageUrl)
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    if random.random() > 0.9:
                        self.place_holder_image = (
                            image  # frequently update the placeholder image
                        )
                    return_sample[v] = self.image_processing.transform(image)
                    break
                except Exception as e:
                    logging.warning(
                        f"Attempt {attempt + 1}/{self.retries} - Error loading image: {e}"
                    )
                    if attempt == self.retries - 1:
                        image = self.place_holder_image
                        return_sample[v] = self.image_processing.transform(image)
                        return_sample["_id"] = "-1"
                        return_sample["caption"] = ""
        return return_sample


class MongoDBParquetDataLoad(MongoDBDataLoad):
    def __init__(
        self,
        num_shards,
        shard_idx,
        query,
        collection_name,
        extract_field,
        mapping_field,
        partition_key,
        parallel_parquet=4,
        batch_size=64,
    ) -> None:
        super().__init__(
            num_shards=num_shards,
            shard_idx=shard_idx,
            collection_name=collection_name,
            query=query,
            partition_key=partition_key,
        )
        self.index_boundaries = []  # Cumulative row boundaries for each file
        self.current_df = None
        self.current_file = None
        self.num_field = extract_field["parquet_size"]
        self.path_field = extract_field["parquet_path"]
        self.mapping_field = mapping_field
        self.parallel_parquet = parallel_parquet
        self.batch_size = batch_size
        self.place_holder_parquet = None

    def __len__(self):
        """Return the total number of rows in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            idx = idx % len(self)
        file = self.data.iloc[idx][self.path_field]
        try:
            # updated to use memory-mapped reading
            table = pq.read_parquet(file, memory_map=True)
            cur_df = table.to_pandas()
            self.place_holder_parquet = file
        except Exception as e:
            logging.warning(f"Error reading parquet file: {file}")
            if self.place_holder_parquet is not None:
                table = pq.read_parquet(self.place_holder_parquet, memory_map=True)
                cur_df = table.to_pandas()
            else:
                return self.__getitem__(random.choice(range(len(self))))
        return_parquet = {}
        for i, sample in cur_df.iterrows():
            sample = sample.to_dict()
            return_sample = {}
            for k, v in sample.items():
                if k in self.mapping_field:
                    k_ = self.mapping_field[k]
                else:
                    k_ = k
                if isinstance(v, ObjectId):
                    return_sample[k_] = [str(v)]
                if isinstance(v, np.ndarray) and "raw_shape" not in k:
                    raw_shape_key = f"{k}_raw_shape"
                    if raw_shape_key in sample:
                        return_sample[k_] = v.reshape(sample[raw_shape_key])
                        return_sample[k_] = [torch.Tensor(np.copy(return_sample[k_]))]
                    elif raw_shape_key in self.data.iloc[idx]:
                        return_sample[k_] = v.reshape(
                            self.data.iloc[idx][raw_shape_key]
                        )
                        return_sample[k_] = [torch.Tensor(np.copy(return_sample[k_]))]
                    else:
                        return_sample[k_] = [torch.Tensor(np.copy(v))]
            if i == 0:
                return_parquet = return_sample
            else:
                for k, v in return_sample.items():
                    return_parquet[k].extend(v)
        for k, v in return_parquet.items():
            if isinstance(v[0], torch.Tensor):
                return_parquet[k] = torch.stack(v, dim=0)
        return return_parquet


class DictTensorBatchIterator:
    def __init__(self, data_dict, batch_size):
        """
        Initialize the iterator for batching a dictionary containing tensors and strings.

        Args:
            data_dict (dict): Dictionary where keys map to strings or tensors.
            batch_size (int): Desired batch size for the tensors.
        """
        self.data_dict = data_dict
        self.batch_size = batch_size

        # Validate and prepare tensors
        self.tensor_keys = [
            key for key, value in data_dict.items() if isinstance(value, torch.Tensor)
        ]

        self.total_batches = None
        self.current_batch = 0

        # Remove singleton dimensions (first dimension = 1)
        for key in self.tensor_keys:
            tensor = data_dict[key]
            if tensor.shape[0] == 1:
                self.data_dict[key] = tensor.squeeze(0)  # Remove singleton dimension

        # Calculate the total number of batches (using the first tensor's shape)
        if self.tensor_keys:
            self.total_batches = (
                self.data_dict[self.tensor_keys[0]].shape[0] // batch_size
            )

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return the next batch of the dictionary.

        Returns:
            dict: A dictionary with batched tensors and unchanged strings.

        Raises:
            StopIteration: If all batches are processed.
        """
        if self.tensor_keys and self.current_batch >= self.total_batches:
            raise StopIteration

        batch = {}
        for key, value in self.data_dict.items():
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            batch[key] = value[start_idx:end_idx]

        self.current_batch += 1
        return batch
