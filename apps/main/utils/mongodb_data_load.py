import io
import os
import requests
import logging
from typing import Any
import time
import pandas as pd

import certifi
from urllib.parse import quote_plus
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
from apps.main.modules.preprocess import PolluxImageProcessing
import time
import random
import numpy as np
import torch
import s3fs
import boto3
import wandb
from apps.main.utils.dict_tensor_data_load import DictTensorBatchIterator
import ijson
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.getLogger("pymongo").setLevel(logging.WARNING)
boto3.set_stream_logger("boto3", level=logging.WARNING)
boto3.set_stream_logger("botocore", level=logging.WARNING)
logging.getLogger("s3fs").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.ERROR)
Image.MAX_IMAGE_PIXELS = None
from apps.main.utils.env import env

MONGODB_URI = env.MONGODB_URI
S3KEY = env.S3KEY
S3SECRET = env.S3SECRET


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
        root_dir: str = None,
    ) -> None:
        super().__init__()
        assert shard_idx >= 0 and shard_idx < num_shards, "Invalid shard index"
        self.collection_name = collection_name
        self.query = query
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        self.data = None
        self.partition_key = partition_key
        self.root_dir = root_dir

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
        start_time = time.time()  # Record the start time
        if self.root_dir is None:
            client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
            db = client["world_model"]
            collection = db[self.collection_name]
            self.query.update(
                {
                    f"{self.partition_key}": {"$exists": True},
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
                    },
                }
            )
            logging.info(f"Query: {self.query}")

            # * download the sub table head for this shard gpu
            # self.data = list(collection.find(self.query))
            self.data = pd.DataFrame(list(collection.find(self.query))).reset_index()

            client.close()
        else:
            logging.info(f"Loading data from local parquet files: {self.root_dir}")

            file_path = os.path.join(self.root_dir, f"{self.collection_name}.json")
            data = []
            with open(file_path, "r") as file:
                for item in tqdm(ijson.items(file, "item"), desc=f"Loading data to shard {self.shard_idx}"):
                    partition_key = int(item[self.partition_key])
                    if partition_key % self.num_shards == self.shard_idx:
                        data.append(item)
                        # # Note: used for debugging
                        # if len(data) > 1400000:
                        #     break
            self.data = pd.DataFrame(data).reset_index()
        end_time = time.time()  # Record the end time
        # Calculate the duration in seconds
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        logging.info(
            f"Data Index retrieval from MongoDB completed in {int(minutes)} minutes and {seconds:.2f} seconds."
        )

    def __len__(self) -> int:
        return self.data.index.max()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


class MongoDBImageDataLoad(MongoDBDataLoad):
    def __init__(
        self,
        num_shards,
        shard_idx,
        query,
        root_dir,
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
            root_dir=root_dir,
        )
        self.image_processing = PolluxImageProcessing(args)
        self.extract_field = extract_field
        self.retries = args.retries
        self.place_holder_image = Image.new("RGB", (args.image_size, args.image_size))
        
        # Create a session with connection pooling and retry strategy
        self.session = requests.Session()
        retries = Retry(
            total=self.retries,
            backoff_factor=0.5,  # Exponential backoff
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET"]
        )
        # Increase max connections per host
        adapter = HTTPAdapter(max_retries=retries, pool_connections=200, pool_maxsize=200)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # sample = self.data[idx]
        # for pd data
        sample = self.data.iloc[idx]  # Use iloc for row access in DataFrame
        return_sample = {}
        return_sample["_id"] = str(sample["_id"])
        caption = sample["caption"]
        if isinstance(caption, tuple):
            caption = caption[0]

        if not isinstance(caption, str):
            logging.warning(f"Expected string but got {type(caption)}:{caption}")
            caption = ""
        return_sample["caption"] = caption
        
        for k, v in self.extract_field.items():
            imageUrl = sample[k]
            try:
                head_response = self.session.head(imageUrl, timeout=1)
                if head_response.status_code != 200:
                    raise requests.HTTPError(f"HEAD request failed with status code {head_response.status_code}")

                # Use session and increase timeout
                response = self.session.get(imageUrl, timeout=2, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                if random.random() > 0.9:
                    self.place_holder_image = image  # frequently update the placeholder image
                
                return_sample[v], return_sample[f"{v}_cond"] = self.image_processing.transform(image)
            except (requests.RequestException, IOError) as e:
                # Handle all request and image processing errors
                if isinstance(e, requests.Timeout):
                    logging.debug(f"Timeout downloading image: {imageUrl}")
                elif isinstance(e, requests.HTTPError):
                    logging.debug(f"HTTP error ({head_response.status_code}) for: {imageUrl}")
                elif isinstance(e, requests.ConnectionError):
                    logging.debug(f"Connection error for: {imageUrl}")
                else:
                    logging.debug(f"Error processing image: {str(e)}")
                
                # Fall back to placeholder image
                image = self.place_holder_image
                return_sample[v], return_sample[f"{v}_cond"] = self.image_processing.transform(image)
                return_sample["_id"] = "-1"
                return_sample["caption"] = ""
                
        return return_sample
    
    def __del__(self):
        # Clean up the session when the dataset object is destroyed
        if hasattr(self, 'session'):
            self.session.close()
            
    def collate_fn(self, batch):
        return_batch = {}
        for k in batch[0].keys():
            items = [item[k] for item in batch]
            # Check if all items are tensors and have the same shape
            if all(isinstance(item, torch.Tensor) for item in items) and all(item.shape == items[0].shape for item in items):
                # Stack tensors if they all have the same shape
                return_batch[k] = torch.stack(items, dim=0)
            else:
                # Keep as list if not tensors or different shapes
                return_batch[k] = items
        return return_batch


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
        root_dir,
        parallel_parquet=4,
        batch_size=64,
    ) -> None:
        super().__init__(
            num_shards=num_shards,
            shard_idx=shard_idx,
            collection_name=collection_name,
            query=query,
            partition_key=partition_key,
            root_dir=root_dir,
        )
        self.index_boundaries = []  # Cumulative row boundaries for each file
        self.current_df = None
        self.current_file = None
        self.num_field = extract_field["parquet_size"]
        self.path_field = extract_field["parquet_path"]
        self.mapping_field = mapping_field
        self.parallel_parquet = parallel_parquet
        self.batch_size = batch_size

    def __len__(self):
        """Return the total number of rows in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            idx = idx % len(self)
        file = self.data.iloc[idx][self.path_field]
        try:
            # updated to use memory-mapped reading
            if file.startswith("s3://"):
                cur_df = pd.read_parquet(
                    file,
                    storage_options={
                        "key": S3KEY,
                        "secret": S3SECRET,
                    },
                )
            elif os.path.exists(file):
                table = pq.read_table(file, memory_map=True)
                cur_df = table.to_pandas()
            else:
                logging.warning(f"Invalid path or file not found: {file}")
        except Exception as e:
            logging.warning(f"Error reading parquet file: {file}")
            return self.__getitem__(random.choice(range(len(self))))
        return_parquet = {}
        for i, sample in cur_df.iterrows():
            sample = sample.to_dict()
            return_sample = {}
            for k, v in sample.items():
                if k in self.mapping_field:
                    k_ = self.mapping_field[k]
                    if isinstance(v, ObjectId):
                        return_sample[k_] = [str(v)]
                    elif isinstance(v, np.ndarray) and "raw_shape" not in k:
                        raw_shape_key = f"{k}_raw_shape"
                        if raw_shape_key in sample:
                            return_sample[k_] = v.reshape(sample[raw_shape_key])
                            return_sample[k_] = [
                                torch.Tensor(np.copy(return_sample[k_]))
                            ]
                        elif raw_shape_key in self.data.iloc[idx]:
                            return_sample[k_] = v.reshape(
                                self.data.iloc[idx][raw_shape_key]
                            )
                            return_sample[k_] = [
                                torch.Tensor(np.copy(return_sample[k_]))
                            ]
                        else:
                            return_sample[k_] = [torch.Tensor(np.copy(v))]
                    else:
                        return_sample[k_] = [v]
            if i == 0:
                return_parquet = return_sample
            else:
                for k, v in return_sample.items():
                    return_parquet[k].extend(v)
        # Note: remove becasue of dynamic resoltion
        # for k, v in return_parquet.items():
        #     if isinstance(v[0], torch.Tensor):
        #         return_parquet[k] = torch.stack(v, dim=0)
        return return_parquet


class MongoDBCaptionDataLoad(MongoDBDataLoad):
    def __init__(
        self,
        num_shards,
        shard_idx,
        query,
        collection_name,
        root_dir,
        mapping_field,
        partition_key,
    ) -> None:
        super().__init__(
            num_shards=num_shards,
            shard_idx=shard_idx,
            collection_name=collection_name,
            query=query,
            partition_key=partition_key,
            root_dir=root_dir,
        )
        self.mapping_field = mapping_field

    def __getitem__(self, idx: int) -> dict[str, Any]:

        # sample = self.data[idx]
        # for pd data
        sample = self.data.iloc[idx]  # Use iloc for row access in DataFrame
        return_sample = {}
        return_sample["_id"] = str(sample["_id"])
        for k, v in self.mapping_field.items():
            caption = sample[k]
            if isinstance(caption, tuple):
                caption = caption[0]
            if not isinstance(caption, str):
                logging.warning(f"Expected string but got {type(caption)}:{caption}")
                caption = ""
            return_sample[v] = caption
        return return_sample

    def collate_fn(self, batch):
        return_batch = {}
        for k in batch[0].keys():
            items = [item[k] for item in batch]
            # Check if all items are tensors and have the same shape
            if all(isinstance(item, torch.Tensor) for item in items) and all(item.shape == items[0].shape for item in items):
                # Stack tensors if they all have the same shape
                return_batch[k] = torch.stack(items, dim=0)
            else:
                # Keep as list if not tensors or different shapes
                return_batch[k] = items
        return return_batch