import io
import os
import requests
import logging
from typing import Any
import time

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


from apps.main.modules.preprocess import ImageProcessing
import time
import random


logging.getLogger("pymongo").setLevel(logging.WARNING)


load_dotenv()

# Iniitialize
MONGODB_URI: Final[str] = os.environ["MONGODB_URI"]
MONGODB_USER: Final[str] = os.environ["MONGODB_USER"]
MONGODB_PASSWORD: Final[str] = os.environ["MONGODB_PASSWORD"]
encoded_user = quote_plus(MONGODB_USER)
encoded_password = quote_plus(MONGODB_PASSWORD)
MONGODB_URI = f"mongodb+srv://{encoded_user}:{encoded_password}@{MONGODB_URI}"
LOCAL_TEMP_DIR: Final[str] = "/dev/shm/"


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
        temporal_cache_name: str,
        partition_key: str,
        query: dict[str, Any], 
    ) -> None:
        super().__init__()
        assert shard_idx >= 0 and shard_idx < num_shards, "Invalid shard index"
        self.pkl_res = str(
            Path(LOCAL_TEMP_DIR)
            / f"{temporal_cache_name}_mongo_{collection_name}_{query}.json"
        )
        self.finish_signal = str(
            Path(LOCAL_TEMP_DIR)
            / f"{temporal_cache_name}_mongo_{collection_name}_{query}_ready.signal"
        )
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
                                {
                                    "$toInt": f"${self.partition_key}"
                                },  
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
        self.data = list(collection.find(self.query))
        end_time = time.time()  # Record the end time
        # Calculate the duration in seconds
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        logging.info(f"Data Index retrieval from MongoDB completed in {int(minutes)} minutes and {seconds:.2f} seconds.")
                
        client.close()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


class MongoDBImageNetDataLoad(MongoDBDataLoad):
    def __init__(
        self,
        num_shards,
        shard_idx,
        collection_name,
        temporal_cache_name,
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
            temporal_cache_name=temporal_cache_name,
            partition_key=partition_key,
        )
        self.image_processing = ImageProcessing(args)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_url = self.data[idx]["image"]
        image = Image.open(image_url)
        return {
            "image": self.image_processing.transform(image),
            "label": self.data[idx]["label"],
            "caption": self.data[idx]["caption"],
            "_id": str(self.data[idx]["_id"]),
        }


class MongoDBCC12MDataLoad(MongoDBDataLoad):
    def __init__(
        self,
        num_shards,
        shard_idx,
        query,
        collection_name,
        temporal_cache_name,
        extract_field,
        partition_key,
        args,
    ) -> None:
        super().__init__(
            num_shards=num_shards,
            shard_idx=shard_idx,
            collection_name=collection_name,
            query=query,
            temporal_cache_name=temporal_cache_name,
            partition_key=partition_key,
        )
        self.image_processing = ImageProcessing(args)
        self.extract_field = extract_field
        self.retries = args.retries
        self.place_holder_image = Image.new("RGB", (args.image_size, args.image_size))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.data[idx]
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
