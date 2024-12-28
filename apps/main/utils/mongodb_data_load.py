import io
import os
import requests
import logging
from typing import Any


from urllib.parse import quote_plus
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Final
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from apps.main.modules.preprocessing import ImageProcessing
from pathlib import Path
import tempfile
from bson import json_util, ObjectId
import time

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
        init_signal_handler: bool,
        temporal_cache_name: str,
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
        self.init_signal_handler = init_signal_handler
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        self.data = None

    def set_local_cache(self):
        if self.init_signal_handler:
            if os.path.exists(self.finish_signal):
                with open(f"{self.pkl_res}", "r") as f:
                    json_data = f.read()
                self.data = json_util.loads(json_data)
            else:
                client = MongoClient(MONGODB_URI)
                db = client["world_model"]
                collection = db[self.collection_name]
                self.data = list(collection.find(self.query))
                client.close()
                with open(
                    self.pkl_res,
                    "w",
                ) as temp_file:
                    logging.info(f"Temporary file in tmpfs: {temp_file.name}")
                    temp_file.write(json_util.dumps(self.data))
                open(f"{self.finish_signal}", "w").close()
        else:
            logging.info("Waiting for data readiness signal...")
            while not os.path.exists(self.finish_signal):
                time.sleep(5)
            with open(f"{self.pkl_res}", "r") as f:
                json_data = f.read()
            self.data = json_util.loads(json_data)

    def set_sharding(self):
        shard_size = len(self.data) // self.num_shards
        remainder = len(self.data) % self.num_shards
        start = self.shard_idx * shard_size + min(self.shard_idx, remainder)
        end = start + shard_size + (1 if self.shard_idx < remainder else 0)
        self.data = self.data[start:end]

    def clean_buffer(self):
        os.remove(f"{self.finish_signal}")
        os.remove(f"{self.pkl_res}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


class MongoDBImageNetDataLoad(MongoDBDataLoad):
    def __init__(
        self,
        num_shards,
        shard_idx,
        collection_name,
        init_signal_handler,
        temporal_cache_name,
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
            init_signal_handler=init_signal_handler,
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
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
