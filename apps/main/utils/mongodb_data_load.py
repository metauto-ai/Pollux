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


load_dotenv()

# Iniitialize
MONGODB_URI: Final[str] = os.environ["MONGODB_URI"]
MONGODB_USER: Final[str] = os.environ["MONGODB_USER"]
MONGODB_PASSWORD: Final[str] = os.environ["MONGODB_PASSWORD"]
encoded_user = quote_plus(MONGODB_USER)
encoded_password = quote_plus(MONGODB_PASSWORD)
MONGODB_URI = f"mongodb+srv://{encoded_user}:{encoded_password}@{MONGODB_URI}"

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