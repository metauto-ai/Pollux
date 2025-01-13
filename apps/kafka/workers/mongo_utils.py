import os
from dotenv import load_dotenv
from torch.utils.data import IterableDataset
from typing import Final
from urllib.parse import quote_plus
from pymongo import MongoClient
from loguru import logger
from tqdm import tqdm

load_dotenv()

# Iniitialize
MONGODB_URI: Final[str] = os.environ["MONGODB_URI"]
MONGODB_USER: Final[str] = os.environ["MONGODB_USER"]
MONGODB_PASSWORD: Final[str] = os.environ["MONGODB_PASSWORD"]
encoded_user = quote_plus(MONGODB_USER)
encoded_password = quote_plus(MONGODB_PASSWORD)
MONGODB_URI = f"mongodb+srv://{encoded_user}:{encoded_password}@{MONGODB_URI}"
LOCAL_TEMP_DIR: Final[str] = "/dev/shm"


class MongoDataset():
    DB_NAME = "world_model"
    DOC_ID_FIELD = "_id"
    IMAGE_URL_FIELD = "s3url"

    def __init__(self, collection_name, num_shards=1, shard_idx = 0):
        self.collection_name = collection_name
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        self.collection = None
        self.query = {
            "$expr": {
                    "$eq": [
                        {
                            "$mod": [
                                {"$toLong": f"$seed"},
                                self.num_shards,  # Total number of shards
                            ]
                        },
                        self.shard_idx,  # Current shard index
                    ]
                }
        } if self.num_shards > 1 else {}

    def set_collection(self):
        self.collection = MongoClient(MONGODB_URI)[self.DB_NAME][self.collection_name]

    def bulkUpdate(self, operations):
        if self.collection is None:
            self.set_collection()
        return self.collection.bulk_write(operations, ordered=False)


class CC12MDataset(MongoDataset, IterableDataset):
    """
    Dataset for CC12M dataset
    """
    COLLECTION_NAME = "cc12m"
    
    def __init__(self, num_shards=1, shard_idx = 0):
        super().__init__(self.COLLECTION_NAME, num_shards, shard_idx)

    def __iter__(self):
        self.set_collection()
        self.collection.create_index([("aesthetic_score", 1)])
        logger.info("Starting to iterate over documents")
        self.query.update({"aesthetic_score": {"$exists": False}})
        for doc in tqdm(self.collection.find(
            self.query,
        ).batch_size(16 * 1024)):
            # logger.info(f"Processing document: {doc[self.DOC_ID_FIELD]}")
            yield {
                "document_ids":  str(doc[self.DOC_ID_FIELD]), 
                "image_urls": str(doc[self.IMAGE_URL_FIELD])
            }


class PD12MDataset(MongoDataset, IterableDataset):
    """
    Dataset for PD12M dataset
    """
    COLLECTION_NAME = "pd12m"

    def __init__(self, num_shards=1, shard_idx = 0):
        super().__init__(self.COLLECTION_NAME, num_shards, shard_idx)

    def __iter__(self):
        self.set_collection()
        self.collection.create_index([("aesthetic_score", 1)])
        logger.info("Starting to iterate over documents")
        self.query.update({"aesthetic_score": {"$exists": False}})
        for doc in tqdm(self.collection.find(
            self.query,
        ).batch_size(16 * 1024)):
            # logger.info(f"Processing document: {doc[self.DOC_ID_FIELD]}")
            yield {
                "document_ids": str(doc[self.DOC_ID_FIELD]), 
                "image_urls": str(doc[self.IMAGE_URL_FIELD])
            }


class DiffusionDataset(MongoDataset, IterableDataset):
    """
    Dataset for Diffusion dataset
    """
    COLLECTION_NAME = "diffusion"

    def __init__(self, num_shards=1, shard_idx = 0):
        super().__init__(self.COLLECTION_NAME, num_shards, shard_idx)

    def __iter__(self):
        self.set_collection()
        self.collection.create_index([("aesthetic_score", 1)])
        logger.info("Starting to iterate over documents")
        self.query.update({"aesthetic_score": {"$exists": False}})
        
        for doc in self.collection.find(
            self.query,
        ).batch_size(16 * 1024):
            # logger.info(f"Processing document: {doc[self.DOC_ID_FIELD]}")
            yield {
                "document_ids": str(doc[self.DOC_ID_FIELD]), 
                "image_urls": str(doc[self.IMAGE_URL_FIELD])
            }
