import os
import time
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


class MongoDataset(IterableDataset):
    DB_NAME = "world_model"
    DOC_ID_FIELD = "_id"
    BAD_IMAGE_FIELD = "bad_image"

    def __init__(self, mongo_config):
        self.collection_name = mongo_config["collection_name"]
        self.doc_id_field = mongo_config.get("doc_id_field", self.DOC_ID_FIELD)
        self.image_url_field = mongo_config["url_field"]
        self.caption_field = mongo_config.get("caption_field", None)
        self.num_shards = mongo_config.get("num_shards", 1)
        self.shard_key = mongo_config.get("shard_key", "hash")
        self.shard_idx = mongo_config.get("shard_idx", 0)
        self.rate_limit = mongo_config.get("rate_limit", -1)
        self.update_field = mongo_config.get("update_field", None)

        self.query = {
            "$expr": {
                    "$eq": [
                        {
                            "$mod": [
                                {"$toLong": f"${self.shard_key}"},
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
        max_retries = 3
        base_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                return self.collection.bulk_write(operations, ordered=False)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Bulk write failed after {max_retries} attempts: {str(e)}")
                    # raise  # Re-raise the exception if we've exhausted all retries
                    return None
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Bulk write failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    def __iter__(self):
        self.set_collection()
        # self.collection.create_index([("clip_score", 1)])
        logger.info("Starting to iterate over documents")
        
        if self.update_field:
            self.query.update({self.update_field: {"$exists": False}})
            if self.update_field != "aesthetic_score":
                self.query.update({"aesthetic_score": {"$gt": 4.5}})
        
        projection = {self.doc_id_field: 1, self.image_url_field: 1, self.BAD_IMAGE_FIELD: 1}
        if self.caption_field:
            projection[self.caption_field] = 1
        
        logger.info(f"Query: {self.query}")
        logger.info(f"Projection: {projection}")

        for doc in tqdm(self.collection.find(
            self.query,
            projection=projection,
        ).batch_size(128 * 1024)):
            if self.BAD_IMAGE_FIELD in doc and doc[self.BAD_IMAGE_FIELD]:
                logger.info(f"Skipping bad image document: {doc[self.doc_id_field]}")
                continue

            skip = self.image_url_field not in doc
            if self.caption_field:
                skip = skip or self.caption_field not in doc
            if skip:
                logger.info(f"Skipping document: {doc[self.doc_id_field]}")
                continue
            yield {
                "doc_id": str(doc[self.doc_id_field]), 
                "image_url": str(doc[self.image_url_field]),
                "caption": str(doc[self.caption_field]) if self.caption_field else ""
            }
            if self.rate_limit > 0:
                time.sleep(1 / self.rate_limit)


if __name__ == "__main__":
    # test
    dataset = MongoDataset(collection_name="laion1b", image_url_field="URL", caption_field="TEXT", num_shards=1, shard_idx=0)
    for doc in tqdm(dataset):
        # print(doc)
        pass
