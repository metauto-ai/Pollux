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


class CC12MDataset(IterableDataset):
    """
    Dataset for CC12M dataset
    """
    DB_NAME = "world_model"
    COLLECTION_NAME = "cc12m"
    DOC_ID_FIELD = "_id"
    IMAGE_URL_FIELD = "s3url"

    def __init__(self):
        collection = MongoClient(MONGODB_URI)[self.DB_NAME][self.COLLECTION_NAME]
        collection.create_index([("aesthetic_score", 1)])
        self.collection = collection

    def __iter__(self):
        logger.info("Starting to iterate over documents")
        for doc in tqdm(self.collection.find(
            {"aesthetic_score": {"$exists": False}},
        ).batch_size(1024)):
            logger.info(f"Processing document: {doc[self.DOC_ID_FIELD]}")
            yield {
                "document_ids":  str(doc[self.DOC_ID_FIELD]), 
                "image_urls": str(doc[self.IMAGE_URL_FIELD])
            }
    
    def bulkUpdate(self, operations):
        return self.collection.bulk_write(operations, ordered=False)


class PD12MDataset(IterableDataset):
    """
    Dataset for PD12M dataset
    """
    DB_NAME = "world_model"
    COLLECTION_NAME = "pd12m"
    DOC_ID_FIELD = "_id"
    IMAGE_URL_FIELD = "s3url"

    def __init__(self):
        collection = MongoClient(MONGODB_URI)[self.DB_NAME][self.COLLECTION_NAME]
        collection.create_index([("aesthetic_score", 1)])
        self.collection = collection

    def __iter__(self):
        logger.info("Starting to iterate over documents")
        for doc in tqdm(self.collection.find(
            {"aesthetic_score": {"$exists": False}},
        ).batch_size(128 * 1024)):
            logger.info(f"Processing document: {doc[self.DOC_ID_FIELD]}")
            yield {
                "document_ids": str(doc[self.DOC_ID_FIELD]), 
                "image_urls": str(doc[self.IMAGE_URL_FIELD])
            }
            
    def bulkUpdate(self, operations):
        return self.collection.bulk_write(operations, ordered=False)