from apps.main.utils.mongodb_data_load import MONGODB_URI
from pymongo import MongoClient
import certifi
import logging
from PIL import Image
import requests
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
Image.MAX_IMAGE_PIXELS = None


class MongoDBPartitionGenerator:
    def __init__(
        self,
        collection_name,
        new_field,
        batch_size,
    ):
        mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = mongodb_client["world_model"]
        self.collection = db[collection_name]
        self.batch_size = batch_size
        self.max_workers = batch_size
        self.new_field = new_field
        self.query = {f"{self.new_field}": {"$exists": False}}

        logging.info(f"Query: {self.query}")

    def random_sample(self):
        cursor = self.collection.find(self.query).limit(self.batch_size)
        return list(cursor)

    def _update_document(self, filter_query, update_data):
        """
        Function to perform an update operation on a document.
        """
        result = self.collection.update_one(filter_query, update_data)
        return result.matched_count, result.modified_count

    def update_data(self, docs):
        ids = [doc["_id"] for doc in docs]
        self.collection.update_many(
            {"_id": {"$in": ids}},  # Match documents in the batch
            [
                {
                    "$set": {
                        f"{self.new_field}": {
                            "$floor": {"$multiply": [{"$rand": {}}, 100000]}
                        }
                    }
                }
            ],
        )
        return len(docs)

    def run(self):
        docs = self.random_sample()
        process_count = self.update_data(docs)
        return process_count


if __name__ == "__main__":
    updater = MongoDBPartitionGenerator(
        collection_name="big35m_new",
        new_field="partition_key",
        batch_size=10000,
    )
    total_acount = 0
    while True:
        try:
            process_count = updater.run()
            total_acount += process_count
        except Exception as e:
            logging.warning(f"Error: {e}")
            break
        logging.info(f"Total Processed: {total_acount}")
