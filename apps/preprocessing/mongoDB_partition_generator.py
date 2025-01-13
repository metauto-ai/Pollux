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

    def model_inference(self, docs):
        for i in range(len(docs)):
            docs[i][self.new_field] = random.randint(0, 100000)
        return docs

    def _update_document(self, filter_query, update_data):
        """
        Function to perform an update operation on a document.
        """
        result = self.collection.update_one(filter_query, update_data)
        return result.matched_count, result.modified_count

    def update_data(self, docs):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._update_document,
                    {"_id": doc["_id"]},
                    {"$set": {self.new_field: doc[self.new_field]}},
                )
                for doc in docs
            ]
            for future in futures:
                try:
                    matched, modified = future.result()
                except Exception as e:
                    logging.warning(f"Error: {e}")
        return len(docs)

    def run(self):
        docs = self.random_sample()
        model_inference_docs = self.model_inference(docs)
        process_count = self.update_data(model_inference_docs)
        return process_count


if __name__ == "__main__":
    updater = MongoDBPartitionGenerator(
        collection_name="big35m_new",
        new_field="partition_key",
        batch_size=256,
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
