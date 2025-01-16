from apps.main.utils.mongodb_data_load import MONGODB_URI
from pymongo import MongoClient
import certifi
import logging
from PIL import Image
import requests
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
Image.MAX_IMAGE_PIXELS = None

MEDIA_FIELD = "media"


class MongoDBUpdater:
    def __init__(
        self,
        collection_name,
        media_field,
        new_field,
        shard_idx,
        num_shards,
        batch_size,
        partition_key,
    ):
        mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = mongodb_client["world_model"]
        self.collection = db[collection_name]
        self.batch_size = batch_size
        self.media_field = media_field
        self.max_workers = batch_size
        self.new_field = new_field
        self.shard_idx = shard_idx
        self.num_shards = num_shards
        self.partition_key = partition_key
        self.query = {
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
        self.query.update({f"{self.new_field}": {"$exists": False}})

        logging.info(f"Query: {self.query}")

    def random_sample(self):
        cursor = self.collection.find(self.query).limit(self.batch_size)
        return list(cursor)

    def pre_process_image(self, image):
        return image

    def download_image(self, s3url):
        response = requests.get(s3url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image = self.pre_process_image(image)
        return image

    def model_inference(self, docs):
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
                    print(f"Matched: {matched}, Modified: {modified}")
                except Exception as e:
                    print(f"Error: {e}")

    def run(self):
        docs = self.random_sample()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self.download_image, doc[self.media_field]): doc
                for doc in docs
            }
            self.batch = []
            for future in as_completed(future_to_id):
                doc = future_to_id[future]
                try:
                    image = future.result()
                    doc[MEDIA_FIELD] = image
                    self.batch.append(doc)
                except Exception as e:
                    logging.warning(f"Erros in handling {_id}:{e}")

            docs = self.model_inference(self.batch)
        print(docs)
        # self.update_data(self.batch)


if __name__ == "__main__":
    updater = MongoDBUpdater(
        collection_name="cc12m",
        media_field="s3url",
        new_field="image",
        shard_idx=0,
        num_shards=1,
        batch_size=100,
        partition_key="key",
    )
    updater.run()
