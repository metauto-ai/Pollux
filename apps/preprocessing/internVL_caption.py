import os
import glob
import json
import base64
from PIL import Image
import io
import certifi
from concurrent.futures import ThreadPoolExecutor, as_completed
from apps.main.utils.mongodb_data_load import MONGODB_URI
from pymongo import MongoClient
import logging
import requests
import apps.preprocessing.wandb_img as wandb_img
import uuid
import re
import time
from typing import Final
from apps.main.utils.mongodb_data_load import MONGODB_URI, MongoDBDataLoad
from typing import Any
from torch.utils.data import DataLoader
import base64
from openai import OpenAI
from apps.preprocessing.wandb_img import WandBLogger
import random
import string
from bson import ObjectId

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

Image.MAX_IMAGE_PIXELS = None
ERROR_ID: Final[str] = "ERROR_ID"


class MongoDBFlickrMetaDataLoad(MongoDBDataLoad):
    def __init__(
        self, num_shards, shard_idx, collection_name, partition_key, caption_field
    ) -> None:
        self.caption_field = caption_field
        query = {f"{self.caption_field}": {"$exists": False}}
        super().__init__(
            num_shards=num_shards,
            shard_idx=shard_idx,
            collection_name=collection_name,
            query=query,
            partition_key=partition_key,
        )

    def __len__(self):
        """Return the total number of rows in the dataset."""
        return len(self.data)

    def process_image(self, image):
        width, height = image.size
        max_dim = max(width, height)

        if max_dim > 448:
            scale = 448 / max_dim
        else:
            # If the size is already 512, return the original image
            return image

        # Calculate new dimensions while keeping the aspect ratio
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    def generate_random_string(self, length=10):
        # Characters to choose from: all ASCII letters and digits
        characters = string.ascii_letters + string.digits
        # Generate a random string of specified length
        random_string = "".join(random.choice(characters) for _ in range(length))
        return random_string

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cur_data = self.data.iloc[idx]
        image_url = cur_data["url"]
        try:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image = self.process_image(image)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            image_bytes = buffer.getvalue()
            image_bytes = base64.b64encode(image_bytes).decode("utf-8")
            return {
                "image": image_bytes,
                "url": cur_data["url"],
                "_id": str(cur_data["_id"]),
            }
        except Exception as e:
            logging.warning(f"Error in handling {cur_data['_id']}:{e}")
        return {
            "image": self.generate_random_string(100),
            "url": cur_data["url"],
            "_id": ERROR_ID,
        }


class InternVL_Captioner:
    def __init__(
        self, num_shard, shard_idx, collection_name, partition_key, caption_field
    ):
        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

        models = self.client.models.list()
        self.model = models.data[0].id
        self.system_prompt = """
            You are tasked with generating image captions within 2-3 short and natural sentences.
            Guidelines for generating image captions:
            1. Please generate a single-paragraph caption that includes every visible element in the image, including any readable text. 
            2. Ensure the caption is free from imaginary content or hallucinations. 
            3. Present all information in one cohesive narrative without using structured lists.
            Use the subject as the main focus of your caption and incorporate elements from the style to enhance the description. 
            """
        self.caption_field = caption_field

        mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = mongodb_client["world_model"]
        self.collection = db[collection_name]

        dataset = MongoDBFlickrMetaDataLoad(
            num_shards=num_shard,
            shard_idx=shard_idx,
            collection_name=collection_name,
            partition_key=partition_key,
            caption_field=caption_field,
        )
        dataset.set_local_partition()
        logging.warning(f"Dataset length: {len(dataset)}")
        self.data_len = len(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=64,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=8,
        )
        self.data_iterator = iter(data_loader)
        self.count = 0
        self.start_time = time.time()

    def run(self):
        batch = next(self.data_iterator)
        with ThreadPoolExecutor(max_workers=len(batch["image"])) as executor:
            future_to_id = {
                executor.submit(
                    self.pipeline,
                    {"_id": batch["_id"][idx], "image": batch["image"][idx]},
                ): batch["_id"][idx]
                for idx in range(len(batch["image"]))
            }

            for future in as_completed(future_to_id):
                _id = future_to_id[future]
                self.count += future.result()
        logging.warning(
            f"Total number of images processed: {self.count} out of {self.data_len} and time taken (hours): {(time.time() - self.start_time) / 3600}"
        )

    def pipeline(self, data):
        if data["_id"] == ERROR_ID:
            return 0
        try:
            caption = self.submit_data(data)
            self.update_data(data["_id"], caption)
        except Exception as e:
            logging.warning(f"Error in handling {data['_id']}:{e}")
            return 0
        return 1

    def submit_data(self, data):
        chat_completion_from_base64 = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{data['image']}",
                            },
                        },
                    ],
                }
            ],
            model=self.model,
            max_tokens=128,
        )
        return chat_completion_from_base64.choices[0].message.content

    def update_data(self, _id, caption):
        query = {"_id": ObjectId(_id)}
        update = {"$set": {f"{self.caption_field}": caption}}
        self.collection.update_one(query, update)


if __name__ == "__main__":
    captioner = InternVL_Captioner(
        num_shard=1,  # Nodes Number
        shard_idx=0,  # Node Index
        collection_name="flickr",
        partition_key="partition_key",
        caption_field="internVL_caption",
    )
    while True:
        captioner.run()
