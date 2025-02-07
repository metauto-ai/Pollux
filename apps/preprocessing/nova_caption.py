import os
import glob
import boto3
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
from botocore.config import Config
import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

Image.MAX_IMAGE_PIXELS = None
config = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    max_pool_connections=200,  # Increase pool size (default is 10)
)


def filter_image_caption(text):
    # TODO: could add more logic here if we find sth that needs to be filtered
    pattern = r"(?i)^image caption:\s*"
    return re.sub(pattern, "", text)


class NovaCaption:
    def __init__(
        self,
        collection_name,
        image_field: str,
        caption_field: str = "nova_lite_caption",
        maxTokens: int = 100,
        topP: float = 0.1,
        temperature: float = 1.0,
        max_workers: int = 32,
        batch_size: int = 500,
    ):
        # -------- AWS --------
        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id="AKIA47CRZU7STC4XUXER",
            aws_secret_access_key="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",
            region_name="us-east-1",
            config=config,
        )
        # -------- MongoDB --------
        mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = mongodb_client["world_model"]
        self.collection = db[collection_name]
        self.image_field = image_field
        self.caption_field = caption_field
        self.batch_size = batch_size
        # -------- System Prompt --------

        system_prompt = """You are tasked with generating image prompts that will be used for training diffusion text-to-image models. 
                          This is a reverse-engineering process: you are creating prompts based on the image, generating the textual description a user would provide to produce a similar result.

        Guidelines for generating image prompts:
        1. Please generate a comprehensive, single-paragraph caption that includes every visible element in the image, including any readable text. 
        2. Include information about style, mood, lighting, and composition when relevant.
        3. Use a mix of concrete and abstract terms.
        4. Incorporate artistic references or techniques when appropriate.
        5. Ensure the caption is free from imaginary content or hallucinations. 
        6. Present all information in one cohesive narrative without using structured lists.
        Use the subject as the main focus of your caption and incorporate elements from the style to enhance the description. Combine these elements creatively to produce a unique and engaging caption.
        Begin your response with "Image Caption:"
        """
        self.system = [{"text": system_prompt}]
        # -------- Inference Parameters --------
        self.inf_params = {
            "maxTokens": maxTokens,
            "topP": topP,
            "temperature": temperature,
        }
        self.additionalModelRequestFields = {"inferenceConfig": {"topK": 20}}
        # -------- ThreadPoolExecutor --------
        self.max_workers = max_workers

    def run(self, wandb_logger=None):
        docs = self.read_data_from_mongoDB()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self.batch_process, doc): doc["_id"] for doc in docs
            }

            for future in as_completed(future_to_id):
                _id = future_to_id[future]
                try:
                    response, image_bytes = future.result(timeout=30)
                except Exception as e:
                    logging.warning(f"Erros in handling {_id}:{e}")
                else:

                    # logging.info(f"\n[Full Response for {_id}]")
                    # logging.info(f"\n[Response Content Text for {_id}]")
                    # logging.info(
                    #     f"{json.dumps(response, indent=2, ensure_ascii=False)}"
                    # )
                    caption = response["output"]["message"]["content"][0]["text"]
                    caption = filter_image_caption(caption)
                    if wandb_logger:
                        wandb_logger.add_image(image_bytes, caption)
                    else:
                        self.update_data(_id, caption)
            if wandb_logger:
                wandb_logger.log_images()

    def read_data_from_mongoDB(self):
        query = {f"{self.caption_field}": {"$exists": False}}
        cursor = self.collection.find(query).limit(self.batch_size)
        return list(cursor)

    def process_image(self, image):
        width, height = image.size
        min_dim = min(width, height)

        if min_dim > 512:
            scale = 512 / min_dim
        elif min_dim < 512:
            scale = 256 / min_dim
        else:
            # If the size is already 512, return the original image
            return image

        # Calculate new dimensions while keeping the aspect ratio
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    def download_image(self, img_path):
        if img_path.startswith("s3://") or img_path.startswith("http"):
            s3url = img_path
            response = requests.get(s3url)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image = self.process_image(image)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            image_bytes = buffer.getvalue()
        else:
            with open(img_path, "rb") as f:
                image = Image.open(f).convert("RGB")
                image = self.process_image(image)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)
                buffer.seek(0)
                image_bytes = buffer.getvalue()
        return image_bytes

    def generate_image_caption(self, imagebytes):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": imagebytes},
                        }
                    }
                ],
            }
        ]

        model_response = self.client.converse(
            modelId="us.amazon.nova-lite-v1:0",
            messages=messages,
            system=self.system,
            inferenceConfig=self.inf_params,
            additionalModelRequestFields=self.additionalModelRequestFields,
        )

        return model_response

    def batch_process(self, doc):
        image_bytes = self.download_image(doc[self.image_field])
        response = self.generate_image_caption(image_bytes)
        return response, image_bytes

    def update_data(self, _id, caption):
        query = {"_id": _id}
        update = {"$set": {f"{self.caption_field}": caption}}
        self.collection.update_one(query, update)


if __name__ == "__main__":
    batch_size = 100
    max_samples_per_min = 500
    nova_caption = NovaCaption(
        collection_name="flickr-part-00-of-08",
        image_field="AZURE_URL",
        caption_field="nova_lite_caption",
        maxTokens=150,
        topP=0.1,
        temperature=1.0,
        max_workers=batch_size,
        batch_size=batch_size,
    )

    def read_data_from_mongoDB(self):
        query = {
            f"{self.caption_field}": {"$exists": False},
            "AZURE_URL": {"$exists": True},
            "aesthetic_score": {"$gt": 6.8},
        }
        cursor = self.collection.find(query).limit(self.batch_size)
        return list(cursor)

    nova_caption.read_data_from_mongoDB = types.MethodType(
        read_data_from_mongoDB, nova_caption
    )
    # from apps.preprocessing.wandb_img import WandBLogger

    # wandb_logger = WandBLogger(
    #     project="pollux", run_name="nova_caption_pexel", entity="metauto"
    # )
    start_time = time.time()
    processed_samples = 0
    total_samples = 0
    while True:
        nova_caption.run()

        elapsed_time = time.time() - start_time
        processed_samples += batch_size
        total_samples += batch_size
        if processed_samples >= max_samples_per_min:
            if elapsed_time < 60:
                logging.warning(f"Sleeping for {60 - elapsed_time} seconds")
                time.sleep(60 - elapsed_time)
            start_time = time.time()
            processed_samples = 0
        logging.warning(f"Total samples processed: {total_samples}")
