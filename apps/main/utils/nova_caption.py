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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class NovaCaption:
    def __init__(
        self,
        collection_name,
        image_field: str,
        caption_field: str = "nova_lite_caption",
        maxTokens: int = 200,
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
        )
        # -------- MongoDB --------
        mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = mongodb_client["world_model"]
        self.collection = db[collection_name]
        self.image_field = image_field
        self.caption_field = caption_field
        self.batch_size = batch_size
        # -------- System Prompt --------

        system_prompt = """You are tasked with generating image captions that will be used for training diffusion text-to-image models. 
                        Your goal is to create captions that imitate what humans might use as prompts when generating images. 
        Guidelines for generating image captions:
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

    def run(self):
        docs = self.read_data_from_mongoDB()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self.batch_process, doc): doc["_id"] for doc in docs
            }

            for future in as_completed(future_to_id):
                _id = future_to_id[future]
                try:
                    response = future.result()
                except Exception as e:
                    logging.warning(f"Erros in handling {_id}:{e}")
                else:

                    logging.info(f"\n[Full Response for {_id}]")
                    logging.info(json.dumps(response, indent=2, ensure_ascii=False))
                    logging.info(f"\n[Response Content Text for {_id}]")
                    logging.info(response["output"]["message"]["content"][0]["text"])

    def read_data_from_mongoDB(self):
        query = {f"{self.caption_field}": {"$exists": False}}
        cursor = self.collection.find(query).limit(self.batch_size)
        return list(cursor)

    def download_image(self, s3url):
        response = requests.get(s3url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
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
        return response


if __name__ == "__main__":
    nova_caption = NovaCaption(
        collection_name="cc12m",
        image_field="s3url",
        caption_field="nova_lite_caption",
        maxTokens=200,
        topP=0.1,
        temperature=1.0,
        max_workers=2,
        batch_size=5,
    )
    nova_caption.run()
