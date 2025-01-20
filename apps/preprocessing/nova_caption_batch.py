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
            config=config,
        )
        
        # Initialize S3 client for batch operations
        self.s3_client = boto3.client('s3')
        
        # -------- MongoDB --------
        mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = mongodb_client["batch_world_model"]
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
        
        # For batch processing, prepare batch input
        batch_input = []
        for doc in docs:
            batch_input.append({
                "_id": str(doc["_id"]),
                "image_url": doc[self.image_field]
            })
            
        if not batch_input:
            logging.info("No documents to process")
            return
            
        # Upload batch input to S3
        input_key = f"batch-inputs/{uuid.uuid4()}.json"
        self.upload_to_s3(batch_input, input_key)
        
        # Configure batch job
        batch_config = {
            "ModelId": "us.amazon.nova-lite-v1:0",
            "InputLocation": f"s3://{os.environ.get('AWS_BUCKET_NAME')}/{input_key}",
            "OutputLocation": f"s3://{os.environ.get('AWS_BUCKET_NAME')}/batch-outputs/",
            "InferenceConfig": self.inf_params
        }
        
        # Create and monitor batch job
        job_id = self.create_batch_inference_job(batch_config)
        self.monitor_batch_job(job_id)
        
        # Process results
        results = self.download_results_from_s3(f"batch-outputs/{job_id}/output.json")
        logging.info(f"Downloaded {len(results)} results from S3")
        
        for result in results:
            try:
                doc_id = result["_id"]
                caption = filter_image_caption(result["caption"])
                self.update_data(doc_id, caption)
                logging.info(f"Processed result for doc {doc_id}: {caption[:100]}...")
                
                if wandb_logger:
                    image_bytes = self.download_image(result["image_url"])
                    wandb_logger.add_image(image_bytes, caption)
            except Exception as e:
                logging.error(f"Error processing result for doc {doc_id}: {e}")
                
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
            return image

        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    def download_image(self, s3url):
        response = requests.get(s3url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
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

    def create_batch_inference_job(self, batch_config):
        response = self.client.create_batch_job(
            ModelId=batch_config["ModelId"],
            InputLocation=batch_config["InputLocation"],
            OutputLocation=batch_config["OutputLocation"],
            InferenceConfig=batch_config["InferenceConfig"],
        )
        job_id = response["JobId"]
        logging.info(f"Batch job created with ID: {job_id}")
        return job_id

    def monitor_batch_job(self, job_id):
        while True:
            response = self.client.describe_batch_job(JobId=job_id)
            status = response["Status"]
            if status in ["COMPLETED", "FAILED"]:
                logging.info(f"Batch job {job_id} finished with status: {status}")
                break
            logging.info(f"Batch job {job_id} is {status}. Waiting...")
            time.sleep(30)

    def upload_to_s3(self, data, key):
        try:
            self.s3_client.put_object(
                Bucket=os.environ.get('AWS_BUCKET_NAME'),
                Key=key,
                Body=json.dumps(data)
            )
            logging.info(f"Successfully uploaded data to s3://{os.environ.get('AWS_BUCKET_NAME')}/{key}")
        except Exception as e:
            logging.error(f"Error uploading to S3: {e}")
            raise

    def download_results_from_s3(self, key):
        try:
            response = self.s3_client.get_object(
                Bucket=os.environ.get('AWS_BUCKET_NAME'),
                Key=key
            )
            results = json.loads(response['Body'].read())
            logging.info(f"Successfully downloaded results from S3: {key}")
            return results
        except Exception as e:
            logging.error(f"Error downloading from S3: {e}")
            raise

if __name__ == "__main__":
    batch_size = 100
    max_samples_per_min = 500

    nova_caption = NovaCaption(
        collection_name="unsplash_images",
        image_field="s3url",
        caption_field="nova_lite_caption",
        maxTokens=150,
        topP=0.1,
        temperature=1.0,
        max_workers=batch_size,
        batch_size=batch_size,
    )

    start_time = time.time()
    processed_samples = 0
    total_samples = 0
    
    while True:
        nova_caption.run()
        elapsed_time = time.time() - start_time
        processed_samples += batch_size
        total_samples += batch_size
        logging.info(f"Total samples processed: {total_samples}")
        # Check S3 results
        try:
            bucket = os.environ.get('AWS_BUCKET_NAME')
            response = nova_caption.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix='batch-outputs/'
            )
            if 'Contents' in response:
                logging.info(f"Found {len(response['Contents'])} files in S3 batch-outputs/")
                for item in response['Contents'][-5:]:  # Show last 5 files
                    logging.info(f"S3 file: {item['Key']}, Size: {item['Size']} bytes")
            else:
                logging.warning("No files found in S3 batch-outputs/")
        except Exception as e:
            logging.error(f"Error checking S3: {e}")
