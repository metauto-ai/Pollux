import base64
import io
import multiprocessing
import threading
import time
from workers.kafka_utils import Consumer, Producer
from workers.common import load_yaml_config, print_counter
from loguru import logger
from PIL import Image

import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from transformers import SiglipImageProcessor


STAGE = "aesthetic_scoring"


class ImageDataset(IterableDataset):
    def __init__(self, consumer_topic, rank):
        self.consumer_topic = consumer_topic
        self.rank = rank
        logger.info(f"Initialized ImageDataset for rank {rank}")

    def __iter__(self):
        consumer = Consumer(self.consumer_topic, partition_id=self.rank)
        preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        logger.info(f"Created consumer for topic {self.consumer_topic} and partition {self.rank}")
        for message in consumer.consumer:
            # logger.debug(f"Processing message with {len(message.value['image_contents'])} images")
            message_data = message.value
            try:
                image_contents = [
                    base64.b64decode(image_content) 
                    for image_content in message_data["image_contents"]
                ]
                images = [
                    Image.open(io.BytesIO(image_content)).convert("RGB")
                    for image_content in image_contents
                ]
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                continue
        
            # Store document IDs
            document_id = message_data["document_ids"]

            # Preprocess images
            processed = preprocessor(images=images, return_tensors="pt").pixel_values.to(torch.bfloat16)
        
            # Return one image from the buffer
            for img, doc_id in zip(processed, document_id):
                yield {
                    "images": img,
                    "document_ids": doc_id
                }
        consumer.close()


class AestheticScorerWorker:
    def __init__(self, config, rank, counter):
        self.stage_config = config["stages"][STAGE]
        logger.info(f"Config: {self.stage_config}")
        
        consumer_topic = self.stage_config["consumer"]

        self.model, _ = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attention_2=True
        )
        
        # Move model to device and convert to bfloat16 after initialization
        n_gpus = torch.cuda.device_count()
        device = torch.device(f"cuda:{rank % n_gpus}")
        self.model = self.model.to(device).to(torch.bfloat16).eval()

        producer_topics = self.stage_config["producer_list"]
        self.producers = [
            Producer(
                producer_topic, 
                config["kafka_topics"][producer_topic]["partitions"]
            ) for producer_topic in producer_topics
        ]

        self.dataset = ImageDataset(consumer_topic, rank)
        batch_size = self.stage_config["batch_size"]
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=3)

        logger.info(f"Initialized model on device cuda:{rank % n_gpus}")
        logger.info("ScorerWorker initialization complete")

        self.counter = counter
    def run(self):
        logger.info("Starting scoring process")
        for idx, batch in enumerate(self.dataloader):
            # logger.info(f"Processing batch {idx} with {len(batch['document_ids'])} documents")
            try:
                images = batch["images"].to(self.model.device)
                with torch.no_grad():
                    scores = self.model(images).logits.to(torch.float32).cpu().numpy()
                # logger.debug(f"Successfully scored batch {idx}")
                for producer in self.producers:
                    producer.send(idx, {
                        "document_ids": batch["document_ids"],
                        "scores": scores
                    })
                with self.counter.get_lock():
                    self.counter.value += len(batch["document_ids"])
            except Exception as e:
                logger.error(f"Error processing batch {idx}: {e}")
        for producer in self.producers:
            producer.close()


def score_image(rank, config, counter):
    logger.info(f"Starting worker process with rank {rank}")
    worker = AestheticScorerWorker(config, rank, counter)
    worker.run()


if __name__ == "__main__":
    # Add this line before any other code
    mp.set_start_method('spawn', force=True)
    
    config = load_yaml_config("configs/example_config.yaml")
    stage_consumer = config["stages"][STAGE]["consumer"]
    N_PROCESSES = config["kafka_topics"][stage_consumer]["partitions"]

    counter = print_counter()

    mp.spawn(
        score_image,
        args=(config, counter),
        nprocs=N_PROCESSES,
        join=True
    )