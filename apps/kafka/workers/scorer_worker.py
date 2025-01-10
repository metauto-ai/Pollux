import base64
import io
from workers.kafka_utils import Consumer, Producer
from workers.common import load_yaml_config
from loguru import logger
from PIL import Image

import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from transformers import SiglipImageProcessor


class ImageDataset(IterableDataset):
    def __init__(self, consumer_topic, rank):
        self.consumer_topic = consumer_topic
        self.rank = rank
        logger.info(f"Initialized ImageDataset for rank {rank}")

    def __iter__(self):
        consumer = Consumer(self.consumer_topic, partition_id=self.rank)
        preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        logger.info(f"Created consumer for topic {self.consumer_topic}")
        for message in consumer.consumer:
            logger.debug(f"Processing message with {len(message.value['image_contents'])} images")
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


class ScorerWorker:
    STAGE = "scorer_worker"

    def __init__(self, config, rank):
        self.stage_config = config["stages"][self.STAGE]
        logger.info(f"Config: {self.stage_config}")
        
        consumer_topic = self.stage_config["consumer_config"]["topic"]

        self.model, _ = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attention_2=True
        )
        
        # Move model to GPU
        n_gpus = torch.cuda.device_count()
        self.model = self.model.to(torch.bfloat16).to(f"cuda:{rank % n_gpus}").eval()

        producer_topic = self.stage_config["producer_config"]["topic"]
        topic_partitions = config["kafka_topics"][producer_topic]["partitions"]
        self.producer = Producer(producer_topic, topic_partitions)

        self.dataset = ImageDataset(consumer_topic, rank)
        batch_size = self.stage_config["batch_size"]
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=3)

        logger.info(f"Initialized model on device cuda:{rank % n_gpus}")
        logger.info("ScorerWorker initialization complete")


    def run(self):
        logger.info("Starting scoring process")
        for idx, batch in enumerate(self.dataloader):
            logger.info(f"Processing batch {idx} with {len(batch['document_ids'])} documents")
            try:
                images = batch["images"].to(self.model.device)
                with torch.no_grad():
                    scores = self.model(images).logits.to(torch.float32).cpu().numpy()
                logger.debug(f"Successfully scored batch {idx}")
                print ({
                    "document_ids": batch["document_ids"],
                    "scores": scores
                })
                self.producer.send(idx, {
                    "document_ids": batch["document_ids"],
                    "scores": scores
                })
            except Exception as e:
                logger.error(f"Error processing batch {idx}: {e}")


def score_image(rank):
    logger.info(f"Starting worker process with rank {rank}")
    config = load_yaml_config("configs/example_config.yaml")
    worker = ScorerWorker(config, rank)
    worker.run()


if __name__ == "__main__":
    N_PROCESSES = 1

    num_gpus = torch.cuda.device_count()

    mp.spawn(
        score_image,
        args=(),
        nprocs=N_PROCESSES,
        join=True
    )