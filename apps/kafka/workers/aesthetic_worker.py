import argparse
import asyncio
import base64
import io
import threading
import time
from typing import List, Optional
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


def worker_init_fn(worker_id, gpu_id, num_workers):
    """Initialize each worker process.
    
    Args:
        worker_id (int): ID of the DataLoader worker
        gpu_id (int): ID of the GPU this worker is associated with
        num_workers (int): Total number of workers in this DataLoader
    """
    # Set thread name for better logging
    thread_name = f'GPU{gpu_id}_Worker{worker_id}'
    threading.current_thread().name = thread_name
    
    # Initialize a new event loop for this worker
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    logger.info(f"Initialized {thread_name} with new event loop")
    logger.info(f"Worker {worker_id}/{num_workers-1} on GPU {gpu_id}")



class ImageDataset(IterableDataset):
    def __init__(self, gpu_id, consumer_topic, consumer_partitions):
        self.gpu_id = gpu_id
        self.consumer_topic = consumer_topic
        self.consumer_partitions = consumer_partitions

        logger.info(f"Initialized ImageDataset for gpu {gpu_id}")
        self.consumer = None
    
    async def preprocess_image(self, image_content, preprocessor):
        try:
            image_data = base64.b64decode(image_content)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            processed = preprocessor(images=image, return_tensors="pt").pixel_values.to(torch.bfloat16).squeeze(0)
            return processed
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    async def preprocess_images(self, image_contents, preprocessor):
        tasks = [self.preprocess_image(image_content, preprocessor) for image_content in image_contents]
        return await asyncio.gather(*tasks)

    def __iter__(self):
        ############################################################
        ###################### Consumer setup #####################
        # get worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or worker_info.num_workers <= 1:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # get number of partitions per worker
        num_partitions_per_worker = self.consumer_partitions // num_workers

        # get base partition for this GPU
        base_partition = self.gpu_id * self.consumer_partitions + worker_id * num_partitions_per_worker
        # Assign all partitions for this GPU to this worker
        partition_ids = [base_partition + i for i in range(num_partitions_per_worker)]
        
        # create consumer
        group_id = f"aesthetic_scoring_GPU{self.gpu_id}_Worker{worker_id}"
        self.consumer = Consumer(self.consumer_topic, group_id=group_id, partition_ids=partition_ids)
        logger.info(f"GPU {self.gpu_id}, Worker {worker_id}/{num_workers} assigned to partitions {partition_ids}")
        ############################################################

        ############################################################
        ##################### Preprocessing #######################
        preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

        last_commit_time = time.time()
        commit_interval = 10  # Commit every 10 seconds
        
        for message in self.consumer.consumer:
            message_data = message.value

            document_ids = message_data["document_ids"]
            # Use existing loop instead of creating new one
            processed = asyncio.run(
                self.preprocess_images(message_data["image_contents"], preprocessor)
            )
        
            # Return one image from the buffer
            for img, doc_id in zip(processed, document_ids):
                if img is not None: 
                    yield {
                        "images": img,
                        "document_ids": doc_id
                    }
            
            current_time = time.time()
            if current_time - last_commit_time > commit_interval:
                # Commit after processing all images in the message
                try:
                    self.consumer.consumer.commit()
                    last_commit_time = current_time
                except Exception as e:
                    logger.error(f"Error committing offset: {e}")
                

    def __del__(self):
        if self.consumer:
            self.consumer.consumer.close()


class AestheticScorerWorker:
    def __init__(self, 
                 gpu_id: int, 
                 batch_size: int,
                 consumer_topic: str, 
                 consumer_partitions_per_gpu: int, 
                 producers_topics: List[str], 
                 producers_partitions: List[int], 
                 counter):
        """
        Initialize the AestheticScorerWorker

        Args:
            gpu_id (int): The ID of the GPU to use
            consumer_topic (str): The topic to consume from
            consumer_partitions (int): The number of partitions to consume from
            producers_topics (List[str]): The topics to produce to
            producers_partitions (List[int]): The number of partitions each producer has
            counter: The counter to use
        """
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.consumer_topic = consumer_topic
        self.consumer_partitions_per_gpu = consumer_partitions_per_gpu
        self.producers_topics = producers_topics
        self.producers_partitions = producers_partitions
        self.counter = counter

        self.num_workers = 4

        # create producers
        self.producers = [
            Producer(
                producer_topic, 
                producer_partitions
            ) for producer_topic, producer_partitions in zip(self.producers_topics, self.producers_partitions)
        ]

        # model init
        n_gpus = torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id % n_gpus}")
        self.model, _ = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attention_2=True
        )
        # Move model to device and convert to bfloat16 after initialization
        self.model = self.model.to(device).to(torch.bfloat16).eval()

        self.dataset = ImageDataset(gpu_id, consumer_topic, consumer_partitions_per_gpu)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, prefetch_factor=4)

        logger.info(f"GPU {self.gpu_id} initialized model on device cuda:{gpu_id % n_gpus}")
        logger.info(f"GPU {self.gpu_id} ScorerWorker initialization complete")


    def run(self):
        logger.info(f"GPU {self.gpu_id} Starting scoring process")
        for idx, batch in enumerate(self.dataloader):
            logger.info(f"GPU {self.gpu_id} Processing batch {idx} with {len(batch['document_ids'])} documents")
            try:
                images = batch["images"].to(self.model.device)
                with torch.no_grad():
                    scores = self.model(images).logits.squeeze(1).to(torch.float32).cpu().numpy()
                logger.debug(f"GPU {self.gpu_id} Successfully scored batch {idx}")
                for producer in self.producers:
                    producer.send(idx, {
                        "document_ids": batch["document_ids"],
                        "scores": scores
                    })
                with self.counter.get_lock():
                    self.counter.value += len(batch["document_ids"])
            except Exception as e:
                logger.error(f"GPU {self.gpu_id} Error processing batch {idx}: {e}")
        
    def __del__(self):
        if self.dataloader is not None:
            self.dataloader._iterator = None  # Force cleanup of iterator
            del self.dataloader
        if self.dataset is not None:
            del self.dataset
        if self.producers is not None:
            for producer in self.producers:
                producer.close()


def score_image(rank, batch_size, consumer_topic, consumer_partitions_per_gpu, producers_topics, producers_partitions, counter):
    logger.info(f"GPU {rank} Starting worker process")
    worker = AestheticScorerWorker(rank, batch_size, consumer_topic, consumer_partitions_per_gpu, producers_topics, producers_partitions, counter)
    worker.run()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/diffusion_config.yaml", help='Path to config file')
    args = parser.parse_args()
    
    config = load_yaml_config(args.config)
    stage_config = config["stages"][STAGE]
    logger.info(f"Config: {stage_config}")
    consumer_topic = stage_config["consumer"]
    consumer_partitions = config["kafka_topics"][consumer_topic]["partitions"]
    producers_topics = stage_config["producer_list"]
    producers_partitions = [config["kafka_topics"][producer_topic]["partitions"] for producer_topic in producers_topics]
    batch_size = stage_config["batch_size"]
    
    N_GPUs = torch.cuda.device_count()
    consumer_partitions_per_gpu = consumer_partitions // N_GPUs

    if consumer_partitions % N_GPUs != 0:
        raise ValueError("Total consumer partitions must be divisible by the number of GPUs")
    
    counter = print_counter()

    logger.info(f"Total consumer partitions: {consumer_partitions}")
    logger.info(f"Number of GPUs: {N_GPUs}")
    logger.info(f"Partitions per GPU: {consumer_partitions_per_gpu}")
    
    # Verify partition distribution
    for gpu_id in range(N_GPUs):
        start_partition = gpu_id * consumer_partitions_per_gpu
        end_partition = start_partition + consumer_partitions_per_gpu
        logger.info(f"GPU {gpu_id} will handle partitions {start_partition} to {end_partition-1}")

    mp.spawn(
        score_image,
        args=(batch_size, consumer_topic, consumer_partitions_per_gpu, producers_topics, producers_partitions, counter),
        nprocs=N_GPUs,
        join=True
    )

    logger.info("ScorerWorker process completed")