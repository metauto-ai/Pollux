import base64
import io
import multiprocessing
import threading
import time
from workers.kafka_utils import Consumer, Producer
from workers.common import load_yaml_config, print_counter
from loguru import logger
from PIL import Image
import aiohttp
import asyncio
import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader


async def download_image(session, url):
    try:
        async with session.get(url) as response:
            image_data = await response.read()
            image_bytes = base64.b64encode(image_data)
            return image_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None

async def download_images(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url) for url in urls]
        return await asyncio.gather(*tasks)


class ImageUrlDataset(IterableDataset):
    def __init__(self, consumer_topic, rank):
        self.consumer_topic = consumer_topic
        self.rank = rank

    def __iter__(self):
        consumer = Consumer(self.consumer_topic, partition_id=self.rank)
        for message in consumer.consumer:
            message_data = message.value

            image_contents = asyncio.run(download_images(message_data["image_urls"]))

            for img, doc_id in zip(image_contents, message_data["document_ids"]):
                if img is None:
                    logger.warning(f"Skipping document {doc_id} due to image download error")
                    continue
                yield {
                    "image_contents": img,
                    "document_ids": doc_id
                }


class DownloadWorker:
    STAGE = "download_images"

    def __init__(self, config, rank, counter):
        self.stage_config = config["stages"][self.STAGE]
        logger.info(f"Config: {self.stage_config}")
        
        consumer_topic = self.stage_config["consumer"]

        producer_topics = self.stage_config["producer_list"]
        self.producers = [
            Producer(
                producer_topic, 
                config["kafka_topics"][producer_topic]["partitions"]
            ) for producer_topic in producer_topics
        ]

        self.dataset = ImageUrlDataset(consumer_topic, rank)
        batch_size = self.stage_config["batch_size"]
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=3)
        
        self.counter = counter

    def run(self):
        for idx, batch in enumerate(self.dataloader):
            # logger.info(f"Processing batch {idx} of {len(batch['document_ids'])} documents")
            with self.counter.get_lock():
                self.counter.value += len(batch['document_ids'])
            
            message = {
                "image_contents": [img for img in batch["image_contents"]],
                "document_ids": batch["document_ids"].tolist() if torch.is_tensor(batch["document_ids"]) else batch["document_ids"]
            }
            total_size = sum(len(img) for img in message["image_contents"])
            logger.info(f"Batch {idx} total image size: {total_size/1024/1024:.2f} MB")
            
            for producer in self.producers:
                producer.send(idx, message)


def download_image_worker(rank, config, counter):
    worker = DownloadWorker(config, rank, counter)
    worker.run()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    config = load_yaml_config("configs/example_config.yaml")

    topic = config["stages"]["download_images"]["consumer"]
    N_PROCESSES = config["kafka_topics"][topic]["partitions"]

    counter = print_counter()

    mp.spawn(
        download_image_worker,
        args=(config, counter),
        nprocs=N_PROCESSES,
        join=True
    )

    # with multiprocessing.Pool(N_PROCESSES) as pool:
    #     pool.starmap(download_image, [(i, config) for i in range(N_PROCESSES)])
