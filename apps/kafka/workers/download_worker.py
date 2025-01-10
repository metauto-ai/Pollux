import base64
import io
import multiprocessing
from workers.kafka_utils import Consumer, Producer
from workers.common import load_yaml_config
from loguru import logger
from PIL import Image
import aiohttp
import asyncio
import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader


async def download_image(session, url):
    async with session.get(url) as response:
        image_data = await response.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((299, 299), resample=Image.Resampling.BILINEAR)
        
        # Convert to JPEG bytes using an in-memory buffer
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = base64.b64encode(buffer.getvalue())
        return image_bytes.decode('utf-8')

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
                yield {
                    "image_contents": img,
                    "document_ids": doc_id
                }


class DownloadWorker:
    STAGE = "downloader_worker"

    def __init__(self, config, rank):
        self.stage_config = config["stages"][self.STAGE]
        logger.info(f"Config: {self.stage_config}")
        
        consumer_topic = self.stage_config["consumer_config"]["topic"]

        producer_topic = self.stage_config["producer_config"]["topic"]
        topic_partitions = config["kafka_topics"][producer_topic]["partitions"]
        self.producer = Producer(producer_topic, topic_partitions)

        self.dataset = ImageUrlDataset(consumer_topic, rank)
        batch_size = self.stage_config["batch_size"]
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=1, pin_memory=True, prefetch_factor=3)


    def run(self):
        for idx, batch in enumerate(self.dataloader):
            logger.info(f"Processing batch of {len(batch['document_ids'])} documents")
            
            message = {
                "image_contents": [img for img in batch["image_contents"]],
                "document_ids": batch["document_ids"].tolist() if torch.is_tensor(batch["document_ids"]) else batch["document_ids"]
            }
            
            self.producer.send(idx, message)


def download_image_worker(rank, config):
    worker = DownloadWorker(config, rank)
    worker.run()


if __name__ == "__main__":
    config = load_yaml_config("configs/example_config.yaml")

    topic = config["stages"]["downloader_worker"]["consumer_config"]["topic"]
    N_PROCESSES = config["kafka_topics"][topic]["partitions"]

    mp.spawn(
        download_image_worker,
        args=(config, ),
        nprocs=N_PROCESSES,
        join=True
    )

    # with multiprocessing.Pool(N_PROCESSES) as pool:
    #     pool.starmap(download_image, [(i, config) for i in range(N_PROCESSES)])
