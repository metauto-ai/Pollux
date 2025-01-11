import multiprocessing
import threading
import time
from bson import ObjectId
from pymongo import UpdateOne
from workers.kafka_utils import Consumer
from workers.mongo_utils import PD12MDataset, DiffusionDataset, CC12MDataset
from workers.common import load_yaml_config, print_counter
import torch.multiprocessing as mp
from loguru import logger


STAGE = "update_database"


class UpdateWorker:

    def __init__(self, config, rank, counter):
        self.rank = rank
        self.stage_config = config["stages"][STAGE]
        logger.info(f"Config: {self.stage_config}")

        consumer_topic = self.stage_config["consumer"]
        self.consumer = Consumer(consumer_topic, partition_id=rank).consumer

        dataset_name = self.stage_config["dataset"]
        if dataset_name == "pd12m":
            self.dataset_cls = PD12MDataset
        elif dataset_name == "diffusion":
            self.dataset_cls = DiffusionDataset
        elif dataset_name == "cc12m":
            self.dataset_cls = CC12MDataset
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        self.dataset = self.dataset_cls()
        self.counter = counter
    def run(self):
        logger.info(f"Rank {self.rank}: Starting UpdateWorker")
        for message in self.consumer:
            message_data = message.value
            
            # logger.debug(f"Rank {self.rank}: Received message with {len(message_data['document_ids'])} documents")

            try:
                operations = [
                    UpdateOne(
                        {"_id": ObjectId(doc_id)},
                        {"$set": {"aesthetic_score": score}}
                    )
                    for doc_id, score in zip(message_data["document_ids"], message_data["scores"])
                ]

                if len(operations) == 0:
                    logger.warning(f"Rank {self.rank}: No operations to update, processed {len(message_data['document_ids'])}")
                    continue

                # logger.debug(f"Rank {self.rank}: Attempting to update {len(operations)} documents")
                result = self.dataset.bulkUpdate(operations)
                with self.counter.get_lock():
                    self.counter.value += result.modified_count
                # logger.info(
                #     f"Rank {self.rank}: Updated {result.modified_count}/{len(operations)} documents "
                #     f"(matched: {result.matched_count}, upserted: {result.upserted_count})"
                # )

                # if result.modified_count == 0:
                #     logger.warning(f"Rank {self.rank}: No documents got updated, processed {len(message_data['document_ids'])}")
                #     continue
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error updating database: {e}", exc_info=True)
                logger.info(f"Rank {self.rank}: Reconnecting to database...")
                self.dataset = self.dataset_cls()
                continue

def update_database(rank, config, counter):
    worker = UpdateWorker(config, rank, counter)
    worker.run()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    config = load_yaml_config("configs/example_config.yaml")

    consumer_topic = config["stages"][STAGE]["consumer"]
    N_PROCESSES = config["kafka_topics"][consumer_topic]["partitions"]

    counter = print_counter()

    mp.spawn(
        update_database,
        args=(config, counter),
        nprocs=N_PROCESSES,
        join=True
    )
