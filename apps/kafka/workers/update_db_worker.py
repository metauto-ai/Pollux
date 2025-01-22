import multiprocessing
import threading
import time
from bson import ObjectId
from pymongo import UpdateOne
from workers.kafka_utils import Consumer
from workers.mongo_utils import MongoDataset
from workers.common import load_yaml_config, print_counter, init_wandb
import torch.multiprocessing as mp
from loguru import logger


STAGE = "update_database"


class UpdateWorker:

    def __init__(self, config, rank, counter):
        self.rank = rank
        self.stage_config = config["stages"][STAGE]
        logger.info(f"Config: {self.stage_config}")

        consumer_topic = self.stage_config["consumer"]
        self.consumer = Consumer(consumer_topic, partition_ids=[rank]).consumer

        self.mongo_config = config["mongo_config"]
        self.dataset = MongoDataset(self.mongo_config)

        self.dataset.set_collection()
        self.counter = counter

    def run(self):
        logger.info(f"Rank {self.rank}: Starting UpdateWorker")
        for message in self.consumer:
            message_data = message.value
            
            doc_ids = message_data.pop("doc_ids")

            # logger.debug(f"Rank {self.rank}: Received message with {len(doc_ids)} documents")

            updates = []
            for idx in range(len(doc_ids)):
                update = {}
                for key in message_data:
                    update[key] = message_data[key][idx]
                updates.append(update)

            try:
                operations = [
                    UpdateOne(
                        {"_id": ObjectId(doc_id)},
                        {"$set": update}
                    )
                    for doc_id, update in zip(doc_ids, updates)
                ]

                if len(operations) == 0:
                    logger.warning(f"Rank {self.rank}: No operations to update, processed {len(message_data['document_ids'])}")
                    continue

                # logger.debug(f"Rank {self.rank}: Attempting to update {len(operations)} documents")
                result = self.dataset.bulkUpdate(operations)
                with self.counter.get_lock():
                    self.counter.value += result.modified_count
                logger.info(
                    f"Rank {self.rank}: Updated {result.modified_count}/{len(operations)} documents "
                    f"(matched: {result.matched_count}, upserted: {result.upserted_count})"
                )

                # if result.modified_count == 0:
                #     logger.warning(f"Rank {self.rank}: No documents got updated, processed {len(message_data['document_ids'])}")
                #     continue
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error updating database: {e}", exc_info=True)
                logger.info(f"Rank {self.rank}: Reconnecting to database...")
                self.dataset = MongoDataset(self.mongo_config)
                self.dataset.set_collection()
                continue

def update_database(rank, config, counter):
    worker = UpdateWorker(config, rank, counter)
    worker.run()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/diffusion_config.yaml", help='Path to config file')
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    consumer_topic = config["stages"][STAGE]["consumer"]
    N_PROCESSES = config["kafka_topics"][consumer_topic]["partitions"]

    dataset_name = config["mongo_config"]["collection_name"]

    # Initialize wandb
    init_wandb(config, run_name=dataset_name)
    
    counter = print_counter()

    mp.spawn(
        update_database,
        args=(config, counter),
        nprocs=N_PROCESSES,
        join=True
    )
