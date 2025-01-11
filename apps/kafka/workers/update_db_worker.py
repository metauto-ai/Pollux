from bson import ObjectId
from pymongo import UpdateOne
from workers.kafka_utils import Consumer
from workers.mongo_utils import PD12MDataset
from workers.common import load_yaml_config
import torch.multiprocessing as mp
from loguru import logger

class UpdateWorker:
    STAGE = "update_database"

    def __init__(self, config, rank):
        self.rank = rank
        self.stage_config = config["stages"][self.STAGE]
        logger.info(f"Config: {self.stage_config}")

        consumer_topic = self.stage_config["consumer_config"]["topic"]
        self.consumer = Consumer(consumer_topic, partition_id=rank).consumer

        self.dataset = PD12MDataset()

    def run(self):
        logger.info(f"Rank {self.rank}: Starting UpdateWorker")
        for message in self.consumer:
            message_data = message.value
            
            logger.debug(f"Rank {self.rank}: Received message with {len(message_data['document_ids'])} documents")

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

                logger.debug(f"Rank {self.rank}: Attempting to update {len(operations)} documents")
                result = self.dataset.bulkUpdate(operations)

                logger.info(
                    f"Rank {self.rank}: Updated {result.modified_count}/{len(operations)} documents "
                    f"(matched: {result.matched_count}, upserted: {result.upserted_count})"
                )

                if result.modified_count == 0:
                    logger.warning(f"Rank {self.rank}: No documents got updated, processed {len(message_data['document_ids'])}")
                    continue
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error updating database: {e}", exc_info=True)
                logger.info(f"Rank {self.rank}: Reconnecting to database...")
                self.dataset = PD12MDataset()
                continue

def update_database(rank, config):
    worker = UpdateWorker(config, rank)
    worker.run()

if __name__ == "__main__":
    config = load_yaml_config("configs/example_config.yaml")

    consumer_topic = config["stages"]["updater_worker"]["consumer_config"]["topic"]
    N_PROCESSES = config["kafka_topics"][consumer_topic]["partitions"]

    mp.spawn(
        update_database,
        args=(config, ),
        nprocs=N_PROCESSES,
        join=True
    )