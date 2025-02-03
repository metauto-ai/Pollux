from multiprocessing import Value
import time
from workers.kafka_utils import Producer, create_kafka_partitions
from workers.mongo_utils import MongoDataset
from workers.common import load_yaml_config, print_counter
from torch.utils.data import DataLoader
from loguru import logger

STAGE = "read_database"


class MainWorker:
    def __init__(self, config, counter):
        create_kafka_partitions(config["kafka_topics"])

        self.mongo_config = config["mongo_config"]
        mongo_dataset = MongoDataset(self.mongo_config)

        self.stage_config = config["stages"][STAGE]
        logger.info(f"Config: {self.stage_config}")
        producer_topic = self.stage_config["producer"]
        self.producer = Producer(
                producer_topic, 
                config["kafka_topics"][producer_topic]["partitions"]
            )

        self.dataloader = DataLoader(
            mongo_dataset, 
            batch_size=self.stage_config["batch_size"], 
            num_workers=0
        )

        self.counter = counter

    def run(self):
        while True:
            try:
                for idx, batch in enumerate(self.dataloader):
                    # logger.info(f"Processing batch of {len(batch['document_ids'])} documents")
                    for idx in range(len(batch['doc_id'])):
                        self.producer.send(idx, {
                            "doc_id": batch['doc_id'][idx],
                            "image_url": batch['image_url'][idx],
                            "caption": batch['caption'][idx],
                        })
                    with self.counter.get_lock():
                        self.counter.value += len(batch['doc_id'])
                
                # completed processing
                # break
            except Exception as e:
                logger.error(f"Error in MainWorker: {e}")
            
            self.dataloader = DataLoader(
                MongoDataset(self.mongo_config), 
                batch_size=self.stage_config["batch_size"], 
                num_workers=0 
            )

            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/diffusion_config.yaml", help='Path to config file')
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    counter = print_counter()

    worker = MainWorker(config, counter)
    worker.run()
