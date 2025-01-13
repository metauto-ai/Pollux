from multiprocessing import Value
from workers.kafka_utils import Producer, create_kafka_partitions
from workers.mongo_utils import PD12MDataset, DiffusionDataset, CC12MDataset
from workers.common import load_yaml_config, print_counter
from torch.utils.data import DataLoader
from loguru import logger

STAGE = "read_database"


class MainWorker:
    def __init__(self, config, counter):
        create_kafka_partitions(config["kafka_topics"])

        self.stage_config = config["stages"][STAGE]
        logger.info(f"Config: {self.stage_config}")
        producer_topics = self.stage_config["producer_list"]
        self.producers = [
            Producer(
                producer_topic, 
                config["kafka_topics"][producer_topic]["partitions"]
            ) for producer_topic in producer_topics
        ]

        # Initialize MongoDB
        dataset_name = self.stage_config["dataset"]
        if dataset_name == "pd12m":
            self.dataset_cls = PD12MDataset
        elif dataset_name == "diffusion":
            self.dataset_cls = DiffusionDataset
        elif dataset_name == "cc12m":
            self.dataset_cls = CC12MDataset
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        self.num_shards = self.stage_config["num_shards"]
        self.shard_idx = self.stage_config["shard_idx"]

        self.dataloader = DataLoader(
            self.dataset_cls(self.num_shards, self.shard_idx), 
            batch_size=self.stage_config["batch_size"], 
            num_workers=0
        )

        self.counter = counter

    def run(self):
        while True:
            try:
                for idx, batch in enumerate(self.dataloader):
                    # logger.info(f"Processing batch of {len(batch['document_ids'])} documents")
                    for producer in self.producers:
                        producer.send(idx, batch)
                    with self.counter.get_lock():
                        self.counter.value += len(batch['document_ids'])
                
                # completed processing
                break
            except Exception as e:
                logger.error(f"Error in MainWorker: {e}")
                self.dataloader = DataLoader(
                    self.dataset_cls(), 
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
