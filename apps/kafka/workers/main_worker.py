from workers.kafka_utils import Producer, create_kafka_partitions
from workers.mongo_utils import PD12MDataset
from workers.common import load_yaml_config
from torch.utils.data import DataLoader
from loguru import logger

class MainWorker:
    STAGE = "main_worker"

    def __init__(self, config):
        create_kafka_partitions(config["kafka_topics"])

        self.stage_config = config["stages"][self.STAGE]
        logger.info(f"Config: {self.stage_config}")
        producer_topic = self.stage_config["producer_config"]["topic"]
        topic_partitions = config["kafka_topics"][producer_topic]["partitions"]
        self.producer = Producer(producer_topic, topic_partitions)

        # Initialize MongoDB
        dataset = PD12MDataset()
        self.dataloader = DataLoader(dataset, batch_size=6, num_workers=1)

    def run(self):
        for idx, batch in enumerate(self.dataloader):
            logger.info(f"Processing batch of {len(batch['document_ids'])} documents")
            self.producer.send(idx, batch)


if __name__ == "__main__":
    config = load_yaml_config("configs/example_config.yaml")
    worker = MainWorker(config)
    worker.run()
