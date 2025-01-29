import time
from typing import List, Union
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from confluent_kafka.admin import AdminClient, NewPartitions, NewTopic
from loguru import logger
import numpy as np
import json
from kafka.client_async import KafkaClient

BOOTSTRAP_SERVERS = ['localhost:9092']

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

class Consumer:
    def __init__(self, topic, group_id=None, partition_ids: List[int]=None):
        self.group_id = group_id
        self.partition_ids = partition_ids
        self.consumer = self._create_consumer()
        
        logger.info(f"Assigning partition {self.partition_ids} to topic {topic}")
        if self.partition_ids:
            partitions = [TopicPartition(topic, partition_id) for partition_id in self.partition_ids]
            self.consumer.assign(partitions)
    
    def json_decode(self, x):
        return json.loads(x.decode('utf-8'))

    def _create_consumer(self):
        return KafkaConsumer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            api_version=get_kafka_api_version(),
            group_id=self.group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            max_poll_records=100,
            max_partition_fetch_bytes=524288000,  # 30MB
            fetch_max_bytes=524288000,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_interval_ms=10000,
            request_timeout_ms=305000,
            value_deserializer=self.json_decode
        )


class Producer:
    def __init__(self, topic, partitions):
        self.topic = topic
        self.partitions = partitions
        self.producer = self._create_producer()
    
    def _create_producer(self):
        return KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            api_version=get_kafka_api_version(),
            acks='all',
            retries=3,
            compression_type='gzip',
            max_request_size=67108864,
            batch_size=67108864,
            linger_ms=100,
            buffer_memory=536870912,
            max_block_ms=30000,
            value_serializer=lambda x: json.dumps(x, cls=NumpyEncoder).encode('utf-8')
        )
        
    def send(self, id, value):
        self.producer.send(self.topic, partition=id % self.partitions, value=value).add_errback(self._on_send_error)

    def _on_send_error(self, excp):
        logger.error(f'Process {self.partition_id}: Error sending message :', exc_info=excp)

    def close(self):
        self.producer.flush()
        self.producer.close()


admin_client = AdminClient({"bootstrap.servers" : BOOTSTRAP_SERVERS[0]})

def create_kafka_partitions(kafka_topics):
    logger.info(f"Creating Kafka topics with configurations: {kafka_topics}")
    # Check if topics already exist with correct partition count
    existing_topics = admin_client.list_topics().topics
    
    topics_to_create = {}
    for topic, config in kafka_topics.items():
        if topic not in existing_topics:
            topics_to_create[topic] = config
        else:
            # Check if existing topic has enough partitions
            topic_info = existing_topics[topic]
            if len(topic_info.partitions) >= config["partitions"]:
                logger.info(f"Topic {topic} already exists with {len(topic_info.partitions)} partitions")
                continue
            topics_to_create[topic] = config
    
    if not topics_to_create:
        logger.info("All required topics already exist with sufficient partitions")
        return
    
    admin_client.create_topics([
        NewTopic(topic, config["partitions"], replication_factor=1) for topic, config in topics_to_create.items()
    ])
    logger.info(f"Created Kafka topics: {list(topics_to_create.keys())}")

    topic_partitions = [
        NewPartitions(topic, config["partitions"])
        for topic, config in topics_to_create.items()
    ]
    admin_client.create_partitions(topic_partitions)

    # Wait for topics to be created
    time.sleep(10)

def get_kafka_api_version():
    client = KafkaClient(bootstrap_servers=BOOTSTRAP_SERVERS)
    versions = client.check_version()
    client.close()
    return versions
