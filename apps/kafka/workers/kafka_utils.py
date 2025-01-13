import time
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from confluent_kafka.admin import AdminClient, NewPartitions, NewTopic
from loguru import logger
import numpy as np
import json

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
    def __init__(self, topic, group_id=None, partition_id=0):
        self.group_id = group_id
        self.partition_id = partition_id
        self.consumer = self._create_consumer()
        if topic:
            logger.info(f"Assigning partition {self.partition_id} to topic {topic}")
            self.consumer.assign([TopicPartition(topic, self.partition_id)])
    
    def json_decode(self, x):
        return json.loads(x.decode('utf-8'))

    def _create_consumer(self):
        return KafkaConsumer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            api_version=(2, 0, 2),
            group_id=self.group_id,
            auto_offset_reset='latest',
            enable_auto_commit=False,
            max_poll_records=100,
            max_partition_fetch_bytes=524288000,  # 30MB
            fetch_max_bytes=524288000,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_interval_ms=300000,
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
            api_version=(2, 0, 2),
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
    admin_client.create_topics([
        NewTopic(topic, config["partitions"], replication_factor=1) for topic, config in kafka_topics.items()
    ])
    logger.info(f"Created Kafka topics: {list(kafka_topics.keys())}")

    topic_partitions = [
        NewPartitions(topic, config["partitions"])
        for topic, config in kafka_topics.items()
    ]
    admin_client.create_partitions(topic_partitions)

    # Wait for topics to be created
    time.sleep(10)
