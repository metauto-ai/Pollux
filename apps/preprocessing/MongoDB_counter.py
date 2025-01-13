from pymongo import MongoClient
from PIL import Image
import certifi
from apps.main.utils.mongodb_data_load import MONGODB_URI
import logging
import copy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
from concurrent.futures import ThreadPoolExecutor, as_completed


def count_doc_multi_proc(query, collection_name, num_workers, partition_key):
    def run(query):
        mongodb_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
        db = mongodb_client["world_model"]
        collection = db[collection_name]
        return collection.count_documents(query, maxTimeMS=6000000)

    queries = []
    for worker_id in range(num_workers):
        partition_query = {
            "$expr": {
                "$eq": [
                    {
                        "$mod": [
                            {"$toInt": f"${partition_key}"},
                            num_workers,  # Total number of shards
                        ]
                    },
                    worker_id,  # Current shard index
                ]
            }
        }
        partition_query.update(query)
        queries.append(partition_query)
    res = 0
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        future_to_query = {executor.submit(run, query): query for query in queries}

        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                res += future.result()
            except Exception as e:
                logging.warning(f"Erros in handling {query}:{e}")
    return res


num = count_doc_multi_proc(
    num_workers=64,
    partition_key="key",
    collection_name="cc12m",
    query={"aesthetic_score": {"$gt": 4.5}},
)
logging.info(f"Number of images with aesthetic score > 4.5: {num}")

num = count_doc_multi_proc(
    num_workers=64,
    partition_key="key",
    collection_name="cc12m",
    query={"aesthetic_score": {"$gt": 5.0}},
)
logging.info(f"Number of images with aesthetic score > 5.0: {num}")

num = count_doc_multi_proc(
    num_workers=64,
    partition_key="key",
    collection_name="cc12m",
    query={"aesthetic_score": {"$gt": 5.5}},
)
logging.info(f"Number of images with aesthetic score > 5.5: {num}")

num = count_doc_multi_proc(
    num_workers=64,
    partition_key="key",
    collection_name="cc12m",
    query={"aesthetic_score": {"$gt": 6.0}},
)
logging.info(f"Number of images with aesthetic score > 6.0: {num}")

num = count_doc_multi_proc(
    num_workers=64,
    partition_key="key",
    collection_name="cc12m",
    query={"aesthetic_score": {"$gt": 6.5}},
)
logging.info(f"Number of images with aesthetic score > 6.5: {num}")
