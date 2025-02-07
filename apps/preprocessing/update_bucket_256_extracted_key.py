"""
mongoexport  --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@imagedata.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
--db=world_model \
--collection=bucket-256-1 \
--out=/mnt/pollux/mongo_db_cache/bucket-256-1.json --jsonArray

mongoimport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@imagedata.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
--db=world_model \
--collection=bucket-256-1 \
--file=/mnt/pollux/mongo_db_cache/bucket-256-1_processed.json --jsonArray
"""

import json
import requests
from PIL import Image
import io
from joblib import Parallel, delayed, parallel_backend
import random
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from pymongo import MongoClient
from bson import ObjectId

file_path = "/mnt/pollux/mongo_db_cache/bucket-256-1.json"


def update_doc(doc):
    doc_return = doc
    doc_return["parquet_cache"] = True
    return doc_return


# URI = "mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
# mongodb_client = MongoClient(URI)
# db = mongodb_client["world_model"]
# collection = db["pexel_images"]


with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # Load JSON array
process_res = []
for doc in tqdm(data):
    doc = update_doc(doc)
    if doc != None:
        process_res.append(doc)
    # processed_results = [res for res in processed_results if res is not None]
with open(
    "/mnt/pollux/mongo_db_cache/bucket-256-1_processed.json", "w", encoding="utf-8"
) as f:
    json.dump(process_res, f, indent=4)
print("Processing finished")
