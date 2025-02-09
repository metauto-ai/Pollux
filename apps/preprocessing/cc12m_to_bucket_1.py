# wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
# echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
# sudo apt update
# sudo apt install mongodb-database-tools
# mongoexport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
# --db=world_model \
# --collection=cc12m \
# --out=/mnt/pollux/mongo_db_cache/cc12m.json --jsonArray
# mongoimport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@imagedata.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
# --db=world_model \
# --collection=bucket-256-1 \
# --file=/mnt/pollux/mongo_db_cache/pexel_images_processed.json --jsonArray

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

file_path = "/mnt/pollux/mongo_db_cache/cc12m.json"


def update_doc(doc):
    doc_return = {}
    if "caption" not in doc:
        return None
    if "width" not in doc:
        return None
    if "height" not in doc:
        return None
    if "partition_key" not in doc:
        doc_return["partition_key"] = random.randint(0, 10000)
    if "aesthetic_score" not in doc and doc["aesthetic_score"] < 5.5:
        return None

    try:
        for key, value in doc.items():
            if key == "_id":
                doc_return["source_id"] = value["$oid"]
            if key == "nova_lite_caption":
                doc_return["caption"] = value
            if key == "partition_key":
                doc_return["partition_key"] = value
            if key == "s3url":
                doc_return["media_path"] = value
            if key == "width":
                doc_return["width"] = value
            if key == "height":
                doc_return["height"] = value
        doc_return["source"] = "pexel_images"
        return doc_return
    except Exception as e:
        print(f"Error processing element {doc['_id']}: {e}")
        return None  # Return None or some default value in case of error


with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # Load JSON array
processed_results = []
for doc in data:
    doc = update_doc(doc)
    if doc is not None:
        processed_results.append(doc)

print(f"Processed {len(processed_results)} elements")
print(f"[:10] {processed_results[:10]}")
with open(
    "/mnt/pollux/mongo_db_cache/cc12m_processed.json", "w", encoding="utf-8"
) as f:
    json.dump(processed_results, f, indent=4)
print("Processing finished")
