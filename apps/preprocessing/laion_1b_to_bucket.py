# wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
# echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
# sudo apt update
# sudo apt install mongodb-database-tools
# mongoexport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
# --db=world_model \
# --collection=LAION1B-02 \
# --out=/mnt/pollux/mongo_db_cache/LAION1B-02.json --jsonArray

# mongoimport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@imagedata.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
# --db=world_model \
# --collection=bucket-256-8 \
# --file=/mnt/pollux/mongo_db_cache/LAION1B-02_processed.json --jsonArray
# python -m apps.preprocessing.laion_1b_to_bucket
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
import ijson

file_path = "/mnt/pollux/mongo_db_cache/LAION1B-02.json"


def update_doc(doc):
    doc_return = {}
    if "partition_id" not in doc:
        doc_return["partition_key"] = random.randint(0, 10000)
    if doc["aesthetic_score"] < 5.0:
        return None
    try:
        doc_return["source_id"] = doc["_id"]["$oid"]
        doc_return["source"] = "LAION1B-02"
        doc_return["caption"] = doc["text"]
        doc_return["media_path"] = doc["azure_url"]
        doc_return["width"] = int(doc["width"])
        doc_return["height"] = int(doc["height"])
        return doc_return
    except Exception as e:
        print(f"Error processing element {doc['_id']}: {e}")
        return None  # Return None or some default value in case of error


# URI = "mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
# mongodb_client = MongoClient(URI)
# db = mongodb_client["world_model"]
# collection = db["pexel_images"]


processed_results = []

with open(file_path, "r") as file:
    for item in tqdm(ijson.items(file, "item")):
        res_doc = update_doc(item)
        if res_doc != None:
            processed_results.append(res_doc)
    # processed_results = [res for res in processed_results if res is not None]
print(processed_results[:10])
with open(
    "/mnt/pollux/mongo_db_cache/LAION1B-02_processed.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(processed_results, f, indent=4)
print("Processing finished")
