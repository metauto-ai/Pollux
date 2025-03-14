# wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
# echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
# sudo apt update
# sudo apt install mongodb-database-tools
# mongoexport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@imagedata.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
#     --db=world_model \
#     --collection=diffusion1b_part_05_of_16_filtered \
#     --out=/mnt/pollux/mongo_db_cache/diffusion1b_part_05_of_16_filtered.json \
#     --jsonArray
# mongoimport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@imagedata.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
# --db=world_model \
# --collection=bucket-256-3 \
# --file=/mnt/pollux/mongo_db_cache/diffusion1b_part_03_of_16_filtered_processed.json --jsonArray
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

file_path = "/mnt/pollux/mongo_db_cache/diffusion1b_part_03_of_16_filtered.json"


def update_doc(doc):
    doc_return = {}
    if "partition_id" not in doc:
        doc_return["partition_key"] = random.randint(0, 10000)
    try:
        for key, value in doc.items():
            if key == "_id":
                doc_return["source_id"] = value
            if key == "prompt":
                doc_return["caption"] = value
            if key == "azure_url":
                doc_return["media_path"] = value
            if key == "width":
                doc_return["width"] = value
            if key == "height":
                doc_return["height"] = value
            if key == "partition_id":
                doc_return["partition_key"] = value
        doc_return["source"] = "diffusion1b_part_03_of_16_filtered"
        return doc_return
    except Exception as e:
        print(f"Error processing element {doc['_id']}: {e}")
        return None  # Return None or some default value in case of error


# URI = "mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
# mongodb_client = MongoClient(URI)
# db = mongodb_client["world_model"]
# collection = db["pexel_images"]


with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # Load JSON array
processed_results = []
for doc in tqdm(data):
    res_doc = update_doc(doc)
    if res_doc is not None:
        processed_results.append(res_doc)
    # processed_results = [res for res in processed_results if res is not None]
print(processed_results[:10])
with open(
    "/mnt/pollux/mongo_db_cache/diffusion1b_part_03_of_16_filtered_processed.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(processed_results, f, indent=4)
print("Processing finished")
