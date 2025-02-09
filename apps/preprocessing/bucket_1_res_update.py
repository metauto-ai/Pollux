# wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
# echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
# sudo apt update
# sudo apt install mongodb-database-tools
# mongoexport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@imagedata.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
# --db=world_model \
# --collection=bucket-256-1 \
# --out=/data/bucket-256-1.json --jsonArray
# mongoimport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
# --db=world_model \
# --collection=bucket-hq \
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
import mmap
import orjson
import ijson

file_path = "/data/bucket-256-1.json"


URI = "mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@imagedata.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
mongodb_client = MongoClient(URI)
db = mongodb_client["world_model"]
collection = db["bucket-256-1"]


def submit_doc(doc):
    if "width" in doc and "height" in doc:
        return 0
    else:
        try:
            update_data = {}
            response = requests.get(doc["media_path"])
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            width, height = image.size
            update_data["width"] = width
            update_data["height"] = height

            update_query = {"$set": update_data}
            filter_query = {"_id": ObjectId(doc["_id"]["$oid"])}
            collection.update_one(filter_query, update_query)
            return 1
        except Exception as e:
            print(f"Error processing element {doc['_id']}: {e}")
            return 0  # Return None or some default value in case of error


with open(file_path, "rb") as f:
    objects = ijson.items(f, "item")  # Stream each JSON object one-by-one

    with parallel_backend("threading"):
        Parallel(n_jobs=16)(
            delayed(submit_doc)(obj) for obj in tqdm(objects, desc="Processing")
        )
    # processed_results = [res for res in processed_results if res is not None]
    # with open(
    #     "/mnt/pollux/mongo_db_cache/pexel_images_processed.json", "w", encoding="utf-8"
    # ) as f:
    #     json.dump(processed_results, f, indent=4)


print("Processing finished")
