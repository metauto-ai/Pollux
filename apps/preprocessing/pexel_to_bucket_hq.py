# wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
# echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
# sudo apt update
# sudo apt install mongodb-database-tools
# mongoexport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
# --db=world_model \
# --collection=pexel_images \
# --out=/mnt/pollux/mongo_db_cache/pexel_images.json --jsonArray
# mongoimport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
# --db=world_model \
# --collection=bucket-hq \
# --file=/mnt/pollux/mongo_db_cache/pexel_images_processed.json --jsonArray

import json
import requests
from PIL import Image
import io
from joblib import Parallel, delayed
import random
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

file_path = "/mnt/pollux/mongo_db_cache/pexel_images.json"


def update_doc(doc):
    doc_return = {}
    if "nova_lite_caption" not in doc:
        return None
    if "partition_key" not in doc:
        doc_return["partition_key"] = random.randint(0, 10000)
    try:
        for key, value in doc.items():
            if key == "_id":
                doc_return["source_id"] = value["$oid"]
            if key == "nova_lite_caption":
                doc_return["caption"] = value
            if key == "partition_key":
                doc_return["partition_key"] = value
            if key == "url":
                doc_return["media"] = value
                response = requests.get(value)
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                width, height = image.size
                doc_return["width"] = width
                doc_return["height"] = height
        doc_return["source"] = "pexel_images"
        return doc_return
    except Exception as e:
        print(f"Error processing element {doc['_id']}: {e}")
        return None  # Return None or some default value in case of error


with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # Load JSON array
    with tqdm_joblib(tqdm(desc="Processing", total=len(data))):
        processed_results = Parallel(n_jobs=64)(delayed(update_doc)(el) for el in data)
    processed_results = [res for res in processed_results if res is not None]
    with open(
        "/mnt/pollux/mongo_db_cache/pexel_images_processed.json", "w", encoding="utf-8"
    ) as f:
        json.dump(processed_results, f, indent=4)
print("Processing finished")
