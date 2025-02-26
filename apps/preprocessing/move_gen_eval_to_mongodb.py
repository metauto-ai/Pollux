import os
from pymongo import MongoClient
from apps.main.utils.mongodb_data_load import MONGODB_URI
import json
import copy

# Define the folder path
file_path = (
    "/home/ubuntu/evaluation_metrics/GenEval_Bench/prompts/evaluation_metadata.jsonl"
)

# Connect to MongoDB
client = MongoClient(MONGODB_URI)  # Replace with your MongoDB URI
db = client["world_model"]  # Replace with your database name
collection = db["gen-eval"]  # Collection name

# List to store all captions
captions = []


documents = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        document = json.loads(line.strip())  # Parse each line as JSON
        if "prompt" in document:
            document["caption"] = document.pop("prompt")
        document["gen_id"] = 0
        documents.append(document)
        document = copy.deepcopy(document)
        document["gen_id"] = 1
        documents.append(document)
        document = copy.deepcopy(document)
        document["gen_id"] = 2
        documents.append(document)
        document = copy.deepcopy(document)
        document["gen_id"] = 3
        documents.append(document)
        # captions.append({"caption": text, "file_name": filename, "gen_id": 0})
        # captions.append({"caption": text, "file_name": filename, "gen_id": 1})
        # captions.append({"caption": text, "file_name": filename, "gen_id": 2})
        # captions.append({"caption": text, "file_name": filename, "gen_id": 3})
# print(captions)
# # Insert all captions into MongoDB

collection.insert_many(documents)
print(f"Uploaded {len(documents)} captions to MongoDB.")
