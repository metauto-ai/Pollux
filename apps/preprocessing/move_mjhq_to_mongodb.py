import pandas as pd
from pymongo import MongoClient
from apps.main.utils.mongodb_data_load import MONGODB_URI
import json

json_file_path = "/home/ubuntu/evaluation_metrics/MJHQ_30K/meta_data.json"
with open(json_file_path, "r") as file:
    data = json.load(file)
records = [
    {"caption": value["prompt"], "category": value["category"]}
    for key, value in data.items()
]
client = MongoClient(MONGODB_URI)  # Replace with your MongoDB URI
db = client["world_model"]  # Replace with your database name
collection = db["MJHQ-30k"]  # Collection name
collection.insert_many(records)
# print(records)
print("Upload complete!")
