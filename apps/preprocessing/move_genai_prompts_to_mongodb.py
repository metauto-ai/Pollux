import pandas as pd
from pymongo import MongoClient
from apps.main.utils.mongodb_data_load import MONGODB_URI
import json

file_path = "/home/ubuntu/lingua_videogen/Pollux/apps/preprocessing/prompt_files/genai_image.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
client = MongoClient(MONGODB_URI)  # Replace with your MongoDB URI
db = client["world_model"]  # Replace with your database name
collection = db["GenAI-Bench-1600"]  # Collection name
# Transform and insert data into MongoDB
for key, value in data.items():
    doc = {
        "index": value["id"],
        "caption": value["prompt"],
        "caption_chinese": value["prompt in Chinese"],
    }
    collection.insert_one(doc)

# df = pd.read_csv(file_path, sep="\t")
# df.rename(columns={"Prompt": "caption"}, inplace=True)
# data = df.to_dict(orient="records")
# client = MongoClient(MONGODB_URI)  # Replace with your MongoDB URI
# db = client["world_model"]  # Replace with your database name
# collection = db["parti-prompts"]  # Collection name
# collection.insert_many(data)

print("Upload complete!")
