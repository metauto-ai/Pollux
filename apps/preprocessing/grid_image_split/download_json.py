import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient
import json

load_dotenv()

# MongoDB connection URI (see `.env.sample`)
mongodb_host = os.environ["MONGODB_URI"]
mongodb_user = quote_plus(os.environ["MONGODB_USER"])
mongodb_password = quote_plus(os.environ["MONGODB_PASSWORD"])
mongo_uri = f"mongodb+srv://{mongodb_user}:{mongodb_password}@{mongodb_host}"

# Connect to MongoDB
client = MongoClient(mongo_uri)

# Select database and collection
db = client["world_model"]
collection = db["midjourney_discord-1"]

# Query all documents (adjust if needed)
cursor = collection.find({})

# File path to save the exported JSON data
output_file = "/jfs/jinjie/code/downloads/temp_data/midjourney_discord-1.json"

# Write data to JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(list(cursor), f, indent=4, ensure_ascii=False)

print(f"Export completed: {output_file}")
