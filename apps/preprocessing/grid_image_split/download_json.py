from pymongo import MongoClient
import json

# MongoDB connection URI
mongo_uri = "mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"

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
