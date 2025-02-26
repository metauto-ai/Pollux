import os
from pymongo import MongoClient
from apps.main.utils.mongodb_data_load import MONGODB_URI

# Define the folder path
folder_path = "/home/ubuntu/evaluation_metrics/DPG_Bench/dpg_bench/prompts/"

# Connect to MongoDB
client = MongoClient(MONGODB_URI)  # Replace with your MongoDB URI
db = client["world_model"]  # Replace with your database name
collection = db["dpg-bench"]  # Collection name

# List to store all captions
captions = []

# Iterate through all text files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Process only .txt files
        file_path = os.path.join(folder_path, filename)

        # Read the content of the file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()  # Read and remove leading/trailing whitespace

            # Add caption to the list as a dictionary
            captions.append({"caption": text, "file_name": filename, "gen_id": 0})
            captions.append({"caption": text, "file_name": filename, "gen_id": 1})
            captions.append({"caption": text, "file_name": filename, "gen_id": 2})
            captions.append({"caption": text, "file_name": filename, "gen_id": 3})
# print(captions)
# # Insert all captions into MongoDB
if captions:
    collection.insert_many(captions)
    print(f"Uploaded {len(captions)} captions to MongoDB.")
else:
    print("No text files found.")
