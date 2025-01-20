import os
from pymongo import MongoClient
from tqdm import tqdm  # Optional, for showing progress
from apps.main.utils.mongodb_data_load import MONGODB_URI

# MongoDB setup
client = MongoClient(MONGODB_URI)
db = client["world_model"]
collection = db["pexel_images_jfs_nova"]

# Directory containing images
directory_path = "/jfs/pexelImages"


def collect_images(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in tqdm(files):  # tqdm is optional, for progress bar
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):  # Filter for image files
                yield {
                    "jfs_path": os.path.join(root, file),
                }


def upload_to_mongodb(image_data):
    collection.insert_many(image_data)


# Collect and upload in chunks to manage memory usage
chunk_size = 10000  # Adjust chunk size based on your memory capacity
image_chunk = []
for image_info in collect_images(directory_path):
    image_chunk.append(image_info)
    if len(image_chunk) >= chunk_size:
        upload_to_mongodb(image_chunk)
        image_chunk = []
# Upload any remaining images
if image_chunk:
    upload_to_mongodb(image_chunk)

print("Upload completed.")
