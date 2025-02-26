import pandas as pd
from pymongo import MongoClient
from apps.main.utils.mongodb_data_load import MONGODB_URI

file_path = "/home/ubuntu/lingua_videogen/Pollux/apps/preprocessing/prompt_files/PartiPrompts.tsv"
df = pd.read_csv(file_path, sep="\t")
df.rename(columns={"Prompt": "caption"}, inplace=True)
data = df.to_dict(orient="records")
client = MongoClient(MONGODB_URI)  # Replace with your MongoDB URI
db = client["world_model"]  # Replace with your database name
collection = db["parti-prompts"]  # Collection name
collection.insert_many(data)

print("Upload complete!")
