"""_summary_
python -m apps.offline_inf.submit --dump-dir /mnt/pollux/test/save_parquet --collection-name bucket-256-parquet-test
"""

import argparse
import logging
import glob
from pathlib import Path
import pandas as pd
from pymongo import MongoClient
from apps.main.utils.mongodb_data_load import MONGODB_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_csv_files(dump_dir):
    csv_files = glob.glob(f"{dump_dir}/*.csv")
    return csv_files


def upload_to_mongodb(documents, collection_name):
    client = MongoClient(MONGODB_URI)
    db = client["world_model"]
    collection = db[collection_name]
    collection.insert_many(documents)


def main():
    parser = argparse.ArgumentParser(description="Submit inference results to MongoDB")
    parser.add_argument(
        "--dump-dir",
        type=str,
        required=True,
        help="Directory where the dump files are stored",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        required=True,
        help="Name of the MongoDB collection to upload to",
    )

    args = parser.parse_args()

    dump_dir = args.dump_dir
    collection_name = args.collection_name

    logger.info(f"Dump directory: {dump_dir}")
    logger.info(f"Collection name: {collection_name}")

    # Add your code here to handle the submission logic
    csv_files = find_csv_files(dump_dir)
    logger.info(f"Found CSV files: {csv_files}")
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        documents = df.to_dict(orient="records")
        upload_to_mongodb(documents, collection_name)
        logger.info(f"Uploaded {len(documents)} documents to MongoDB")
    logger.info("Submission complete")


if __name__ == "__main__":
    main()
