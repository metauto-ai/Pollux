"""_summary_
python -m apps.offline_inf.submit --dump-dir /jfs/data/tmp/cc12m_test_1m --collection-name cc12m_llama3bf128_hunyuanr256_test1m
"""

import argparse
import logging
import glob
from pathlib import Path
import pandas as pd
from pymongo import MongoClient
from apps.main.utils.mongodb_data_load import MONGODB_URI
from apps.main.utils.mongodb_data_load import MongoDBDataLoad
from torch.utils.data import Dataset, DataLoader
from bson import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBParquetUpdator(MongoDBDataLoad):
    def __init__(
        self,
        num_shards,
        shard_idx,
        query,
        collection_name,
        extract_field,
        partition_key,
    ) -> None:
        super().__init__(
            num_shards=num_shards,
            shard_idx=shard_idx,
            collection_name=collection_name,
            query=query,
            partition_key=partition_key,
        )
        self.path_field = extract_field["parquet_path"]
        self.id_field = extract_field["id"]
        self.collection_name = collection_name

    def __len__(self):
        """Return the total number of rows in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            idx = idx % len(self)
        file = self.data.iloc[idx][self.path_field]
        try:
            cur_df = pd.read_parquet(file, engine="pyarrow")
            logger.info(f"Read parquet file: {file}")
        except Exception as e:
            logger.error(f"Error reading parquet file: {file}")
            client = MongoClient(MONGODB_URI)
            db = client["world_model"]
            collection = db[self.collection_name]
            object_id = self.data.iloc[idx][self.id_field]
            result = collection.delete_one({"_id": ObjectId(object_id)})
            client.close()
            if result.deleted_count > 0:
                logger.info(f"Document with _id {object_id} deleted successfully.")
            else:
                logger.info(f"No document found with _id {object_id}.")
        return 0


def main():

    dataset = MongoDBParquetUpdator(
        num_shards=1,
        shard_idx=0,
        query={},
        collection_name="cc12m_l3bf128_hr256",
        extract_field={
            "parquet_size": "sample_num",
            "parquet_path": "path",
            "id": "_id",
        },
        partition_key="partition_key",
    )
    dataset.set_local_partition()
    data = DataLoader(dataset, batch_size=32, num_workers=128, pin_memory=False)
    for _ in data:
        pass


if __name__ == "__main__":
    main()
