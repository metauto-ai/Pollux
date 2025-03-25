import io
import os
import requests
import logging
from typing import Any
import time
import pandas as pd

import certifi
from urllib.parse import quote_plus
from pymongo import MongoClient
from typing import Final
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from bson import json_util, ObjectId
import pandas as pd
import pyarrow.parquet as pq
import bisect
from apps.main.modules.preprocess import PolluxImageProcessing
from apps.main.utils.dict_tensor_data_load import DictTensorBatchIterator
import time
import random
import numpy as np
import torch
import glob


class LocalParquetDataLoad(Dataset):
    def __init__(
        self,
        root_dir,
        num_shards,
        shard_idx,
        mapping_field,
        shape_field,
    ) -> None:
        super().__init__()
        self.num_shards = num_shards
        self.shard_idx = shard_idx
        self.mapping_field = mapping_field
        self.shape_field = shape_field
        self.data = self.consolidate_files(root_dir)

    def consolidate_files(self, root_dir):
        """
        Consolidate all parquet files in the root directory.

        Args:
            root_dir (str): Root directory containing the parquet files.
        """
        # Get all parquet files in the root directory
        total_parquet_files = glob.glob(
            os.path.join(root_dir, "**/*.parquet"), recursive=True
        )
        data = self.get_shard(total_parquet_files, self.num_shards, self.shard_idx)
        return data

    def get_shard(self, files, num_shards, shard_idx):
        return files[shard_idx::num_shards]

    def __len__(self):
        """Return the total number of rows in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self):
            idx = idx % len(self)
        file = self.data[idx]
        try:
            # updated to use memory-mapped reading
            if os.path.exists(file):
                table = pq.read_table(file, memory_map=True)
                cur_df = table.to_pandas()
            else:
                logging.warning(f"Invalid path or file not found: {file}")
        except Exception as e:
            logging.warning(f"Error reading parquet file: {file}")
            return self.__getitem__(random.choice(range(len(self))))
        return_parquet = {}
        records = cur_df.to_dict(orient="records")
        return_parquet = {self.mapping_field[k]: [] for k in self.mapping_field}
        for sample in records:
            for k, v in sample.items():
                if k in self.mapping_field:
                    k_ = self.mapping_field[k]
                    if isinstance(v, ObjectId):
                        return_parquet[k_].append(str(v))
                    elif isinstance(v, np.ndarray):
                        raw_shape_key = self.shape_field[k]
                        if raw_shape_key in sample:
                            reshaped_array = v.reshape(sample[raw_shape_key])
                            return_parquet[k_].append(
                                torch.from_numpy(np.copy(reshaped_array))
                            )
                        else:
                            raise ValueError(
                                f"Shape field {raw_shape_key} not found in the sample."
                            )
                    else:
                        return_parquet[k_].append(v)
        return return_parquet


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    from apps.main_dynamic_res.utils.dict_tensor_data_load import (
        DictTensorBatchIterator,
    )

    data = LocalParquetDataLoad(
        root_dir="/mnt/pollux/nemo/data/sample-latents",
        num_shards=1,
        shard_idx=0,
        mapping_field={
            "image_latent_512": "gen_latent_code",
            "image_latent_256": "plan_latent_code",
            "caption": "caption",
        },
        shape_field={
            "image_latent_256": "image_latent_shape_256",
            "image_latent_512": "image_latent_shape_512",
        },
    )
    sample_data = DataLoader(
        data,
        batch_size=1,
        collate_fn=lambda x: x,
    )
    sample_data = iter(sample_data)
    doc = next(sample_data)
    batch = next(DictTensorBatchIterator(doc[0], batch_size=64))
    for plan, gen in zip(batch["plan_latent_code"], batch["gen_latent_code"]):
        print(plan.shape)
        print(gen.shape)
        print("==================")
    print(batch["caption"])
print("Done")
