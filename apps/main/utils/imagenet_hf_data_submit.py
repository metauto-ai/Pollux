import os
import logging
import datasets
from apps.main.utils.mongodb_data_load import MONGODB_URI
from apps.main.utils.imagenet_classes import IMAGENET2012_CLASSES
from pymongo import MongoClient
from typing import Dict
from pathlib import Path
import uuid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger()


def data_submit(
    data_name: str = "ILSVRC/imagenet-1k",
    cache_dir="/jfs/data/imagenet",
    split="validation",
) -> str:

    data = datasets.load_dataset(path=data_name, cache_dir=cache_dir)
    logger.info(f'Dataset "{data_name}" loaded from local cache.')
    data = data[split]
    data_pipeline = ImageProcessing(split)
    data.set_transform(data_pipeline)
    return DataLoader(data, batch_size=64, num_workers=32)


class ImageProcessing(nn.Module):
    def __init__(self, split="val") -> nn.Module:
        super().__init__()
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client["world_model"]
        self.collection = self.db["imagenet-1k"]
        self.cls_list = list(IMAGENET2012_CLASSES.items())
        self.dir = f"/jfs/data/imagenet-MongoDB/{split}"
        self.split = split

    def forward(self, data: Dict) -> Dict:
        success = 0.0
        for idx, image in enumerate(data["image"]):
            if image.mode != "RGB":
                continue
            try:
                img_path = str(Path(self.dir) / f"{uuid.uuid4()}.jpg")
                label = data["label"][idx]
                _, caption = self.cls_list[label]
                width, height = image.size
                image.save(img_path)
                res = self.collection.insert_one(
                    {
                        "image": img_path,
                        "label": label,
                        "caption": caption,
                        "width": width,
                        "height": height,
                        "split": self.split,
                    }
                )
                logger.info(f"success to upload the image with {res}")
                success += 1
            except Exception as e:
                logger.warning(f"Error to process the image: {e}")
                # pass
                success -= 1
        return {"label": data["label"]}


if __name__ == "__main__":
    data = data_submit(split="train")
    for batch in tqdm(data):
        pass
