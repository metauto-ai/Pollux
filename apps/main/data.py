import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import datasets
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional, TypedDict
import logging
from torch import nn

# Configure logging
logger = logging.getLogger()


@dataclass
class DataArgs:
    root_dir: str = ""
    batch_size: int = 2
    num_workers: int = 8
    image_size: int = 256
    split: str = "train"


# Instantiate the dataset and dataloader
def create_dummy_dataloader(
    batch_size=32, num_samples=1000, num_classes=10, image_size=(3, 64, 64)
):
    """
    Create a dataloader for the diffusion model.
    Args:
        batch_size (int): Batch size for the dataloader.
        num_samples (int): Number of samples in the dataset.
        num_classes (int): Number of classes in the dataset.
        image_size (tuple): Size of the generated images (C, H, W).
    Returns:
        DataLoader: A DataLoader object for the dummy dataset.
    Usage:
    if __name__ == "__main__":
        dataloader = create_dataloader(batch_size=16)

        # Iterate through the dataloader
        for class_idx, time_step, image in dataloader:
            print(f"Class Index: {class_idx.shape}, Time Step: {time_step.shape}, Image: {image.shape}")
            break
    """
    dataset = DiffusionDummyDataset(
        num_samples=num_samples, num_classes=num_classes, image_size=image_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def may_download_image_dataset(path_name):
    data = datasets.load_dataset("ILSVRC/imagenet-1k", cache_dir=path_name)
    print(data["train"][0])


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def create_imagenet_dataloader(
    shard_id: int,
    num_shards: int,
    args: DataArgs,
) -> DataLoader:
    data = datasets.load_dataset("ILSVRC/imagenet-1k", cache_dir=args.root_dir)
    train_data = data[args.split]
    data_pipeline = DataPipeline(args)
    train_data.set_transform(data_pipeline)
    logger.warning(
        f"Read Data with Total Shard: {num_shards} Current Index: {shard_id} Split: {args.split}"
    )
    train_data = train_data.shard(num_shards=num_shards, index=shard_id)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.num_workers
    )
    return train_loader


# Dummy Dataset Class
class DiffusionDummyDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=10, image_size=(3, 64, 64)):
        """
        Initialize the dummy dataset.
        Args:
            num_samples (int): Number of samples in the dataset.
            num_classes (int): Number of distinct classes.
            image_size (tuple): Shape of the dummy images (C, H, W).
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a random sample.
        Returns:
            class_idx (int): A random class index.
            time_step (int): A random time step for the diffusion process.
            image (torch.Tensor): A random image tensor.
        """
        class_idx = np.random.randint(0, self.num_classes)
        time_step = np.random.randint(0, 1000)  # Assuming diffusion has 1000 steps
        image = torch.randn(self.image_size)  # Random image tensor
        return class_idx, time_step, image


class DataPipeline(nn.Module):
    def __init__(self, args: DataArgs) -> nn.Module:
        super().__init__()
        self.normalize_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        self.image_size = args.image_size

    def transform(self, x: Image) -> torch.Tensor:
        x = center_crop_arr(x, self.image_size)
        return self.normalize_transform(x)

    def forward(self, data: Dict) -> Dict:
        processed_image = []
        for image in data["image"]:
            try:
                processed_image.append(self.transform(image))
            except Exception as e:
                pass
                # logger.warning(f'Error to process the image: {e}')
        if len(data["image"]) != len(processed_image):
            dup_num = len(data["image"]) - len(processed_image)
            processed_image.extend([processed_image[-1]] * dup_num)
            for k in data.keys():
                if k != "image":
                    data[k][-dup_num:] = [data[k][-dup_num - 1]] * dup_num
        data["image"] = processed_image
        return data
