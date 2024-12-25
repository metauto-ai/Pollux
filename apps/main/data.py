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
from utils.imagenet_classes import IMAGENET2012_CLASSES
import logging
from torch import nn
import random
import string
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger()


@dataclass
class DataArgs:
    source: str = "imagenet"
    batch_size: int = 2
    num_workers: int = 8
    image_size: int = 256
    split: str = "train"
    root_dir: Optional[str] = None


# Instantiate the dataset and dataloader
def create_dummy_dataloader(
    batch_size=32,
    num_samples=1000,
    num_classes=10,
    image_size=(3, 64, 64),
    word_count=256,
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
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        word_count=word_count,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def may_download_image_dataset(path_name):
    data = datasets.load_dataset("ILSVRC/imagenet-1k", cache_dir=path_name)
    print(data["train"][0])


def random_mask_images(img_tensor, mask_ratio, mask_patch, mask_all=False):
    """
    Randomly masks an image tensor or applies full masking if mask_all is True.

    Args:
        img_tensor (torch.Tensor): Input tensor with shape (N, C, H, W).
        mask_ratio (float): Ratio of the total patches to be masked (0 to 1).
        mask_patch (int): Size of the square patch to be masked (in pixels).
        mask_all (bool): If True, mask the entire tensor.

    Returns:
        torch.Tensor: Mask tensor resized to (N, 1, H/8, W/8).
        torch.Tensor: Masked image tensor.
    """
    N, C, H, W = img_tensor.shape

    if mask_all:
        # Create a full mask with ones (everything masked)
        mask = torch.ones(
            (N, 1, H, W), device=img_tensor.device, dtype=img_tensor.dtype
        )
    else:
        # Compute the total number of patches to mask
        num_patches = int(mask_ratio * (H * W) / (mask_patch**2))

        # Generate random patch indices for all images in parallel
        x_coords = torch.randint(
            0, H - mask_patch + 1, (N, num_patches), device=img_tensor.device
        )
        y_coords = torch.randint(
            0, W - mask_patch + 1, (N, num_patches), device=img_tensor.device
        )

        # Create the mask tensor
        mask = torch.zeros(
            (N, 1, H, W), device=img_tensor.device, dtype=img_tensor.dtype
        )

        # Apply patches to the mask efficiently
        for dx in range(mask_patch):
            for dy in range(mask_patch):
                mask[
                    torch.arange(N, device=img_tensor.device)
                    .view(-1, 1)
                    .repeat(1, num_patches)
                    .flatten(),
                    :,
                    (x_coords + dx).clamp(0, H - 1).flatten(),
                    (y_coords + dy).clamp(0, W - 1).flatten(),
                ] = 1

    # Apply the mask to the input tensor
    masked_tensor = img_tensor * (1 - mask)

    return mask, masked_tensor


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


def create_dataloader(
    shard_id: int,
    num_shards: int,
    args: DataArgs,
):
    if args.source == "imagenet":
        return create_imagenet_dataloader(
            shard_id,
            num_shards,
            args,
        )
    if args.source == "dummy":
        return create_dummy_dataloader(
            batch_size=args.batch_size,
            image_size=(3, args.image_size, args.image_size),
            word_count=32,
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
    def __init__(
        self,
        num_samples=1000,
        num_classes=10,
        image_size=(3, 64, 64),
        word_count=256,
    ):
        """
        Initialize the dummy dataset.
        Args:
            num_samples (int): Number of samples in the dataset.
            num_classes (int): Number of distinct classes.
            image_size (tuple): Shape of the dummy images (C, H, W).
            word_count: Number of the word in this caption
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.word_count = word_count

    def __len__(self):
        return self.num_samples

    def generate_random_text(self, word_count=50):
        words = []
        for _ in range(word_count):
            word_length = random.randint(3, 10)  # Random word length between 3 and 10
            word = "".join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)
        return " ".join(words)

    def __getitem__(self, idx):
        """
        Generate a random sample.
        Returns:
            class_idx (int): A random class index.
            time_step (int): A random time step for the diffusion process.
            image (torch.Tensor): A random image tensor.
        """
        class_idx = np.random.randint(0, self.num_classes)
        image = torch.randn(self.image_size)  # Random image tensor
        caption = self.generate_random_text()
        batch = {"label": class_idx, "caption": caption, "image": image}
        return batch


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
        self.cls_list = list(IMAGENET2012_CLASSES.items())

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
        caption_list = []
        for label in data["label"]:
            _, cap = self.cls_list[label]
            caption_list.append(cap)
        data["caption"] = caption_list
        return data
