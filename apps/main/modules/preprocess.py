import torch
import random
from PIL import Image
import numpy as np
import string
from dataclasses import dataclass
from torchvision import transforms
from typing import Dict, Optional
from torch import nn

from apps.main.utils.imagenet_classes import IMAGENET2012_CLASSES


######################## FOR IMAGE ########################


class ImageProcessing(nn.Module):
    def __init__(self, args) -> nn.Module:
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
                # logger.warning(f'Error to process the image: {e}')
                pass
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


class PolluxImageProcessing:
    def __init__(self, args):
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
        self.condition_image_size = args.condition_image_size

    def transform(self, x: Image) -> torch.Tensor:
        x = center_crop_arr(x, self.image_size)
        cond_x = x.resize(
            (self.condition_image_size, self.condition_image_size),
            resample=Image.BICUBIC,
        )
        return self.normalize_transform(x), self.normalize_transform(cond_x)


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

        # Create grids for patching
        patch_offsets = torch.arange(mask_patch, device=img_tensor.device).view(1, -1)

        x_indices = (x_coords.unsqueeze(-1) + patch_offsets).clamp(0, H - 1)
        y_indices = (y_coords.unsqueeze(-1) + patch_offsets).clamp(0, W - 1)

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


######################## FOR TEXT ########################


def generate_random_text(word_count=50):
    words = []
    for _ in range(word_count):
        word_length = random.randint(3, 10)  # Random word length between 3 and 10
        word = "".join(random.choices(string.ascii_lowercase, k=word_length))
        words.append(word)
    return " ".join(words)
