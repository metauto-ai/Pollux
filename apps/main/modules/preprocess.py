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
        self.max_ratio = args.max_ratio
        self.cls_list = list(IMAGENET2012_CLASSES.items())

    def transform(self, x: Image) -> torch.Tensor:
        x = center_crop_arr(x, self.image_size, self.max_ratio)
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
        self.max_ratio = args.max_ratio

    def transform(self, x: Image) -> torch.Tensor:
        x = center_crop_arr(x, self.image_size, self.max_ratio)
        # TODO: cond_x should also be resized according to the max_ratio
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


def center_crop_arr(pil_image, image_size, max_ratio=1.0):
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

    crop_size = var_center_crop_size_fn(pil_image.size, image_size, max_ratio=max_ratio)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - crop_size[0]) // 2
    crop_x = (arr.shape[1] - crop_size[1]) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + crop_size[0], crop_x : crop_x + crop_size[1]]
    )


def generate_crop_size_list(image_size, max_ratio=2.0):
    assert max_ratio >= 1.0
    patch_size = 32     # patch size increments
    assert image_size % patch_size == 0
    min_wp, min_hp = image_size // patch_size, image_size // patch_size
    crop_size_list = []
    wp, hp = min_wp, min_hp
    while hp / wp <= max_ratio:
        crop_size_list.append((wp * patch_size, hp * patch_size))
        hp += 1
    wp, hp = min_wp + 1, min_hp
    while wp / hp <= max_ratio:
        crop_size_list.append((wp * patch_size, hp * patch_size))
        wp += 1
    return crop_size_list


def is_valid_crop_size(cw, ch, orig_w, orig_h):
    down_scale = max(cw / orig_w, ch / orig_h)
    return cw <= orig_w * down_scale and ch <= orig_h * down_scale


# TODO: patch_size and max_ratio should be args
def var_center_crop_size_fn(orig_img_shape, image_size, max_ratio=2.0):
    """
    Dynamic cropping from Lumina-Image-2.0
    https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/imgproc.py#L39
    """
    w, h = orig_img_shape[:2]
    crop_size_list = generate_crop_size_list(
        image_size=image_size, 
        max_ratio=max_ratio
    )
    rem_percent = [
        min(cw / w, ch / h) / max(cw / w, ch / h) 
        if is_valid_crop_size(cw, ch, w, h) else 0 
        for cw, ch in crop_size_list
    ]
    crop_size = sorted(((x, y) for x, y in zip(rem_percent, crop_size_list) if x > 0), reverse=True)[0][1]
    return np.array(crop_size, dtype=np.int32)

######################## FOR TEXT ########################


def generate_random_text(word_count=50):
    words = []
    for _ in range(word_count):
        word_length = random.randint(3, 10)  # Random word length between 3 and 10
        word = "".join(random.choices(string.ascii_lowercase, k=word_length))
        words.append(word)
    return " ".join(words)
