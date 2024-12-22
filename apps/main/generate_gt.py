"""
python -m apps.main.generate_gt
"""

import logging
from lingua.transformer import precompute_freqs_cis

from apps.main.data import (
    create_dummy_dataloader,
    create_imagenet_dataloader,
    DataArgs,
    may_download_image_dataset,
)
from apps.main.eval import save_images
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format for log messages
    handlers=[logging.StreamHandler()],  # Also log to console
)


if __name__ == "__main__":
    # may_download_image_dataset('/mnt/data/imagenet')
    data_arg = DataArgs(
        root_dir="/mnt/data/imagenet",
        image_size=256,
        num_workers=8,
        batch_size=16,
        split="validation",
    )
    train_loader = create_imagenet_dataloader(shard_id=0, num_shards=1, args=data_arg)
    for idx, data in tqdm(enumerate(train_loader)):
        assert isinstance(data, dict)
        save_images(
            data["image"],
            output_dir="/mnt/data/tmp/imagenet_val_gt",
            prefix=f"{idx}_",
        )
