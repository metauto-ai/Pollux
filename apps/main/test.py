"""
python -m apps.main.test
"""

import logging
from torchvision.utils import save_image
from apps.main.modules.vae import LatentVideoVAEArgs
from apps.main.modules.schedulers import SchedulerArgs
from apps.main.modules.plan_transformer import PlanTransformerArgs
from apps.main.modules.gen_transformer import GenTransformerArgs
from apps.main.modules.tokenizer import TokenizerArgs
from apps.main.model import ModelArgs, Pollux
from dotenv import load_dotenv
from apps.main.data import AutoDataLoader, DataArgs
import torch
from apps.main.data import DataLoaderArgs
from apps.main.utils.mongodb_data_load import DictTensorBatchIterator

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format for log messages
    handlers=[logging.StreamHandler()],  # Also log to console
)

if __name__ == "__main__":
    dataloader_arg = DataLoaderArgs(
        batch_size=1,
        num_workers=16,
        seed=1024,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    data_config_dict = {
        "data": {
            "preliminary": {
                "cc12m_llama3bf128_hunyuanr256_test1m": {
                    "use": True,
                    "data_name": "cc12m_llama3bf128_hunyuanr256_test1m",
                    "source": "mongodb",
                    "task": "text_to_image",
                    "retries": 3,
                    "partition_key": "partition_key",
                    "extract_field": {
                        "parquet_size": "sample_num",
                        "parquet_path": "path",
                    },
                    "mapping_field": {
                        "HunyuanVideo_latent_code": "latent_code",
                        "LLAMA3_3B_text_embedding": "text_embedding",
                    },
                    "dataloader": dataloader_arg,
                }
            }
        }
    }

    data_config = [
        DataArgs(
            stage=stage,
            use=config["use"],
            data_name=config["data_name"],
            source=config["source"],
            task=config["task"],
            retries=config["retries"],
            partition_key=config["partition_key"],
            extract_field=config["extract_field"],
            mapping_field=config["mapping_field"],
            dataloader=config["dataloader"],
        )
        for stage, datasets in data_config_dict["data"].items()
        for dataset_name, config in datasets.items()
    ]
    dp_rank = 0
    dp_degree = 1
    data_loader_factory = AutoDataLoader(
        dp_rank,
        dp_degree,
        train_stage="preliminary",
        data_config=data_config,
    )
    data_loader, _ = data_loader_factory.create_dataloader()
    print(len(data_loader))
    print(len(data_loader_factory.dataset))
    count = 0
    import time

    start_times = time.time()
    for idx, batch in enumerate(data_loader):
        logging.info(f"step {count}")
        # print(key, value.size())
        # Forward pass
        # model(batch)
        # for key, value in batch.items():
        # print(key, value.size())
        iterrator = iter(DictTensorBatchIterator(batch, 64))
        while True:
            try:
                b = next(iterrator)
                count += 1
            except StopIteration:
                break
        if count > 1000:
            break
    logging.info(f"Time: {time.time() - start_times}")
