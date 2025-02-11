from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from pymongo import MongoClient
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import torch
from lingua.args import dump_config
from lingua.logger import init_logger
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import (
    DistributedArgs,
    dist_mean_dict,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
    get_local_rank,
)
import uuid
import time
from apps.main.data import AutoDataLoader, DataArgs
import pandas as pd
from apps.offline_inf.model import OfflineInference, ModelArgs
from apps.offline_inf.data import (
    AverageMeter,
    StorageMeter,
    save_tensor,
    transform_dict,
    remove_tensors,
    benchmark_loading,
    benchmark_url_loading,
    save_parquet,
    upload_parquet,
)
import numpy as np
from apps.main.utils.mongodb_data_load import MONGODB_URI
import glob
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@dataclass
class InferenceArgs:
    name: str = "evals"
    stage: str = "eval"
    dump_dir: Optional[str] = None
    model: ModelArgs = field(default_factory=ModelArgs)
    source_data: List[DataArgs] = field(default_factory=list)
    prefix_mapping: Optional[dict[str, str]] = field(default_factory=dict)
    parque_size: int = 32768  # Number of samples to save in a single parquet file
    # * whether do profiling
    profile: Optional[bool] = False
    max_save_attempt: int = 3  # max number of attempts to save a parquet file
    delay: int = 5  # seconds


def launch_inference(cfg: InferenceArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)
    init_logger(Path(cfg.dump_dir) / f"{cfg.stage}.log")
    torch.distributed.barrier()
    world_size = get_world_size()
    global_rank = get_global_rank()
    csv_path = Path(cfg.dump_dir) / f"{world_size}_{global_rank}_metadata.csv"
    logger.info("Loading model")
    model = OfflineInference(cfg.model)
    logger.info("Model loaded")
    is_s3 = cfg.dump_dir.startswith("s3:")
    if is_s3:
        s3_path = cfg.dump_dir.rstrip("/")
        logger.info(f"Uploading to S3 path: {s3_path}")
    model.cuda().eval()

    active_data = [d for d in cfg.source_data if d.stage == cfg.stage and d.use]
    data_loader_factory = AutoDataLoader(
        shard_id=global_rank,
        num_shards=world_size,
        train_stage=cfg.stage,
        data_config=active_data,  # Pass the filtered data configuration
    )
    data_loader, sampler = data_loader_factory.create_dataloader()
    inference_meters = {
        tensor_type: AverageMeter() for tensor_type in cfg.prefix_mapping
    }
    save_meters = {"parquet": AverageMeter()}
    storage_meters = {"parquet": StorageMeter()}
    save_batch = {}
    in_parquet_num = 0

    # * == stateful inference, if s3_path is not none means saving to dump_dir ==
    # saving to s3_path
    if is_s3:
        logger.warning(f"saving to s3 cloud: {s3_path}")
        if os.path.exists(
            os.path.join(cfg.dump_dir, f"{world_size}_{global_rank}_metadata.csv")
        ):
            df = pd.read_csv(
                os.path.join(cfg.dump_dir, f"{world_size}_{global_rank}_metadata.csv")
            )
            saved_parquet_num = len(df)
            previous_parquet_size = df.iloc[0]["sample_num"]
            assert (
                previous_parquet_size == cfg.parque_size
            ), f"Parquet size must be consistent, prevs {previous_parquet_size} == but now {cfg.parque_size}"

            index_to_start = cfg.parque_size * saved_parquet_num
            logger.warning(
                f"{saved_parquet_num} saved parquet found, with parquet_size={cfg.parque_size}, resuming inference starting from {index_to_start}-th item, continue to generate {saved_parquet_num+1}-th parquet ..."
            )
            # set sampler state and counter
            sampler.load_state_dict({"start_index": index_to_start})
        else:
            logger.warning(f"No saved parquet found, doing fresh start now...")
            saved_parquet_num = 0
            index_to_start = 0
    # support resume if dump_dir is the same
    else:
        logger.warning(f"s3_path not found, saving to local {cfg.dump_dir}")
        saved_parquet = list(
            glob.glob(
                os.path.join(cfg.dump_dir, f"{world_size}_{global_rank}_*.parquet")
            )
        )
        saved_parquet_num = len(saved_parquet)
        if saved_parquet_num <= 0:
            logger.warning(f"No saved parquet found, doing fresh start now...")
            index_to_start = 0
        else:
            df = pd.read_csv(
                os.path.join(cfg.dump_dir, f"{world_size}_{global_rank}_metadata.csv")
            )
            assert saved_parquet_num == len(
                df
            ), f"Parquet CSV record must be consistent with parquet files on the disk, csv record has {len(df)} == but now on the disk {saved_parquet_num}"
            previous_parquet_size = df.iloc[0]["sample_num"]
            assert (
                previous_parquet_size == cfg.parque_size
            ), f"Parquet size must be consistent, prevs {previous_parquet_size} == but now {cfg.parque_size}"
            index_to_start = cfg.parque_size * saved_parquet_num
            logger.warning(
                f"{saved_parquet_num} saved parquet found, with parquet_size={cfg.parque_size}, resuming inference starting from {index_to_start}-th item, continue to generate {saved_parquet_num+1}-th parquet ..."
            )
            # set sampler state and counter
            sampler.load_state_dict({"start_index": index_to_start})

    count = saved_parquet_num

    # * == start inference ==
    logger.info("Start inference now....")
    sleep_time = random.randint(1, 30)
    time.sleep(
        sleep_time
    )  # Haozhe: Add a trick here, sleep for a random time to avoid all workers start at the same time,
    # Reduce the load on the storage
    for idx, batch in enumerate(data_loader):
        batch = model.forward(batch, inference_meters)

        if len(save_batch) == 0 or in_parquet_num < cfg.parque_size:
            for key, prefix in cfg.prefix_mapping.items():
                if isinstance(batch[key], torch.Tensor):
                    data = batch[key].detach().cpu().to(torch.float32).numpy()
                    batch[key] = [d.reshape(-1) for d in data]
                    batch[f"{key}_raw_shape"] = [d.shape for d in data]

                if prefix not in save_batch:
                    save_batch[prefix] = batch[key]
                    if f"{key}_raw_shape" in batch:
                        save_batch[f"{prefix}_raw_shape"] = batch[f"{key}_raw_shape"]
                else:
                    save_batch[prefix].extend(batch[key])
                    if f"{key}_raw_shape" in batch:
                        save_batch[f"{prefix}_raw_shape"].extend(
                            batch[f"{key}_raw_shape"]
                        )
            in_parquet_num += len(batch[key])
        if in_parquet_num >= cfg.parque_size:
            for attempt in range(cfg.max_save_attempt):
                try:
                    file_name = f"{world_size}_{global_rank}_{uuid.uuid4()}"
                    if is_s3:
                        parquet_path = upload_parquet(
                            save_batch,
                            s3_path,
                            file_name,
                            save_meter=storage_meters["parquet"],
                        )
                    else:
                        parquet_path = save_parquet(
                            save_batch,
                            cfg.dump_dir,
                            file_name,
                            save_meter=save_meters["parquet"],
                            storage_meter=storage_meters["parquet"],
                        )
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < cfg.max_retries - 1:
                        logger.info(f"Retrying in {cfg.delay} seconds...")
                        time.sleep(cfg.delay)
                    else:
                        logger.error("All attempts failed. Unable to save to S3.")
                else:
                    count += 1

                    batch_df = {
                        "path": [parquet_path],
                        "sample_num": [in_parquet_num],
                        "timestamp": [datetime.now()],
                        "data_source": [active_data[0].data_name],
                        "resolution": [active_data[0].image_size],
                        "token_length": [cfg.model.text_encoder.text_seqlen],
                    }
                    batch_df = pd.DataFrame(batch_df)
                    if csv_path.exists():
                        batch_df.to_csv(csv_path, mode="a", header=False, index=False)
                    else:
                        batch_df.to_csv(csv_path, mode="w", header=True, index=False)
                    break
            save_batch = {}
            in_parquet_num = 0
        # Jinjie: if we need profile, early break here
    # Conclude profiling
    for name, meter in inference_meters.items():
        meter.conclude(f"Inference ({name})")
    for name, meter in save_meters.items():
        meter.conclude(f"Saving ({name})")
    for name, meter in storage_meters.items():
        meter.conclude(f"Storage ({name})")

    # Benchmark loading from disk
    # benchmark_loading(cfg.dump_dir, cfg.prefix_mapping, cfg.save_format)

    del model
    # client.close()


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(InferenceArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_inference(cfg)


if __name__ == "__main__":
    main()
