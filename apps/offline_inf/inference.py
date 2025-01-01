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
)
import numpy as np
from apps.main.utils.mongodb_data_load import MONGODB_URI

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
    model.init_weights(cfg.model)
    logger.info("Model loaded")
    model.cuda().eval()

    active_data = [d for d in cfg.source_data if d.stage == cfg.stage and d.use]
    data_loader_factory = AutoDataLoader(
        shard_id=global_rank,
        num_shards=world_size,
        train_stage=cfg.stage,
        data_config=active_data,  # Pass the filtered data configuration
        drop_last=False,
    )
    data_loader, _ = data_loader_factory.create_dataloader()

    # Benchmark loading from MongoDB web
    # benchmark_url_loading(data_loader)

    # * init our profiling meters for different type of tensors we want to save,
    # latent_code or text_embedding
    # Initialize profiling meters for all tensor types
    inference_meters = {
        tensor_type: AverageMeter() for tensor_type in cfg.prefix_mapping
    }
    save_meters = {"parquet": AverageMeter()}
    storage_meters = {"parquet": StorageMeter()}
    save_batch = {}
    in_parquet_num = 0
    count = 0
    logger.info("Start inference now....")

    for idx, batch in enumerate(data_loader):
        batch = model.forward(batch, inference_meters)
        if len(save_batch) == 0 or in_parquet_num < cfg.parque_size:
            for key, prefix in cfg.prefix_mapping.items():
                if isinstance(batch[key], torch.Tensor):
                    data = batch[key].detach().cpu().numpy()
                    batch[key] = [d.reshape(-1) for d in data]
                    batch[f"{key}_raw_shape"] = [d.shape for d in data]
                if prefix not in save_batch:
                    save_batch[prefix] = batch[key]
                    if f"{key}_raw_shape" in batch:
                        save_batch[f"{key}_raw_shape"] = batch[f"{key}_raw_shape"]
                else:
                    save_batch[prefix].extend(batch[key])
                    if f"{key}_raw_shape" in batch:
                        save_batch[f"{key}_raw_shape"].extend(batch[f"{key}_raw_shape"])
            in_parquet_num += len(save_batch[prefix])

        if in_parquet_num >= cfg.parque_size:
            parquet_path = save_parquet(
                save_batch,
                cfg.dump_dir,
                f"{world_size}_{global_rank}_{count}",
                save_meter=save_meters["parquet"],
                storage_meter=storage_meters["parquet"],
            )
            count += 1

            batch_df = {
                "path": [parquet_path],
                "sample_num": [in_parquet_num],
                "timestamp": [datetime.now()],
                "data_source": [active_data[0].data_name],
                "resolution": [active_data[0].image_size],
                "token_length": [cfg.model.plan_transformer.text_seqlen],
            }
            batch_df = pd.DataFrame(batch_df)
            if csv_path.exists():
                batch_df.to_csv(csv_path, mode="a", header=False, index=False)
            else:
                batch_df.to_csv(csv_path, mode="w", header=True, index=False)
            save_batch = {}
            in_parquet_num = 0
        # Jinjie: if we need profile, early break here
        if idx > 1000:
            break
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
