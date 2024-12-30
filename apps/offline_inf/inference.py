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

from apps.offline_inf.model import OfflineInference, ModelArgs
from apps.offline_inf.data import (
    AverageMeter,
    StorageMeter,
    save_tensor,
    transform_dict,
    remove_tensors,
    benchmark_loading,
)
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
    collection_name: str = ""
    save_format: str = (
        ".npy"  # Acceptable formats: ".pkl", ".pt", ".npy", ".safetensors"
    )


def launch_inference(cfg: InferenceArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)
    torch.distributed.barrier()
    world_size = get_world_size()
    global_rank = get_global_rank()

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

    client = MongoClient(MONGODB_URI)
    db = client["world_model"]
    collection = db[cfg.collection_name]

    # * init our profiling meters for different type of tensors we want to save,
    # latent_code or text_embedding
    # Initialize profiling meters for all tensor types
    inference_meters = {
        tensor_type: AverageMeter() for tensor_type in cfg.prefix_mapping
    }
    save_meters = {tensor_type: AverageMeter() for tensor_type in cfg.prefix_mapping}
    storage_meters = {tensor_type: StorageMeter() for tensor_type in cfg.prefix_mapping}

    for idx, batch in enumerate(data_loader):
        batch = model.forward(batch, inference_meters)

        for key, prefix in cfg.prefix_mapping.items():
            save_tensor(
                tensors=batch[key],
                batch=batch,
                output_dir=Path(cfg.dump_dir),
                prefix=prefix,
                save_format=cfg.save_format,
                save_meter=save_meters[key],
                storage_meter=storage_meters[key],
            )

        batch = remove_tensors(batch)
        batch = transform_dict(batch)
        # TODO: Jinjie: if I enable this line will have bug
        # ** And if I enable it the speed will be very slow !
        # collection.insert_many(batch)

        # Jinjie: if we need profile, early break here
        if idx >= 100:
            break

    # Conclude profiling
    for name, meter in inference_meters.items():
        meter.conclude(f"Inference ({name})")
    for name, meter in save_meters.items():
        meter.conclude(f"Saving ({name})")
    for name, meter in storage_meters.items():
        meter.conclude(f"Storage ({name})")

    # TODO: same problem as above, maybe we write DB multiple times?
    # all_tensor_paths = []  # Collect all tensor paths
    # for k, v in cfg.prefix_mapping.items():
    #     all_tensor_paths.extend(
    #         [
    #             Path(cfg.dump_dir) / v / f"{v}_{str(id)}{cfg.save_format}"
    #             for id in batch["_id"]
    #         ]
    #     )

    # Benchmark loading
    benchmark_loading(cfg.dump_dir, cfg.prefix_mapping, cfg.save_format)
    del model
    client.close()
    logger.info(f"Profiling for {cfg.save_format} finished")


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
