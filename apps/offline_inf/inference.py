from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

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
from apps.main.data import AutoDataLoader, DataArgs
from apps.gen_tran.generate import (
    LatentGenerator,
    GeneratorArgs,
    load_consolidated_model,
)

from apps.offline_inf.model import OfflineInference, ModelArgs


logger = logging.getLogger()


@dataclass
class EvalArgs:
    name: str = "evals"
    stage: str = "eval"
    dump_dir: Optional[str] = None
    ckpt_dir: str = ""
    generator: GeneratorArgs = field(default_factory=GeneratorArgs)
    source_data: List[DataArgs] = field(default_factory=list)
    target_data: DataArgs = field(default_factory=DataArgs)

    global_step: Optional[int] = None  # for in-training evaluation


def launch_eval(cfg: EvalArgs):
    if not torch.distributed.is_initialized():
        setup_torch_distributed(DistributedArgs())

    if (
        not (Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER).exists()
        and get_global_rank() == 0
    ):
        consolidate_checkpoints(cfg.ckpt_dir)
    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)
    torch.distributed.barrier()
    world_size = get_world_size()
    global_rank = get_global_rank()
    logger.info("Loading model")
    model, _ = load_consolidated_model(
        consolidated_path=cfg.ckpt_dir,
        model_cls=Pollux,
        model_args_cls=ModelArgs,
    )
    logger.info("Model loaded")
    model.eval()
    generator = LatentGenerator(cfg.generator, model)
    active_data = [d for d in cfg.eval_data if d.stage == cfg.stage and d.use]
    data_loader_factory = AutoDataLoader(
        shard_id=global_rank,
        num_shards=world_size,
        train_stage=cfg.stage,
        init_signal_handler=get_local_rank() == 0,
        data_config=active_data,  # Pass the filtered data configuration
        drop_last=False,
    )
    data_loader, _ = data_loader_factory.create_dataloader()
    torch.distributed.barrier()
    if get_local_rank() == 0 and hasattr(data_loader_factory.dataset, "clean_buffer"):
        data_loader_factory.dataset.clean_buffer()
    max_steps = cfg.sample_num // (active_data[0].batch_size * world_size)
    for idx, batch in enumerate(data_loader):
        generated_samples = generator(batch)
        save_images(
            generated_samples,
            output_dir=Path(cfg.dump_dir) / f"samples",
            prefix=f"image_rank{global_rank}_batch{idx}",
        )
        if idx + 1 >= max_steps:
            break
    del generator


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(EvalArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    launch_eval(cfg)


if __name__ == "__main__":
    main()
