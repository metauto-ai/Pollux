"""
CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes 1 --nproc-per-node 2 -m apps.main.eval config=apps/main/configs/eval.yaml                                                         
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import csv
from omegaconf import OmegaConf
import torchvision.transforms as transforms
import torch
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from lingua.distributed import (
    DistributedArgs,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
)
from apps.main.data import AutoDataLoader, DataArgs
from apps.main.generate import LatentGenerator, GeneratorArgs, load_consolidated_model
from apps.main.modules.vae import build_vae
from apps.main.model import Latent_Pollux, ModelArgs

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


@dataclass
class EvalArgs:
    name: str = "evals"
    stage: str = "eval"
    dump_dir: Optional[str] = None
    ckpt_dir: str = ""
    generator: GeneratorArgs = field(default_factory=GeneratorArgs)
    data: List[DataArgs] = field(default_factory=list)


def save_images(
    batch: dict,
    output_dir: str,
    prefix: str = "image",
    csv_name: str = "meta.csv",
):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / f"{csv_name}"
    tensors = batch["generated_samples"]
    tensors = (tensors + 1) * 127.5
    tensors = tensors.clamp(0, 255).byte()

    # Convert each tensor image to PIL format and save
    for i, img_tensor in enumerate(tensors):
        # Permute channels to HWC for PIL
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        # Define image path
        image_path = os.path.join(output_dir, f"{prefix}_{i}.png")
        # Save image
        img_pil.save(image_path)
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([batch["caption"][i], image_path])

    logger.warning(f"Saved {len(tensors)} images to {output_dir}")


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
        model_cls=Latent_Pollux,
        model_args_cls=ModelArgs,
    )
    logger.info("Model loaded")
    model.eval()
    tvae = build_vae(cfg.generator.tvae)
    generator = LatentGenerator(cfg.generator, model, tvae).cuda()
    active_data = [d for d in cfg.data if d.stage == cfg.stage and d.use]
    data_loader_factory = AutoDataLoader(
        shard_id=global_rank,
        num_shards=world_size,
        train_stage=cfg.stage,
        data_config=active_data,  # Pass the filtered data configuration
    )
    data_loader, sampler = data_loader_factory.create_dataloader()
    logger.info("Data loader is built !")
    logger.info(f"Data loader size: {len(data_loader)}")
    for idx, batch in enumerate(data_loader):
        batch["generated_samples"] = generator(batch)
        save_images(
            batch,
            output_dir=Path(cfg.dump_dir) / f"samples",
            prefix=f"image_rank{global_rank}_batch{idx}",
            csv_name=f"meta_rank{global_rank}.csv",
        )
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
