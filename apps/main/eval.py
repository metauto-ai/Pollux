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
from apps.main.data import create_imagenet_dataloader, DataArgs
from apps.main.generate import LatentGenerator, GeneratorArgs, load_consolidated_model

from apps.main.model import LatentDiffusionTransformer, ModelArgs

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


@dataclass
class EvalArgs:
    name: str = "evals"
    dump_dir: Optional[str] = None
    ckpt_dir: str = ""
    generator: GeneratorArgs = field(default_factory=GeneratorArgs)
    eval_data: DataArgs = field(default_factory=DataArgs)
    wandb: Optional[Any] = None
    sample_num: int = 1000

    global_step: Optional[int] = None  # for in-training evaluation


def save_images(
    tensors: torch.Tensor,
    output_dir: str,
    prefix: str = "image",
):
    os.makedirs(output_dir, exist_ok=True)
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
    logger.info(f"Saved {len(tensors)} images to {output_dir}")


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
        model_cls=LatentDiffusionTransformer,
        model_args_cls=ModelArgs,
    )
    logger.info("Model loaded")
    model.eval()
    generator = LatentGenerator(cfg.generator, model)
    data_loader = create_imagenet_dataloader(
        shard_id=global_rank,
        num_shards=world_size,
        args=cfg.eval_data,
    )
    max_steps = cfg.sample_num // (cfg.eval_data.batch_size * world_size)
    for idx, batch in enumerate(data_loader):
        generated_samples = generator(batch["label"].cuda())
        save_images(
            generated_samples,
            output_dir=Path(cfg.dump_dir) / f"samples",
            prefix=f"image_rank{global_rank}_batch{idx}",
        )
        if idx + 1 >= max_steps:
            break
    del generator


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate EvalArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call eval.py with eval.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in EvalArgs dataclass.
    """
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
