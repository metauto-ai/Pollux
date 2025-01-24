import torch
import time
from dataclasses import dataclass
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
from apps.plan.model import Pollux, ModelArgs
from typing import List, Optional, Tuple, Union, Dict, Any
from apps.main.modules.schedulers import retrieve_timesteps, calculate_shift
from lingua.args import dataclass_from_dict
import logging
from pathlib import Path
from lingua.checkpoint import (
    CONSOLIDATE_NAME,
    consolidate_checkpoints,
    CONSOLIDATE_FOLDER,
)
from apps.plan.data import AutoDataLoader, DataArgs

logger = logging.getLogger()


@dataclass
class GeneratorArgs:
    resolution: int = 256
    in_channel: int = 3
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"


class LatentGenerator(nn.Module):
    def __init__(self, cfg: GeneratorArgs, model: nn.Module):
        super().__init__()
        self.model = model
        self.device = cfg.device
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

    @torch.no_grad()
    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        cur_device = next(self.model.parameters()).device
        cur_type = next(self.model.parameters()).dtype
        batch, _, _ = self.model.forward(data)
        batch["pred_img_tensor"] = self.model.tvae.decode(batch["pred_latent"])
        batch["ori_img_tensor"] = data["image"]
        return batch


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU. from https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/torch_utils.py
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = (
            generator.device.type
            if not isinstance(generator, list)
            else generator[0].device.type
        )
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents


def load_consolidated_model(
    consolidated_path,
    model_cls,
    model_args_cls,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    model = model_cls(model_args)
    consolidated_path = consolidate_checkpoints(ckpt_path)
    st_dict = torch.load(Path(consolidated_path) / CONSOLIDATE_NAME, weights_only=True)
    model.load_state_dict(st_dict["model"])
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, config


def main():
    # Load CLI arguments (overrides) and combine with a YAML config

    cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load(cfg.config)
    gen_cfg = dataclass_from_dict(GeneratorArgs, cfg, strict=False)
    print(cfg)

    model, _ = load_consolidated_model(
        cfg.ckpt_dir, model_cls=Pollux, model_args_cls=ModelArgs
    )

    generator = LatentGenerator(gen_cfg, model)
    active_data = [d for d in cfg.data if d.stage == "test" and d.use]
    logger.info(f"Actively using dataset: {active_data}")
    data_loader_factory = AutoDataLoader(
        shard_id=0,
        num_shards=1,
        train_stage="test",
        # init_signal_handler=get_local_rank() == 0,
        data_config=active_data,  # Pass the filtered data configuration
    )

    data_loader, sampler = data_loader_factory.create_dataloader()
    dataloader_iterator = iter(data_loader)

    # Start generation
    data = next(dataloader_iterator)
    data["image"] = data["image"].cuda()
    data["label"] = data["label"].cuda()
    start_time = time.time()
    samples = generator(data)
    end_time = time.time()

    # Calculate tokens per second
    save_image(
        samples["pred_img_tensor"],
        "pred_sample.png",
        nrow=3,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(
        samples["ori_img_tensor"],
        "raw_sample.png",
        nrow=3,
        normalize=True,
        value_range=(-1, 1),
    )
    logger.info(f"inference time is {end_time-start_time} seconds")


if __name__ == "__main__":
    main()
