# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Dict, Any
import logging
import time
import torch
from torch import nn
from apps.main.modules.vae import build_vae, LatentVideoVAEArgs
from apps.offline_inf.data import AverageMeter
from apps.main.modules.text_encoder import LLAMATransformerArgs
from apps.main.modules.text_encoder import CLIP, CLIPArgs
import os

logger = logging.getLogger()


@dataclass
class ModelArgs:
    gen_vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)


class OfflineInference(nn.Module):
    """
    OfflineInference Model.
    This model integrates a VAE for latent compression
    """

    VERSION: str = "v1.0"

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.vae_compressor = build_vae(args.gen_vae)

    @torch.no_grad()
    def forward(
        self, batch: dict[str, Any], inference_meters: Dict[str, AverageMeter]
    ) -> dict[str, Any]:

        # Process latent code
        image = batch["image"].cuda()
        start_time = time.time()
        gen_latent_code = self.vae_compressor.encode(image)
        inference_time = time.time() - start_time
        inference_meters["latent_code"].update(inference_time, len(gen_latent_code))
        batch["latent_code"] = gen_latent_code
        return batch
