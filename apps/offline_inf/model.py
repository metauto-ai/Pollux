# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import logging
import random
import torch
from torch import nn
import torch.nn.functional as F
from apps.main.modules.plan_transformer import (
    PlanTransformerArgs,
)
from apps.main.modules.vae import LatentVideoVAE, LatentVideoVAEArgs


logger = logging.getLogger()


@dataclass
class ModelArgs:
    plan_transformer: PlanTransformerArgs = field(default_factory=PlanTransformerArgs)
    vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)


class OfflineInference(nn.Module):
    """
    OfflineInference Model.
    This model integrates a VAE for latent compression
    """

    VERSION: str = "v0.1"

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.compressor = LatentVideoVAE(args.vae)
        # self.plan_transformer = LabelEmbedder(
        #     num_classes=args.num_classes,
        #     hidden_size=args.gen_transformer.dim,
        #     dropout_prob=args.image_cfg_ratio,
        # )

    def forward(self, batch: dict[str:any]) -> dict[str:any]:

        image = batch["image"]
        latent_code = self.compressor.encode(image)
        return latent_code
