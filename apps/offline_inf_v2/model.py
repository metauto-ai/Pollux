# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
import logging
import torch
from torch import nn
from apps.main.modules.vae import build_vae, LatentVideoVAEArgs

logger = logging.getLogger()


@dataclass
class ModelArgs:
    gen_vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    autocast: bool = field(default=False)
    use_compile: bool = field(default=False)

class VAE(nn.Module):
    """
    VAE Model
    """

    VERSION: str = "v1.0"

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.vae_compressor = build_vae(args.gen_vae)

    @torch.no_grad()
    def forward(
        self, image: torch.Tensor
    ) -> torch.Tensor:

        # Process latent code
        image = image.cuda()
        latent_code = self.vae_compressor.encode(image)

        return latent_code
