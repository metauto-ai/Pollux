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
    plan_vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    gen_vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    text_encoder: CLIPArgs = field(default_factory=CLIPArgs)


class OfflineInference(nn.Module):
    """
    OfflineInference Model.
    This model integrates a VAE for latent compression
    """

    VERSION: str = "v1.0"

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.gen_compressor = build_vae(args.gen_vae)
        self.text_encoder = CLIP(args.text_encoder)
        self.plan_compressor = build_vae(args.plan_vae)

    @torch.no_grad()
    def forward(
        self, batch: dict[str, Any], inference_meters: Dict[str, AverageMeter]
    ) -> dict[str, Any]:
        # Process text embedding
        start_time = time.time()
        batch["text_embedding"] = self.text_encoder(batch)
        inference_time = time.time() - start_time
        inference_meters["text_embedding"].update(
            inference_time, len(batch["text_embedding"])
        )

        # Process latent code
        image = batch["image"].cuda()
        start_time = time.time()
        gen_latent_code = self.gen_compressor.encode(image)
        inference_time = time.time() - start_time
        inference_meters["gen_latent_code"].update(inference_time, len(gen_latent_code))
        batch["gen_latent_code"] = gen_latent_code

        start_time = time.time()
        plan_vae_indices, plan_vae_latent = self.plan_compressor.encode(image)
        inference_time = time.time() - start_time
        inference_meters["plan_latent_code"].update(
            inference_time, len(plan_vae_latent)
        )
        batch["plan_latent_code"] = plan_vae_latent
        batch["plan_latent_code_indices"] = plan_vae_indices
        return batch
