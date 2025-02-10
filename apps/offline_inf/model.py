# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Dict, Any
import logging
import time
import torch
from torch import nn
from apps.main.modules.vae import build_vae, LatentVideoVAEArgs
from apps.main.modules.tokenizer import Tokenizer, TokenizerArgs
from apps.offline_inf.data import AverageMeter
from apps.main.modules.text_encoder import LLAMATransformerArgs, LLAMA3
import os

logger = logging.getLogger()


@dataclass
class ModelArgs:
    plan_vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    gen_vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    text_encoder: LLAMATransformerArgs = field(default_factory=LLAMATransformerArgs)


class OfflineInference(nn.Module):
    """
    OfflineInference Model.
    This model integrates a VAE for latent compression
    """

    VERSION: str = "v1.0"

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.gen_compressor = build_vae(args.gen_vae)
        self.tokenizer = Tokenizer(model_path=args.tokenizer.model_path)
        self.text_encoder = LLAMA3(args.text_encoder)
        self.plan_compressor = build_vae(args.plan_vae)

    def cap_pos_tokenize(self, batch: dict[str:any]) -> dict[str:any]:
        batch["cap_token"] = []
        for x in batch["caption"]:
            if not isinstance(x, str):
                logger.warning(f"Expected string but got {type(x)}: {x}")
                batch["cap_token"].append(
                    self.tokenizer.encode("", bos=True, eos=False)
                )
            else:
                batch["cap_token"].append(self.tokenizer.encode(x, bos=True, eos=False))

        pad_id = self.tokenizer.pad_id
        bsz = len(batch["cap_token"])
        tokens = torch.full(
            (bsz, self.text_encoder.text_seqlen),
            pad_id,
            dtype=torch.long,
        ).cuda()
        for k, t in enumerate(batch["cap_token"]):
            if len(t) < tokens.size(1):
                tokens[k, : len(t)] = torch.tensor(
                    t[:], dtype=torch.long, device="cuda"
                )
            else:
                tokens[k, :] = torch.tensor(
                    t[: tokens.size(1)], dtype=torch.long, device="cuda"
                )
        batch["cap_token"] = tokens.cuda()
        return batch

    @torch.no_grad()
    def forward(
        self, batch: dict[str, Any], inference_meters: Dict[str, AverageMeter]
    ) -> dict[str, Any]:
        # Process text embedding
        start_time = time.time()
        batch = self.cap_pos_tokenize(batch)
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

    def init_weights(self, args: ModelArgs):
        self.text_encoder.init_weights(args.text_encoder.pre_trained_path)
