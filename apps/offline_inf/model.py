# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import logging
import random
import time
import torch
from torch import nn
import torch.nn.functional as F
from apps.main.modules.vae import LatentVideoVAE, LatentVideoVAEArgs
from apps.main.modules.plan_transformer import (
    BasePlanTransformer,
    RotaryEmbedding1D,
    RMSNorm,
)
from apps.main.modules.tokenizer import Tokenizer, TokenizerArgs
from apps.main.modules.ops import create_causal_mask
from lingua.transformer import (
    RMSNorm,
    BaseTransformerArgs,
)
from apps.offline_inf.data import AverageMeter

logger = logging.getLogger()


@dataclass
class PlanTransformerArgs(BaseTransformerArgs):

    seed: int = 42
    patch_size: int = 16
    in_channels: int = 3
    pre_trained_path: Optional[str] = None
    text_seqlen: int = 256
    vocab_size: int = -1
    attn_type: str = "causal"
    text_only: bool = True


@dataclass
class ModelArgs:
    plan_transformer: PlanTransformerArgs = field(default_factory=PlanTransformerArgs)
    vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)


class BaseLanguageTransformer(BasePlanTransformer):
    def __init__(self, args: PlanTransformerArgs):
        super().__init__(args)

    def forward(
        self,
        h,
        freqs_cis,
        attn_impl: str = "sdpa",
    ):
        seq_len = h.size(1)
        mask = create_causal_mask(seq_len, attn_impl)
        for idx, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, mask=mask, attn_impl=attn_impl)
        return h


class PlanTransformer(
    BaseLanguageTransformer
):  # TODO As planning model is not finished, we use a pure LLAMA model here, we will update this with the latest planning  model
    def __init__(self, args: PlanTransformerArgs):
        super().__init__(args)
        self.patch_size = args.patch_size
        self.text_seqlen = args.text_seqlen
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.rope_embeddings_cap = RotaryEmbedding1D(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.text_seqlen,
        )
        self.dim = args.dim
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        batch: dict[str:any],
        attn_impl: str = "sdpa",
    ):
        x_cap = self.tok_embeddings(batch["cap_token"])

        freqs_cis = self.rope_embeddings_cap.freqs_cis[: x_cap.size(1)]
        h = super().forward(x_cap, freqs_cis, attn_impl=attn_impl)

        return self.norm(h)

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        self.rope_embeddings_cap.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )


class OfflineInference(nn.Module):
    """
    OfflineInference Model.
    This model integrates a VAE for latent compression
    """

    VERSION: str = "v0.1"

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.compressor = LatentVideoVAE(args.vae)
        self.tokenizer = Tokenizer(model_path=args.tokenizer.model_path)
        self.plan_transformer = PlanTransformer(args.plan_transformer)

    def cap_pos_tokenize(self, batch: dict[str:any]) -> dict[str:any]:
        batch["cap_token"] = [
            self.tokenizer.encode(x, bos=True, eos=False) for x in batch["caption"]
        ]
        pad_id = self.tokenizer.pad_id
        bsz = len(batch["cap_token"])
        tokens = torch.full(
            (bsz, self.plan_transformer.text_seqlen),
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
    def forward(self, batch: dict[str, Any], inference_meters: Dict[str, AverageMeter]) -> dict[str, Any]:
        # Process text embedding
        start_time = time.time()
        batch = self.cap_pos_tokenize(batch)
        batch["text_embedding"] = self.plan_transformer(batch)
        inference_time = time.time() - start_time
        inference_meters["text_embedding"].update(inference_time, len(batch["text_embedding"]))

        # Process latent code
        image = batch["image"].cuda()
        start_time = time.time()
        latent_code = self.compressor.encode(image)
        inference_time = time.time() - start_time
        inference_meters["latent_code"].update(inference_time, len(latent_code))
        batch["latent_code"] = latent_code

        return batch

    def init_weights(self, args: ModelArgs):
        self.plan_transformer.init_weights(args.plan_transformer.pre_trained_path)
