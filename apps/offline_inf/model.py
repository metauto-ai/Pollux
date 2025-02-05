# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import logging
import random
import time
import torch
from torch import nn
import torch.nn.functional as F
from apps.main.modules.vae import build_vae, LatentVideoVAEArgs
from apps.main.modules.ops import (
    RotaryEmbedding1D,
)
from apps.main.modules.tokenizer import Tokenizer, TokenizerArgs
from apps.main.modules.ops import create_causal_mask
from lingua.transformer import (
    RMSNorm,
    BaseTransformerArgs,
    TransformerBlock,
    InitStdFactor,
)
from apps.offline_inf.data import AverageMeter

import os

logger = logging.getLogger()


@dataclass
class LLAMATransformerArgs(BaseTransformerArgs):

    seed: int = 42
    patch_size: int = 16
    in_channels: int = 3
    pre_trained_path: Optional[str] = None
    text_seqlen: int = 256
    gen_seqlen: int = 256
    vocab_size: int = -1


@dataclass
class ModelArgs:
    plan_vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    gen_vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    text_encoder: LLAMATransformerArgs = field(default_factory=LLAMATransformerArgs)


class BaseTransformer(nn.Module):
    def __init__(self, args: LLAMATransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.layers = nn.ModuleList()
        assert not (args.n_layers % 2 != 0)
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

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

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        pass

    def init_weights(self, pre_trained_path: Optional[str] = None):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
        if pre_trained_path:
            assert os.path.exists(pre_trained_path)
            ckpt_state_dict = torch.load(pre_trained_path, map_location="cpu")
            target_state_dict = self.state_dict()
            filtered_state_dict = {
                k: v
                for k, v in ckpt_state_dict.items()
                if k in target_state_dict and v.shape == target_state_dict[k].shape
            }
            target_state_dict.update(filtered_state_dict)
            self.load_state_dict(target_state_dict)
            missing_keys = set(target_state_dict.keys()) - set(
                filtered_state_dict.keys()
            )
            unexpected_keys = set(ckpt_state_dict.keys()) - set(
                target_state_dict.keys()
            )
            logger.info(f"Load the checkpoints from {pre_trained_path}")
            logger.warning(f"Missing keys: {missing_keys}")
            logger.warning(f"Unexpected keys: {unexpected_keys}")


class LLAMA3(BaseTransformer):
    def __init__(self, args: LLAMATransformerArgs):
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

    VERSION: str = "v1.0"

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.gen_compressor = build_vae(args.gen_vae)
        self.tokenizer = Tokenizer(model_path=args.tokenizer.model_path)
        self.text_encoder = LLAMA3(args.text_encoder)
        self.plan_compressor = build_vae(args.plan_vae)

    def cap_pos_tokenize(self, batch: dict[str:any]) -> dict[str:any]:
        batch["cap_token"] = [
            self.tokenizer.encode(x, bos=True, eos=False) for x in batch["caption"]
        ]
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
