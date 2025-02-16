# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional
import logging
import random
import time
import torch
from torch import nn
import torch.nn.functional as F
from apps.main.modules.ops import (
    RotaryEmbedding1D,
)
import clip
from apps.main.modules.ops import create_causal_mask
from lingua.transformer import (
    RMSNorm,
    BaseTransformerArgs,
    TransformerBlock,
    InitStdFactor,
)

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
class CLIPArgs:
    config_name: str = "ViT-B/32"
    dtype: str = "bf16"
    text_seqlen: int = 77


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


class CLIP:
    def __init__(self, args: CLIPArgs):
        super().__init__()
        self.clip_model, _ = clip.load(args.config_name, jit=False)
        self.clip_model = self.clip_model.to(torch.float32).cuda()
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[args.dtype]

    def __call__(self, batch: dict[str:any]) -> torch.Tensor:
        assert "caption" in batch
        if isinstance(batch["caption"][0], tuple):
            batch["caption"] = [x[0] for x in batch["caption"]]
        for idx, x in enumerate(batch["caption"]):
            if not isinstance(x, str):
                logger.warning(f"Expected string but got {type(x)}: {x}")
                batch["caption"][idx] = ""
        text_tokens = clip.tokenize(batch["caption"], truncate=True).cuda()
        x = self.clip_model.token_embedding(text_tokens)
        x = x + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        return x.to(self.dtype)
