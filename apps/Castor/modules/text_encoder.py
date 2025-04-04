# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Tuple
import logging
import torch
import clip


import os

logger = logging.getLogger()


@dataclass
class CLIPArgs:
    config_name: str = "ViT-B/32"
    dtype: str = "bf16"
    text_seqlen: int = 77


class CLIP:
    def __init__(self, args: CLIPArgs):
        super().__init__()
        self.clip_model, _ = clip.load(args.config_name, jit=False)
        self.clip_model = self.clip_model.to(torch.float32).cuda()
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[args.dtype]

    def __call__(self, batch: dict[str:any]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert "caption" in batch
        if isinstance(batch["caption"][0], tuple):
            batch["caption"] = [x[0] for x in batch["caption"]]
        for idx, x in enumerate(batch["caption"]):
            if not isinstance(x, str):
                logger.warning(f"Expected string but got {type(x)}: {x}")
                batch["caption"][idx] = ""
        text_tokens = clip.tokenize(batch["caption"], truncate=True).cuda()
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (text_tokens != 0).to(self.dtype)

        x = self.clip_model.token_embedding(text_tokens)
        x = x + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)

        return x.to(self.dtype), attention_mask

