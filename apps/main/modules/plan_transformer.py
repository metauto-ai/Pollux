# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from apps.main.modules.embedder import ImageEmbedder, TimestepEmbedder, LabelEmbedder
from apps.main.modules.ops import (
    # create_causal_mask,
    RotaryEmbedding1D,
    RotaryEmbedding2D,
)

from lingua.transformer import (
    BaseTransformerArgs,
    TransformerBlock,
    RMSNorm,
    InitStdFactor,
)

logger = logging.getLogger()


@dataclass
class PlanTransformerArgs(BaseTransformerArgs):

    seed: int = 42
    patch_size: int = 16
    in_channels: int = 3
    pre_trained_path: Optional[str] = None
    attn_type: str = (
        "multimodal"  # Options: 'multimodal', 'full', 'bi_causal' and 'causal'.
    )
    text_seqlen: int = 256
    gen_seqlen: int = 256
    vocab_size: int = -1
    text_only: bool = False


def create_multimodal_mask(
    seq_len_txt: int,
    seq_len_img1: int,
    seq_len_img2: int,
    attn_impl: str = "sdpa",
) -> torch.Tensor:

    # NOTE: text: `causal`, img1: `full`, img2: `full`

    total_len = seq_len_txt + seq_len_img1 + seq_len_img2
    mask = torch.zeros(total_len, total_len, dtype=torch.bool)  # 先全0

    # text attn: [0, seq_len_cap)
    for i in range(seq_len_txt):
        for j in range(seq_len_txt):
            if j <= i:
                mask[i, j] = True

    # img1 attn: [seq_len_cap, seq_len_cap + seq_len_img1)
    img1_start = seq_len_txt
    img1_end = seq_len_txt + seq_len_img1

    for i in range(img1_start, img1_end):
        for j in range(img1_start, img1_end):
            mask[i, j] = True

    # img2 attn: [img1_end, img1_end + seq_len_img2)
    img2_start = img1_end
    img2_end = img1_end + seq_len_img2

    for i in range(img2_start, img2_end):
        for j in range(img2_start, img2_end):
            mask[i, j] = True


@dataclass
class PlanTransformerArgs(BaseTransformerArgs):

    seed: int = 42
    patch_size: int = 16
    in_channels: int = 3
    pre_trained_path: Optional[str] = None
    attn_type: str = "bi_causal"  # Options: 'full', 'bi_causal' and 'causal'.
    text_seqlen: int = 256
    gen_seqlen: int = 256
    vocab_size: int = -1


class BasePlanTransformer(nn.Module):
    def __init__(self, args: PlanTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.attn_type = args.attn_type
        self.layers = nn.ModuleList()
        # assert not (self.attn_type == "bi_causal" and args.n_layers % 2 != 0)
        assert not (args.n_layers % 2 != 0)
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.text_only = args.text_only

    def forward(
        self,
        h,
        freqs_cis,
        attn_impl: str = "sdpa",
    ):

        if self.attn_type == "multimodal":
            mask = create_multimodal_mask(
                self.text_seqlen,
                self.gen_seqlen,
                self.gen_seqlen,
                attn_impl,
            )
        # elif self.attn_type in ["causal", "bi_causal"]:
        #     seqlen = h.size(1)
        #     mask = create_causal_mask(seqlen, attn_impl, None)
        elif self.attn_type == "full":
            mask = None
        else:
            raise NotImplementedError(f"Not support attention type: {self.attn_type}")

        for idx, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, mask=mask, attn_impl=attn_impl)
            # NOTE: no need for current planning transformer
            # if self.attn_type == "bi_causal":
            #      h = h.flip(1)
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


class PlanTransformer(BasePlanTransformer):
    def __init__(self, args: PlanTransformerArgs):
        super().__init__(args)

        self.text_seqlen = args.text_seqlen
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.rope_embeddings_text = RotaryEmbedding1D(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.text_seqlen,
        )

        self.patch_size = args.patch_size
        self.in_channels = args.in_channels

        # TODO: check wether need 2 image embedders
        self.img_embedder = ImageEmbedder(
            in_dim=self.patch_size * self.patch_size * args.in_channels,
            out_dim=args.dim,
        )
        self.gen_seqlen = args.gen_seqlen
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.rope_embeddings_image = RotaryEmbedding2D(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.gen_seqlen,
        )

        self.dim = args.dim
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.txt_head = nn.Linear(args.dim, args.vocab_size)

        # TODO: need to setup the vit_emb_dim and vae_emb_dim
        self.img_emb_head = nn.Linear(args.dim, args.vit_emb_dim)
        self.img_latent_head = nn.Linear(args.dim, args.vae_emb_dim)

    def patchify_and_embed_image(  # TODO: Rewrite to concat the mask
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], torch.Tensor]:

        self.rope_embeddings_image.freqs_cis = self.rope_embeddings_image.freqs_cis.to(
            x[0].device
        )
        pH = pW = self.patch_size
        B, C, H, W = x.size()
        x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
        x = self.img_embedder(x)
        x = x.flatten(1, 2)
        freqs_cis = self.rope_embeddings_image.freqs_cis[: H // pH, : W // pW].flatten(
            0, 1
        )
        return (
            x,
            (H, W),
            freqs_cis,
        )

    def forward(
        self,
        batch: dict[str:any],
        attn_impl: str = "sdpa",
    ):

        txt_emb = self.tok_embeddings(batch["cap_token"])
        freqs_cis_txt = self.rope_embeddings_cap.freqs_cis[: txt_emb.size(1)]

        # TODO: directly use patchify
        img_latent, _, freqs_cis_img_latent = self.patchify_and_embed_image(
            batch["masked_latent"]
        )
        freqs_cis_img_latent = freqs_cis_img_latent.to(img_latent.device)

        # TODO: only embedding
        img_emb, _, freqs_cis_img_emb = self.patchify_and_embed_image(
            batch["masked_latent"]
        )
        freqs_cis_img_emb = freqs_cis_img_emb.to(img_emb.device)

        input_tokens = torch.cat([txt_emb, img_emb, img_latent], dim=1)
        freqs_cis = torch.cat(
            [freqs_cis_txt, freqs_cis_img_emb, freqs_cis_img_latent], dim=0
        )

        h = super().forward(input_tokens, freqs_cis, attn_impl=attn_impl)
        h = self.norm(h)

        b = h.size(0)
        len_txt = txt_emb.size(1)
        len_img_emb = img_emb.size(1)
        len_img_latent = img_latent.size(1)

        h_txt = h[:, :len_txt]  # [b, len_txt, dim]
        h_img_emb = h[:, len_txt : len_txt + len_img_emb]  # [b, len_img_emb, dim]
        h_img_latent = h[:, len_txt + len_img_emb :]  # [b, len_img_latent, dim]

        text_logits = self.txt_head(h[:, : self.text_seqlen])
        img_emb_preds = self.img_emb_head(
            h[:, self.text_seqlen : self.text_seqlen + self.gen_seqlen]
        )
        img_latent_preds = self.img_latent_head(
            h[:, self.text_seqlen + self.gen_seqlen :]
        )

        layout = {
            "txt_emb": txt_emb.size(1),
            "img_emb": img_latent.size(1),
            "img_latent": img_latent.size(1),
        }

        return {
            "text_logits": text_logits,
            "img_emb_preds": img_emb_preds,
            "img_latent_preds": img_latent_preds,
            "layout": layout,
        }

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        self.rope_embeddings_image.reset_parameters()
        self.rope_embeddings_cap.reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        self.img_embedder.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
