# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from lingua.transformer import (
    BaseTransformerArgs,
    TransformerBlock,
    RMSNorm,
    FeedForward,
    Attention,
    InitStdFactor,
)
from apps.main.modules.ops import (
    RotaryEmbedding1D,
    RotaryEmbedding2D,
    AdaLN,
    modulate,
    create_causal_mask,
)
from apps.main.modules.embedder import ImageEmbedder, TimestepEmbedder


logger = logging.getLogger()


@dataclass
class GenTransformerArgs(BaseTransformerArgs):

    seed: int = 42
    ada_dim: int = 512
    patch_size: int = 16
    in_channels: int = 3
    out_channels: int = 3
    tmb_size: int = 320
    condition_seqlen: int = 1000
    gen_seqlen: int = 1000
    pre_trained_path: Optional[str] = None
    attn_type: str = "full"  # Options: 'full', 'bi_causal' and 'causal'.


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, args: GenTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.adaLN_modulation = AdaLN(
            in_dim=args.ada_dim,
            out_dim=4 * args.dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        modulation_signal: torch.Tensor,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
            modulation_signal
        ).chunk(4, dim=1)
        h = x + gate_msa.unsqueeze(1).tanh() * self.attention(
            modulate(self.attention_norm(x), scale_msa),
            freqs_cis,
            tok_idx=None,
            mask=mask,
            attn_impl=attn_impl,
        )
        out = h + gate_mlp.unsqueeze(1).tanh() * self.feed_forward(
            modulate(self.ffn_norm(h), scale_mlp)
        )
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.adaLN_modulation.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class BaseDiffusionTransformer(nn.Module):
    def __init__(self, args: GenTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.gen_seqlen = args.gen_seqlen
        self.attn_type = args.attn_type
        self.layers = nn.ModuleList()
        assert not (self.attn_type == "bi_causal" and args.n_layers % 2 != 0)
        for _ in range(args.n_layers):
            self.layers.append(DiffusionTransformerBlock(args))

    def forward(
        self,
        h,
        freqs_cis,
        modulation_signal: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ):

        if self.attn_type in ["causal", "bi_causal"]:
            seqlen = h.size(1)
            mask = create_causal_mask(seqlen, attn_impl)

        elif self.attn_type == "full":
            mask = None
        else:
            raise NotImplementedError(f"Not support attention type: {self.attn_type}")

        for idx, layer in enumerate(self.layers):
            if modulation_signal == None:
                h = layer(h, freqs_cis, mask=mask, attn_impl=attn_impl)
            else:
                h = layer(
                    h, freqs_cis, modulation_signal, mask=mask, attn_impl=attn_impl
                )
            if self.attn_type == "bi_causal":
                h = h.flip(1)
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


class GenTransformer(BaseDiffusionTransformer):
    """
    Diffusion Transformer capable of handling both images and video sequences (in the future).
    Uses patchify for images and a similar approach for video (flattening spatial and temporal dims).
    """

    def __init__(self, args: GenTransformerArgs):
        super().__init__(args)
        self.patch_size = args.patch_size
        self.out_channels = args.out_channels
        self.in_channels = args.in_channels
        self.tmb_embed = TimestepEmbedder(
            hidden_size=args.ada_dim, time_embedding_size=args.tmb_size
        )
        self.img_embed = ImageEmbedder(
            in_dim=self.patch_size * self.patch_size * args.in_channels,
            out_dim=args.dim,
        )
        self.img_output = nn.Linear(
            args.dim,
            self.patch_size * self.patch_size * args.out_channels,
            bias=False,
        )
        self.rope_embeddings_image = RotaryEmbedding2D(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.gen_seqlen,
        )
        self.rope_embeddings_conditions = RotaryEmbedding1D(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.condition_seqlen,
        )
        self.ada_dim = args.ada_dim
        self.dim = args.dim
        # [BEGING_OF_TEXT, END_OF_TEXT, BEGIN_OF_IMAGE, END_OF_IMAGE]
        self.bot_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.eot_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        # self.cap_cond_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

    def patchify_and_embed_image(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], torch.Tensor]:
        self.rope_embeddings_image.freqs_cis = self.rope_embeddings_image.freqs_cis.to(
            x[0].device
        )
        pH = pW = self.patch_size
        B, C, H, W = x.size()
        x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
        x = self.img_embed(x)
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
        x: torch.Tensor,
        time_steps: torch.Tensor,
        condition: torch.Tensor,
        # layout: Dict[str, int],
        attn_impl: str = "sdpa",
    ):
        x, img_size, freqs_cis_img = self.patchify_and_embed_image(x)
        x_l = x.size(1)
        modulation_signal = self.tmb_embed(time_steps)

        # indicator = torch.cat(
        #     [
        #         torch.cat([self.cap_cond_token] * layout["cap"], dim=1),
        #         torch.cat([self.img_cond_token] * layout["img"], dim=1),
        #     ],
        #     dim=1,
        # )
        # condition = condition + indicator

        condition_ = torch.cat(
            [
                self.bot_token.repeat(len(condition), 1, 1),
                condition.unsqueeze(1),
                self.eot_token.repeat(len(condition), 1, 1),
            ],
            dim=1,
        )

        c_l = condition_.size(1)

        freqs_cis_img = freqs_cis_img.to(x.device)
        freqs_cis_cond = self.rope_embeddings_conditions.freqs_cis[:c_l].to(x.device)
        x = torch.cat([condition_, x], dim=1)
        freqs_cis = torch.cat([freqs_cis_cond, freqs_cis_img], dim=0)

        h = super().forward(x, freqs_cis, modulation_signal, attn_impl=attn_impl)

        h = h[:, -x_l:, :]

        out = self.img_output(self.norm(h))

        x = self.unpatchify_image(out, img_size)

        return x

    def unpatchify_image(
        self, x: torch.Tensor, img_size: Tuple[int, int]
    ) -> torch.Tensor:
        pH = pW = self.patch_size
        H, W = img_size
        B = x.size(0)
        L = (H // pH) * (W // pW)
        x = x[:, :L].view(B, H // pH, W // pW, pH, pW, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
        return x

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        self.rope_embeddings_image.reset_parameters()
        self.rope_embeddings_conditions.reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        self.tmb_embed.reset_parameters()
        self.img_embed.reset_parameters()
        nn.init.trunc_normal_(
            self.img_output.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        # nn.init.normal_(self.img_cond_token, std=0.02)
        # nn.init.normal_(self.cap_cond_token, std=0.02)
        nn.init.normal_(self.bot_token, std=0.02)
        nn.init.normal_(self.eot_token, std=0.02)
