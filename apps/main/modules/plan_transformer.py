# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask


from xformers.ops import fmha, AttentionBias
from lingua.transformer import (
    BaseTransformerArgs,
    TransformerBlock,
    RMSNorm,
    FeedForward,
    Attention,
    AdaLN_Modulation,
    InitStdFactor,
    TimestepEmbedder,
    ImageEmbedder,
    LabelEmbedder,
    modulate,
)
from lingua.transformer import RotaryEmbedding as RotaryEmbedding1D
import os
import logging

logger = logging.getLogger()


def precompute_2d_freqs_cls(
    dim: int,
    end: int,
    theta: float = 10000.0,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with
    given dimensions.

    This function calculates a frequency tensor with complex exponentials
    using the given dimension 'dim' and the end index 'end'. The 'theta'
    parameter scales the frequencies. The returned tensor contains complex
    values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation.
            Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex
            exponentials.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    freqs_cis = torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)

    freqs_cis_h = freqs_cis.view(end, 1, dim // 4, 2, 2).repeat(1, end, 1, 1, 1)
    freqs_cis_w = freqs_cis.view(1, end, dim // 4, 2, 2).repeat(end, 1, 1, 1, 1)
    freqs_cis = torch.cat([freqs_cis_h, freqs_cis_w], dim=2)

    return freqs_cis


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if sliding_window is not None and attn_impl == "xformers":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, False
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class RotaryEmbedding2D(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_2d_freqs_cls(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_2d_freqs_cls(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


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
        assert not (self.attn_type == "bi_causal" and args.n_layers % 2 != 0)
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        h,
        freqs_cis,
        attn_impl: str = "sdpa",
    ):

        if self.attn_type in ["causal", "bi_causal"]:
            seqlen = h.size(1)
            mask = create_causal_mask(seqlen, attn_impl, None)

        elif self.attn_type == "full":
            mask = None
        else:
            raise NotImplementedError(f"Not support attention type: {self.attn_type}")

        for idx, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, mask=mask, attn_impl=attn_impl)
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


class PlanTransformer(BasePlanTransformer):
    def __init__(self, args: PlanTransformerArgs):
        super().__init__(args)
        self.patch_size = args.patch_size
        self.text_seqlen = args.text_seqlen
        self.in_channels = args.in_channels
        self.img_embed = ImageEmbedder(
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
        self.rope_embeddings_cap = RotaryEmbedding1D(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.text_seqlen,
        )
        self.dim = args.dim
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

    def patchify_and_embed_image(  # TODO: Rewrite to concat the mask
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
        batch: dict[str:any],
        attn_impl: str = "sdpa",
    ):
        x_cap = self.tok_embeddings(batch["cap_token"])
        x_img, _, freqs_cis_img = self.patchify_and_embed_image(batch["masked_latent"])

        freqs_cis_img = freqs_cis_img.to(x_img.device)
        freqs_cis_cap = self.rope_embeddings_cap.freqs_cis[: x_cap.size(1)]
        x = torch.cat([x_cap, x_img], dim=1)
        freqs_cis = torch.cat([freqs_cis_cap, freqs_cis_img], dim=0)
        h = super().forward(x, freqs_cis, attn_impl=attn_impl)
        layout = {
            "cap": x_cap.size(1),
            "img": x_img.size(1),
        }
        return self.norm(h), layout

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        self.rope_embeddings_image.reset_parameters()
        self.rope_embeddings_cap.reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        self.img_embed.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
