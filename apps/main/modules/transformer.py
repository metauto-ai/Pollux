# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

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
class DiffusionTransformerArgs(BaseTransformerArgs):

    seed: int = 42
    ada_dim: int = 512
    patch_size: int = 16
    in_channels: int = 3
    out_channels: int = 3
    tmb_size: int = 320
    cfg_drop_ratio: float = 0.1
    num_classes: int = 1000
    pre_trained_path: Optional[str] = None
    block_type: str = "dit"  # Options: 'dit', 'language_model'.
    # 1) 'dit': Embeds time using AdaIn.
    # 2) 'language_model': Embeds time into the sequence input using a special token.
    max_condition_seqlen: int = 2
    attn_type: str = "full"  # Options: 'full', 'bi_causal' and 'causal'.


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, args: DiffusionTransformerArgs):
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
        self.adaLN_modulation = AdaLN_Modulation(
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
    def __init__(self, args: DiffusionTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.block_type = args.block_type
        self.attn_type = args.attn_type
        self.layers = nn.ModuleList()
        if args.block_type == "dit":
            block = DiffusionTransformerBlock
        elif args.block_type == "language_model":
            block = TransformerBlock
        else:
            raise NotImplementedError(f"Not support {args.block_type}")
        assert not (self.attn_type == "bi_causal" and args.n_layers % 2 != 0)
        for _ in range(args.n_layers):
            self.layers.append(block(args))

    def forward(
        self,
        h,
        freqs_cis,
        modulation_signal: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ):
        assert (self.block_type == "language_model" and modulation_signal == None) or (
            self.block_type == "dit" and modulation_signal != None
        )

        if self.attn_type in ["causal", "bi_causal"]:
            seqlen = h.size(1)
            mask = create_causal_mask(seqlen, attn_impl, None)

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


class DiffusionTransformer(BaseDiffusionTransformer):
    """
    Diffusion Transformer capable of handling both images and video sequences.
    Uses patchify for images and a similar approach for video (flattening spatial and temporal dims).
    """

    def __init__(self, args: DiffusionTransformerArgs):
        super().__init__(args)
        self.patch_size = args.patch_size
        self.out_channels = args.out_channels
        self.in_channels = args.in_channels
        self.num_classes = args.num_classes
        if self.block_type == "language_model":
            assert args.dim == args.ada_dim
        self.tmb_embed = TimestepEmbedder(
            hidden_size=args.ada_dim, time_embedding_size=args.tmb_size
        )
        self.img_embed = ImageEmbedder(
            in_dim=self.patch_size * self.patch_size * args.in_channels,
            out_dim=args.dim,
        )
        self.cls_embed = LabelEmbedder(
            num_classes=args.num_classes,
            hidden_size=args.ada_dim,
            dropout_prob=args.cfg_drop_ratio,
        )
        self.img_output = nn.Linear(
            args.dim,
            self.patch_size * self.patch_size * args.out_channels,
            bias=False,
        )
        self.rope_embeddings_image = RotaryEmbedding2D(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
        )
        self.rope_embeddings_global = RotaryEmbedding1D(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_condition_seqlen,
        )
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

    def patchify_and_embed_video(
        self, v: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int], torch.Tensor]:
        """
        Patchify a video (B,T,C,H,W) into sequences of patches. We consider T*H/pH*W/pW patches.
        """
        self.rope_embeddings_image.freqs_cis = self.rope_embeddings_image.freqs_cis.to(
            v.device
        )
        pH = pW = self.patch_size
        B, T, C, H, W = v.size()
        # (B,T,C,H,W) -> (B,T,H//pH,W//pW,C*pH*pW)
        v = v.view(B, T, C, H // pH, pH, W // pW, pW)
        v = v.permute(0, 1, 3, 5, 2, 4, 6).flatten(-3)
        # (B,T,H//pH,W//pW, C*pH*pW) -> (B, T*(H//pH)*(W//pW), D)
        v = v.flatten(1, 3)
        v = self.img_embed(
            v
        )  # TODO here, we consider use a video embedder to embed video a patch size with 2x2x2.
        # v shape now (B, T*(H//pH)*(W//pW), D)
        # For freqs_cis, we consider spatial positions, time can be encoded in the modulation signal
        freqs_cis = self.rope_embeddings_image.freqs_cis[: H // pH, : W // pW].flatten(
            0, 1
        )
        return v, (T, H, W), freqs_cis

    def forward(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        condition: torch.Tensor,
        train: bool = True,
        attn_impl: str = "sdpa",
        is_video: bool = False,
    ):
        if not is_video:
            x, img_size, freqs_cis = self.patchify_and_embed_image(x)
        else:
            x, vid_size, freqs_cis = self.patchify_and_embed_video(x)

        freqs_cis = freqs_cis.to(x.device)
        t_emb = self.tmb_embed(time_steps)
        cls_emb = self.cls_embed(condition, train=train)
        if self.block_type == "dit":
            modulation_signal = t_emb + cls_emb
        elif self.block_type == "language_model":
            modulation_signal = None
            t_emb = t_emb.unsqueeze(1)
            cls_emb = cls_emb.unsqueeze(1)
            x = torch.cat([t_emb, cls_emb, x], dim=1)
            cond_freqs_cis = self.rope_embeddings_global.freqs_cis.to(freqs_cis.device)
            cond_freqs_cis = cond_freqs_cis[:2]
            freqs_cis = torch.cat([cond_freqs_cis, freqs_cis], dim=0)
        h = super().forward(x, freqs_cis, modulation_signal, attn_impl=attn_impl)

        out = self.img_output(self.norm(h))
        if self.block_type == "language_model":
            out = out[:, 2:, :]

        if not is_video:
            x = self.unpatchify_image(out, img_size)
        else:
            x = self.unpatchify_video(out, vid_size)
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

    def unpatchify_video(
        self, x: torch.Tensor, vid_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        pH = pW = self.patch_size
        T, H, W = vid_size
        B = x.size(0)
        L = (H // pH) * (W // pW) * T
        x = x[:, :L]
        x = x.view(B, T, H // pH, W // pW, pH, pW, self.out_channels)
        x = x.permute(0, 1, 6, 2, 4, 3, 5)
        x = x.reshape(B, T, self.out_channels, H, W)
        return x

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        self.rope_embeddings_image.reset_parameters()
        self.rope_embeddings_global.reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        self.tmb_embed.reset_parameters()
        self.img_embed.reset_parameters()
        self.cls_embed.reset_parameters()
        nn.init.trunc_normal_(
            self.img_output.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
