# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import BlockMask
import torch.nn.functional as F
import random
from .component import (
    BaseTransformerArgs,
    RMSNorm,
    FeedForward,
    Attention,
    FlashAttention,
    InitStdFactor,
    RotaryEmbedding1D,
    RotaryEmbedding2D,
    AdaLN,
    ImageEmbedder,
    TimestepEmbedder,
    modulate,
    create_causal_mask,
)


logger = logging.getLogger()


@dataclass
class TransformerArgs(BaseTransformerArgs):
    seed: int = 42
    condition_dim: int = 512
    time_step_dim: int = 512
    patch_size: int = 2
    in_channels: int = 16
    out_channels: int = 16
    tmb_size: int = 256
    condition_seqlen: int = 1000
    gen_seqlen: int = 1000
    pre_trained_path: Optional[str] = None
    qk_norm: bool = True


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        # self.attention = Attention(
        #     dim=args.dim,
        #     head_dim=self.head_dim,
        #     n_heads=self.n_heads,
        #     n_kv_heads=self.n_kv_heads,
        #     rope_theta=args.rope_theta,
        #     qk_norm=args.qk_norm,
        # )
        self.attention = FlashAttention(
            dim=args.dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            qk_norm=args.qk_norm,
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
            in_dim=args.time_step_dim,
            out_dim=4 * args.dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
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
            x_mask,
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
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.gen_seqlen = args.gen_seqlen
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DiffusionTransformerBlock(args))

    def forward(
        self,
        h,
        h_mask,
        freqs_cis,
        modulation_signal: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ):
        for idx, layer in enumerate(self.layers):
            if modulation_signal == None:
                h = layer(h, h_mask, freqs_cis, mask=None, attn_impl=attn_impl)
            else:
                h = layer(
                    h, h_mask, freqs_cis, modulation_signal, mask=None, attn_impl=attn_impl
                )
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
    Diffusion Transformer capable of handling both images and video sequences (in the future).
    Uses patchify for images and a similar approach for video (flattening spatial and temporal dims).
    """

    def __init__(self, args: TransformerArgs):
        super().__init__(args)
        self.patch_size = args.patch_size
        self.out_channels = args.out_channels
        self.in_channels = args.in_channels
        self.tmb_embed = TimestepEmbedder(
            hidden_size=args.time_step_dim, time_embedding_size=args.tmb_size
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
        self.time_step_dim = args.time_step_dim
        self.dim = args.dim
        self.cos_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.coe_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.negative_token = nn.Parameter(torch.zeros(1, 1, args.condition_dim))
        self.cond_proj = nn.Linear(
            in_features=args.condition_dim,
            out_features=args.dim,
            bias=False,
        )

    def patchify_and_embed_image(
        self, 
        x: Union[torch.Tensor, List[torch.Tensor]],
        condition: torch.Tensor,
        condition_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int], torch.Tensor]:
        self.rope_embeddings_image.freqs_cis = self.rope_embeddings_image.freqs_cis.to(
            x[0].device
        )
        pH = pW = self.patch_size
        use_dynamic_res = isinstance(x, list)
        if use_dynamic_res:
            cond_l = condition_mask.sum(dim=1, dtype=torch.int32).tolist()
            max_cond_l = max(cond_l)
            bsz = len(x)
            H_list = [x[i].size(1) for i in range(bsz)]
            W_list = [x[i].size(2) for i in range(bsz)]
            H_max = max(H_list)
            W_max = max(W_list)
            max_seq_len = max_cond_l + (H_max // pH) * (W_max // pW)
            x_new = torch.zeros(bsz, max_seq_len, self.dim, dtype=x[0].dtype).to(x[0].device)
            x_mask = torch.zeros(bsz, max_seq_len, dtype=torch.bool).to(x[0].device)
            for i in range(bsz):
                _x = x[i]
                C, H, W = x[i].size()
                assert H % (pH) == 0, f"H should be divisible by {pH}, but now H = {H}."
                assert W % (pW) == 0, f"W should be divisible by {pW}, but now W = {W}."
                _x = _x.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 0, 2, 4).flatten(2)
                _x = self.img_embed(_x)
                _x = _x.flatten(0, 1)  # [H/16*W/16, D]

                x_new[i, :cond_l[i]] = condition[i, :cond_l[i]]     # TODO: assumes condition is right padded!
                x_new[i, cond_l[i]:cond_l[i] + (H // pH) * (W // pW)] = _x
                x_mask[i, :cond_l[i] + (H // pH) * (W // pW)] = True
            # rope embeddings
            freqs_cis_cond = self.rope_embeddings_conditions.freqs_cis[:max_cond_l].to(x[0].device)
            freqs_cis_img = self.rope_embeddings_image.freqs_cis[: H_max // pH, : W_max // pW].flatten(0, 1)
            freqs_cis = torch.cat([freqs_cis_cond, freqs_cis_img], dim=0)
            return x_new, x_mask, cond_l, (H_list, W_list), freqs_cis
        else:
            B, C, H, W = x.size()
            cond_l = condition.size(1)
            x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
            x = self.img_embed(x)
            x = x.flatten(1, 2)
            x_mask = torch.ones(B, (H // pH) * (W // pW), dtype=torch.bool).to(x.device)
            x_mask = torch.cat([condition_mask, x_mask], dim=1)
            freqs_cis_img = self.rope_embeddings_image.freqs_cis[: H // pH, : W // pW].flatten(
                0, 1
            )
            
            x = torch.cat([condition, x], dim=1)
            freqs_cis_cond = self.rope_embeddings_conditions.freqs_cis[:cond_l].to(x.device)
            freqs_cis = torch.cat([freqs_cis_cond, freqs_cis_img], dim=0)
            return (
                x,
                x_mask,
                cond_l,
                (H, W),
                freqs_cis,
            )

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        time_steps: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
        attn_impl: str = "sdpa",
    ):
        condition = self.cond_proj(condition)

        modulation_signal = self.tmb_embed(time_steps)

        x, x_mask, cond_l, img_size, freqs_cis = self.patchify_and_embed_image(x, condition, condition_mask)

        h = super().forward(x, x_mask, freqs_cis, modulation_signal, attn_impl=attn_impl)

        out = self.img_output(self.norm(h))

        x = self.unpatchify_image(out, cond_l, img_size)

        return x

    def unpatchify_image(
        self, x: torch.Tensor, cond_l: Union[List[int], int], img_size: Tuple[int, int]
    ) -> torch.Tensor:
        pH = pW = self.patch_size
        H, W = img_size
        use_dynamic_res = isinstance(H, list) and isinstance(W, list)
        if use_dynamic_res:
            out_x_list = []
            for i, (_H, _W) in enumerate(zip(H, W)):
                _x = x[i, cond_l[i]:cond_l[i] + (_H // pH) * (_W // pW)]
                _x = _x.view(_H // pH, _W // pW, pH, pW, -1)
                _x = _x.permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2)  # [16,H/8,W/8]
                out_x_list.append(_x)
            return out_x_list
        else:
            # Handle the case where cond_l is an integer, not a list
            if isinstance(cond_l, list):
                max_cond_l = max(cond_l)
            else:
                max_cond_l = cond_l
                
            B = x.size(0)
            L = (H // pH) * (W // pW)
            x = x[:, max_cond_l:max_cond_l + L].view(B, H // pH, W // pW, pH, pW, self.out_channels)
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
        nn.init.trunc_normal_(
            self.cond_proj.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        nn.init.normal_(self.cos_token, std=0.02)
        nn.init.normal_(self.coe_token, std=0.02)
        nn.init.normal_(self.negative_token, std=0.02)
