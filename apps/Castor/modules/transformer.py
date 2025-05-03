# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias, fmha

from .component import (AdaLN, Attention, BaseTransformerArgs, FeedForward,
                        FlashAttention, ImageEmbedder, InitStdFactor, RMSNorm,
                        RotaryEmbedding1D, RotaryEmbedding2D, TimestepEmbedder,
                        create_causal_mask, modulate_and_gate)

logger = logging.getLogger()

def nearest_multiple_of_8(x):
    return ((x + 7) // 8) * 8

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
    shared_adaLN: bool = False


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
            liger_rms_norm=args.liger_rms_norm,
            liger_rotary_emb=args.liger_rotary_emb,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            liger_ffn=args.liger_ffn,
        )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm)
        self.shared_adaLN = args.shared_adaLN
        if not args.shared_adaLN:
            self.adaLN_modulation = AdaLN(
                in_dim=args.time_step_dim,
                out_dim=4 * args.dim,
            )
        else:
            self.register_parameter(
                'modulation', 
                nn.Parameter(torch.randn(1, 4, args.dim) / args.dim**0.5)
            )


    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        modulation_signal: torch.Tensor,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
        modulation_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if modulation_values is None and not self.shared_adaLN:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                modulation_signal
            ).chunk(4, dim=1)
        else:
            scale_msa, gate_msa, scale_mlp, gate_mlp = (self.modulation + modulation_values).unbind(dim=1)

        h = x + modulate_and_gate(
            self.attention(
                self.attention_norm(x),
                x_mask,
                freqs_cis,
                tok_idx=None,
                mask=mask,
                attn_impl=attn_impl,
            ),
            scale=scale_msa,
            gate=gate_msa
        )

        h = h + modulate_and_gate(
            self.feed_forward(
                self.ffn_norm(h)
            ),
            scale=scale_mlp,
            gate=gate_mlp
        )

        return h

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()
        if not self.shared_adaLN:
            self.adaLN_modulation.reset_parameters()


@dataclass
class BaseDiffusionTransformerOutputs:
    output: torch.Tensor
    align_hidden_state: Optional[torch.Tensor] = None
    cond_l: Optional[List[int]] = None
    img_size: Optional[Tuple[int, int]] = None


class BaseDiffusionTransformer(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.gen_seqlen = args.gen_seqlen
        self.layers = nn.ModuleList()
        self.shared_adaLN = args.shared_adaLN
        for _ in range(args.n_layers):
            self.layers.append(DiffusionTransformerBlock(args))
        self.align_layer = args.align_layer

    def forward(
        self,
        h,
        h_mask,
        freqs_cis,
        modulation_signal: Optional[torch.Tensor] = None,
        modulation_values: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
    ):
        for idx, layer in enumerate(self.layers):
            h = layer(
                h,
                h_mask,
                freqs_cis,
                modulation_signal,
                mask=None,
                attn_impl=attn_impl,
                modulation_values=modulation_values,
            )
            if idx == self.align_layer - 1:
                align_hidden_state = h
        return h, align_hidden_state

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
        print("##########", args)
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
        self.norm = RMSNorm(args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm)
        self.cond_norm = RMSNorm(args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm)
        self.negative_token = nn.Parameter(torch.zeros(1, 1, args.condition_dim))
        self.cond_proj = nn.Linear(
            in_features=args.condition_dim,
            out_features=args.dim,
            bias=False,
        )
        if self.shared_adaLN:
            # Single AdaLN instance shared across all transformer blocks
            self.adaLN_modulation = AdaLN(
                in_dim=args.time_step_dim,
                out_dim=4 * args.dim,
            )

    def patchify_and_embed_image(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int], torch.Tensor]:
        self.rope_embeddings_image.freqs_cis = self.rope_embeddings_image.freqs_cis.to(
            x[0].device
        )
        pH = pW = self.patch_size
        use_dynamic_res = isinstance(x, list)
        if use_dynamic_res:
            cond_l = condition_mask.sum(dim=1, dtype=torch.int32).tolist()
            bsz = len(x)
            H_list = [x[i].size(1) for i in range(bsz)]
            W_list = [x[i].size(2) for i in range(bsz)]
            max_seq_len = max(
                [cond_l[i] + (H_list[i] // pH) * (W_list[i] // pW) for i in range(bsz)]
            )
            max_seq_len = nearest_multiple_of_8(max_seq_len)
            x_new = torch.zeros(bsz, max_seq_len, self.dim, dtype=x[0].dtype).to(
                x[0].device
            )
            x_mask = torch.zeros(bsz, max_seq_len, dtype=torch.bool).to(x[0].device)
            freqs_cis = torch.zeros(
                (
                    bsz,
                    max_seq_len,
                )
                + (self.rope_embeddings_conditions.freqs_cis.shape[-3:]),
                dtype=x[0].dtype,
            ).to(x[0].device)
            for i in range(bsz):
                _x = x[i]
                C, H, W = x[i].size()
                assert H % (pH) == 0, f"H should be divisible by {pH}, but now H = {H}."
                assert W % (pW) == 0, f"W should be divisible by {pW}, but now W = {W}."
                _x = (
                    _x.view(C, H // pH, pH, W // pW, pW)
                    .permute(1, 3, 2, 4, 0)
                    .flatten(2)
                )
                _x = self.img_embed(_x)
                _x = _x.flatten(0, 1)  # [H/16*W/16, D]
                x_new[i, : cond_l[i]] = condition[
                    i, : cond_l[i]
                ]  # TODO: assumes condition is right padded!
                x_new[i, cond_l[i] : cond_l[i] + (H // pH) * (W // pW)] = _x
                x_mask[i, : cond_l[i] + (H // pH) * (W // pW)] = True

                # rope embeddings
                freqs_cis[i, : cond_l[i]] = self.rope_embeddings_conditions.freqs_cis[
                    : cond_l[i]
                ].to(x[0].device)
                freqs_cis[i, cond_l[i] : cond_l[i] + (H // pH) * (W // pW)] = (
                    self.rope_embeddings_image.freqs_cis[: H // pH, : W // pW]
                    .flatten(0, 1)
                    .to(x[0].device)
                )
            return x_new, x_mask, cond_l, (H_list, W_list), freqs_cis
        else:
            B, C, H, W = x.size()
            condition = condition.to(x.dtype)
            assert H % pH == 0, f"H ({H}) should be divisible by pH ({pH})"
            assert W % pW == 0, f"W ({W}) should be divisible by pW ({pW})"


            # 1. Get actual condition lengths from mask
            cond_l = condition_mask.sum(dim=1, dtype=torch.int32) # Shape: [B]
            max_cond_l = cond_l.max(dim=0)[0].item()

            # 2. Patchify and embed image tensor
            x_img_patch = (
                x.view(B, C, H // pH, pH, W // pW, pW)
                .permute(0, 2, 4, 3, 5, 1)
                .flatten(3)
            )
            x_img_embed = self.img_embed(x_img_patch) # Assume img_embed preserves or outputs target_dtype
            x_img_embed = x_img_embed.flatten(1, 2) # Shape: [B, L_img, D]
            L_img = x_img_embed.size(1)

            # 4. Initialize combined tensors
            max_seq_len = nearest_multiple_of_8(max_cond_l + L_img)
            x_combined = torch.zeros(B, max_seq_len, self.dim, dtype=x.dtype, device=x.device)
            x_mask = torch.zeros(B, max_seq_len, dtype=torch.bool, device=x.device)
            freqs_cis = torch.zeros(
                (
                    B,
                    max_seq_len,
                )
                + (self.rope_embeddings_conditions.freqs_cis.shape[-3:]),
                dtype=x.dtype,
            ).to(x.device)

            # 5. Precompute RoPE embeddings
            freqs_cis_cond_all = self.rope_embeddings_conditions.freqs_cis[:max_cond_l].to(device=x.device, dtype=x.dtype)
            freqs_cis_img_all = self.rope_embeddings_image.freqs_cis[: H // pH, : W // pW].flatten(0, 1).to(device=x.device, dtype=x.dtype) # Shape [L_img, ..., D/2, 2]

            # 6. Populate tensors respecting actual condition lengths
            for i in range(B):
                # Place condition and image embeddings
                x_combined[i, :cond_l[i]] = condition[i, :cond_l[i]]
                x_combined[i, cond_l[i] : cond_l[i] + L_img] = x_img_embed[i]

                # Create mask
                x_mask[i, : cond_l[i] + L_img] = True

                # Create RoPE embeddings
                freqs_cis[i, :cond_l[i]] = freqs_cis_cond_all[:cond_l[i]]
                freqs_cis[i, cond_l[i] : cond_l[i] + L_img] = freqs_cis_img_all # Image RoPE is the same for all in batch here

            # Return list of actual lengths for consistency
            return (
                x_combined,
                x_mask,
                cond_l.tolist(),
                (H, W), # Return single H, W tuple
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
        condition = self.cond_norm(condition)

        modulation_signal = self.tmb_embed(time_steps)

        x_patched, x_mask, cond_l, img_size, freqs_cis = self.patchify_and_embed_image(
            x, condition, condition_mask
        )

        # Ensure modulation values match the patched data dtype if shared_adaLN is used
        if self.shared_adaLN:
            modulation_values = self.adaLN_modulation(
                modulation_signal
            ).unflatten(1, (4, self.dim))
        else:
            modulation_values = None

        last_hidden_state, align_hidden_state = super().forward(
            x_patched, x_mask, freqs_cis, modulation_signal, attn_impl=attn_impl, modulation_values=modulation_values
        )

        out = self.img_output(self.norm(last_hidden_state))
        out = self.unpatchify_image(out, cond_l, img_size) # unpatchify handles list/tensor output

        output = BaseDiffusionTransformerOutputs(
            output=out,
            align_hidden_state=align_hidden_state if align_hidden_state is not None else None,
            cond_l=cond_l,
            img_size=img_size
        )

        return output

    def unpatchify_image(
        self, x: torch.Tensor, cond_l: List[int], img_size: Union[Tuple[int, int], Tuple[List[int], List[int]]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Convert patched image features back to the original latent format.
        
        Args:
            image_features: Patched image features
            img_size: Original image size (H, W) or list of sizes for dynamic resolution
            
        Returns:
            Reconstructed image tensor(s) in shape [B, C, H, W] or list of [C, H, W] tensors
        """
        pH = pW = self.patch_size
        H, W = img_size
        use_dynamic_res = isinstance(H, list) and isinstance(W, list)
        
        if use_dynamic_res:
            out_x_list = []
            for i, (_H, _W) in enumerate(zip(H, W)):
                _x = x[i, cond_l[i] : cond_l[i] + (_H // pH) * (_W // pW)]
                _x = _x.view(_H // pH, _W // pW, pH, pW, -1)
                _x = (
                    _x.permute(4, 0, 2, 1, 3).flatten(3, 4).flatten(1, 2)
                )  # [16,H/8,W/8]
                out_x_list.append(_x)
            return out_x_list
        else:
            # Handle non-dynamic (tensor) case
            H, W = img_size
            L_img = (H // pH) * (W // pW)
            B = x.size(0)

            # Initialize output tensor
            # Note: Need to ensure self.out_channels is correctly defined/accessed
            out_features = torch.zeros(B, self.out_channels, H, W, dtype=x.dtype, device=x.device)

            for i in range(B):
                l_cond = cond_l[i]

                # Extract the image part for this batch item
                img_features_i = x[i, l_cond : l_cond + L_img] # Shape [L_img, D_out_patch]

                # Reshape back to image format
                # D_out_patch = pH * pW * self.out_channels must hold
                img_features_i = img_features_i.view(
                    H // pH, W // pW, pH, pW, self.out_channels
                )
                img_features_i = img_features_i.permute( 5, 1, 3, 2, 4).flatten(3, 4).flatten(2, 3) # [C_out, H, W]
                out_features[i] = img_features_i

            return out_features

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        self.rope_embeddings_image.reset_parameters()
        self.rope_embeddings_conditions.reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        self.cond_norm.reset_parameters()
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
        nn.init.normal_(self.negative_token, std=0.02)
        if self.shared_adaLN:
            self.adaLN_modulation.reset_parameters()
