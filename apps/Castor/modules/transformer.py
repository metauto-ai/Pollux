# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from xformers.ops import AttentionBias, fmha

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from .component import (
    AdaLN,
    BaseTransformerArgs,
    FeedForward,
    FlashAttention,
    ImageEmbedder,
    InitStdFactor,
    RMSNorm,
    RotaryEmbedding1D,
    RotaryEmbedding2D,
    TimestepEmbedder,
    create_causal_mask,
    modulate_and_gate,
    nearest_multiple_of_8,
    modulate_and_gate_unpadded,
)
from flash_attn.bert_padding import unpad_input, pad_input
from apps.Castor.utils.pad import pad_flat_tokens_to_multiple, unpad_flat_tokens
import copy

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
    shared_adaLN: bool = False
    attention_window: Tuple[int, int] = (-1, -1)
    full_attention_layers: Optional[List[int]] = None
    unpadded: bool = False
    fp8_ffn_skip_layers: Optional[List[int]] = None


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (args.n_heads is not None), (
            "Should specify at least head_dim or n_heads"
        )
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        self.unpadded = args.unpadded
        if self.unpadded:
            self.modulate_and_gate = modulate_and_gate_unpadded
        else:
            self.modulate_and_gate = modulate_and_gate

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
            window_size=args.attention_window,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            liger_ffn=args.liger_ffn,
            use_fp8_ffn=args.use_fp8_ffn,
            rms_eps = args.norm_eps
        )

        self.attention_norm = RMSNorm(
            args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm
        )

        # fp8 ffn does not need normalization layer
        self.ffn_norm = (
            nn.Identity()
            if args.use_fp8_ffn
            else RMSNorm(
                args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm
            )
        )

        self.sandwich_norm = (
            RMSNorm(args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm)
            if args.sandwich_norm
            else nn.Identity()
        )

        self.shared_adaLN = args.shared_adaLN
        if not args.shared_adaLN:
            self.adaLN_modulation = AdaLN(
                in_dim=args.time_step_dim,
                out_dim=4 * args.dim,
            )
        else:
            self.register_parameter(
                "modulation", nn.Parameter(torch.randn(1, 4, args.dim) / args.dim**0.5)
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
        cu_seqlens: Optional[torch.Tensor] = None,
        batch_indices: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        if modulation_values is None and not self.shared_adaLN:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                modulation_signal
            ).chunk(4, dim=1)
        elif self.unpadded:
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.modulation + modulation_values
            )[batch_indices].unbind(dim=1)
        else:
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.modulation + modulation_values
            ).unbind(dim=1)

        h = x + self.modulate_and_gate(
            self.attention(
                self.attention_norm(x),
                x_mask,
                freqs_cis,
                tok_idx=None,
                mask=mask,
                attn_impl=attn_impl,
                cu_seqlens=cu_seqlens,
                batch_indices=batch_indices,
                max_seqlen=max_seqlen,
            ),
            scale=scale_msa,
            gate=gate_msa,
        )

        h = h + self.sandwich_norm(
            self.modulate_and_gate(
                self.feed_forward(self.ffn_norm(h)), scale=scale_mlp, gate=gate_mlp
            )
        )

        return h

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()
        self.feed_forward.reset_parameters(init_std, factor)
        if not isinstance(self.ffn_norm, nn.Identity):
            self.ffn_norm.reset_parameters()
        if not isinstance(self.sandwich_norm, nn.Identity):
            self.sandwich_norm.reset_parameters()
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

        if args.full_attention_layers is None:
            args.full_attention_layers = list(range(args.n_layers))

        for i in range(args.n_layers):
            layer_args = copy.deepcopy(args)
            if i in args.full_attention_layers:
                layer_args.attention_window = (-1, -1)
            if i in args.fp8_ffn_skip_layers:
                layer_args.liger_ffn = True
                layer_args.use_fp8_ffn = False
            self.layers.append(DiffusionTransformerBlock(layer_args))
        self.align_layer = args.align_layer

    def forward(
        self,
        h,
        h_mask,
        freqs_cis,
        modulation_signal: Optional[torch.Tensor] = None,
        modulation_values: Optional[torch.Tensor] = None,
        attn_impl: str = "sdpa",
        cu_seqlens: Optional[torch.Tensor] = None,
        batch_indices: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
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
                cu_seqlens=cu_seqlens,
                batch_indices=batch_indices,
                max_seqlen=max_seqlen,
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
        self.unpadded = args.unpadded
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
        self.norm = RMSNorm(
            args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm
        )
        self.cond_norm = RMSNorm(
            args.dim, eps=args.norm_eps, liger_rms_norm=args.liger_rms_norm
        )
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
        pad: Optional[bool] = True,
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
            # max_seq_len = nearest_multiple_of_8(max_seq_len)
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
            target_dtype = x.dtype
            cond_l = condition.size(1)

            # 1. Patchify and embed image tensor
            x_img_patch = (
                x.view(B, C, H // pH, pH, W // pW, pW)
                .permute(0, 2, 4, 3, 5, 1)
                .flatten(3)
            )
            x_img_embed = self.img_embed(x_img_patch)
            x_img_embed = x_img_embed.flatten(1, 2)  # Shape: [B, L_img, D]

            # Ensure image embedding has the target dtype
            x_img_embed = x_img_embed.to(target_dtype)

            # 2. Prepare condition tensor
            condition = condition.to(target_dtype)  # Shape: [B, L_cond, D]

            # 3. Calculate total sequence length and pad to multiple of 8
            img_l = (H // pH) * (W // pW)
            total_len = cond_l + img_l
            if pad:
                padded_len = nearest_multiple_of_8(total_len)
            else:
                padded_len = total_len

            # 4. Initialize tensors with padded length
            embed_dim = condition.size(-1)
            x_combined = torch.zeros(
                (B, padded_len, embed_dim), dtype=target_dtype, device=x.device
            )
            x_mask = torch.zeros((B, padded_len), dtype=torch.bool, device=x.device)

            # 5. Copy tensors into the initialized containers
            x_combined[:, :cond_l] = condition
            x_combined[:, cond_l : cond_l + img_l] = x_img_embed

            x_mask[:, :cond_l] = condition_mask
            x_mask[:, cond_l : cond_l + img_l] = True

            # 6. Prepare RoPE embeddings
            freqs_cis_cond = self.rope_embeddings_conditions.freqs_cis[:cond_l].to(
                device=x.device, dtype=target_dtype
            )
            freqs_cis_img = (
                self.rope_embeddings_image.freqs_cis[: H // pH, : W // pW]
                .flatten(0, 1)
                .to(device=x.device, dtype=target_dtype)
            )

            # Initialize padded freqs_cis
            rope_dim = freqs_cis_cond.shape[-2:]
            freqs_cis_shape = list(
                freqs_cis_cond.shape[1:-2]
            )  # Get middle dimensions if any
            freqs_cis = torch.zeros(
                (padded_len, *freqs_cis_shape, *rope_dim),
                dtype=target_dtype,
                device=x.device,
            )

            # Copy RoPE embeddings
            freqs_cis[:cond_l] = freqs_cis_cond
            freqs_cis[cond_l : cond_l + img_l] = freqs_cis_img

            # Expand for batch dimension
            freqs_cis = freqs_cis.unsqueeze(0).expand(
                B, -1, *([-1] * (freqs_cis.dim() - 1))
            )

            return (
                x_combined,
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
        flops_meter=None,
    ):
        condition = self.cond_proj(condition)
        condition = self.cond_norm(condition)

        modulation_signal = self.tmb_embed(time_steps)

        x_patched, x_mask, cond_l, img_size, freqs_cis = self.patchify_and_embed_image(
            x, condition, condition_mask, pad=not self.unpadded
        )

        if flops_meter is not None:
            flops_meter.log_diffusion_flops(x_patched.shape)

        if self.unpadded:
            B, S, D = x_patched.shape
            x_patched, indices, cu_seqlens, max_seqlen, used_seqlens = unpad_input(
                x_patched, x_mask
            )
            seqlens = torch.sum(x_mask, dim=-1, dtype=torch.int32)
            freqs_cis, _, _, _, _ = unpad_input(freqs_cis, x_mask)
            (
                x_patched,
                freqs_cis,
                seqlens,
                cu_seqlens,
                max_seqlen,
                modulation_signal,
                flat_pad_len,
            ) = pad_flat_tokens_to_multiple(
                x_patched,
                freqs_cis,
                seqlens,
                cu_seqlens,
                max_seqlen,
                modulation_signal,
                multiple=128,
            )
            batch_indices = torch.arange(
                seqlens.shape[0], device=x_patched.device, dtype=torch.int32
            ).repeat_interleave(repeats=seqlens)
            x_mask = None
        else:
            batch_indices = None
            cu_seqlens = None
            max_seqlen = None

        # Ensure modulation values match the patched data dtype if shared_adaLN is used
        if self.shared_adaLN:
            modulation_values = self.adaLN_modulation(modulation_signal).unflatten(
                1, (4, self.dim)
            )
        else:
            modulation_values = None

        last_hidden_state, align_hidden_state = super().forward(
            x_patched,
            x_mask,
            freqs_cis,
            modulation_signal,
            attn_impl=attn_impl,
            modulation_values=modulation_values,
            cu_seqlens=cu_seqlens,
            batch_indices=batch_indices,
            max_seqlen=max_seqlen,
        )

        out = self.img_output(self.norm(last_hidden_state))
        if self.unpadded:
            out = unpad_flat_tokens(out, flat_pad_len)
            align_hidden_state = unpad_flat_tokens(align_hidden_state, flat_pad_len)
            out = pad_input(out, indices, B, S)
            align_hidden_state = pad_input(align_hidden_state, indices, B, S)
        out = self.unpatchify_image(
            out, cond_l, img_size
        )  # unpatchify handles list/tensor output

        output = BaseDiffusionTransformerOutputs(
            output=out,
            align_hidden_state=align_hidden_state
            if align_hidden_state is not None
            else None,
            cond_l=cond_l,
            img_size=img_size,
        )

        return output

    def unpatchify_image(
        self, x: torch.Tensor, cond_l: Union[List[int], int], img_size: Tuple[int, int]
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
            # Handle the case where cond_l is an integer, not a list
            if isinstance(cond_l, list):
                # If cond_l is unexpectedly a list here, take the first element or max?
                # Assuming it should be consistent with the input type logic.
                # If input was Tensor, cond_l should be int. If list, list.
                # This path assumes input was Tensor, so cond_l should be int.
                # If it's a list, maybe log a warning or error.
                # For now, assume it's an int if we reach here.
                if len(cond_l) > 0:
                    max_cond_l = cond_l[0]  # Or max(cond_l) if that makes sense
                    # logger.warning("cond_l was a list in unpatchify_image tensor path.")
                else:
                    max_cond_l = 0  # Handle empty list case
            else:
                max_cond_l = cond_l  # It's already an int

            B = x.size(0)
            L = (H // pH) * (W // pW)
            # Ensure slicing indices are correct
            img_features = x[:, max_cond_l : max_cond_l + L]

            # Ensure the view dimensions match the extracted features
            # B, L_img, D_out_patch = img_features.shape
            # D_out_patch should be pH * pW * self.out_channels
            # L_img should be (H // pH) * (W // pW)
            img_features = img_features.view(
                B, H // pH, W // pW, pH, pW, self.out_channels
            )
            img_features = (
                img_features.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
            )
            return img_features

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
