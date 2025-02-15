from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import logging
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import LlamaTokenizerFast

from apps.main.modules.embedder import LabelEmbedder
from apps.main.modules.gen_transformer import (
    RotaryEmbedding1D,
    RotaryEmbedding2D,
)
from lingua.transformer import (
    BaseTransformerArgs,
    Attention,
    RMSNorm,
    InitStdFactor,
    FeedForward,
    BlockMask,
    AttentionBias,
    fmha,
    flex_attention_comp,
    repeat_kv,
    apply_rotary_emb,
    cross_entropy,
)
import random

logger = logging.getLogger()


@dataclass
class LatentProjecterArgs:
    latent_dim: int = 16
    output_dim: int = 3072
    patchify_size: int = 1


@dataclass
class TokenizerArgs:
    model_name: str = "/jfs/checkpoints/Llama-3.2-3B"


@dataclass
class LlamaArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"
    gen_seqlen: int = 256
    condition_seqlen: int = 320
    text_seqlen: int = 128
    vocab_size: int = 128256
    pre_trained_path: Optional[str] = None
    from_llama: bool = False


@dataclass
class ModelArgs:
    latent_projector: LatentProjecterArgs = field(default_factory=LatentProjecterArgs)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    llm: LlamaArgs = field(default_factory=LlamaArgs)
    text_cfg_ratio: float = 0.1
    image_cfg_ratio: float = 0.1
    codebook_size: int = 512
    random_rate: Optional[float] = None


class PlanTransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention_text = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward_text = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm_text = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm_text = RMSNorm(args.dim, eps=args.norm_eps)

        self.attention_visual = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )

        self.feed_forward_visual = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm_visual = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm_visual = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        text_l: int,
        visual_l: int,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        assert text_l + visual_l == x.size(1)
        text_x = x[:, :text_l]
        visual_x = x[:, text_l:]
        text_xq, text_xk, text_xv = self.text_op_before_attention(text_x)
        visual_xq, visual_xk, visual_xv = self.visual_op_before_attention(visual_x)
        xq = torch.cat([text_xq, visual_xq], dim=1)
        xk = torch.cat([text_xk, visual_xk], dim=1)
        xv = torch.cat([text_xv, visual_xv], dim=1)
        x_out = self.fused_attention(
            x=x,
            freq_cis=freq_cis,
            xq=xq,
            xk=xk,
            xv=xv,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )

        text_h = self.text_op_after_attention(pre_x=text_x, h=x_out[:, :text_l])
        visual_h = self.visual_op_after_attention(pre_x=visual_x, h=x_out[:, text_l:])
        return torch.cat([text_h, visual_h], dim=1)

    def fused_attention(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ):
        assert self.attention_text.n_heads == self.attention_visual.n_heads
        assert self.attention_text.n_kv_heads == self.attention_visual.n_kv_heads
        assert self.attention_text.head_dim == self.attention_visual.head_dim
        assert (
            self.attention_text.heads_per_group == self.attention_visual.heads_per_group
        )
        bsz, seq_len, dim = x.shape
        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(
            bsz, seq_len, self.attention_text.n_heads, self.attention_text.head_dim
        )
        xk = xk.view(
            bsz, seq_len, self.attention_text.n_kv_heads, self.attention_text.head_dim
        )
        xv = xv.view(
            bsz, seq_len, self.attention_text.n_kv_heads, self.attention_text.head_dim
        )

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):  # Don't support visual branch kv cache
            xk, xv = self.attention_text.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.attention_text.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.attention_text.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        return output.reshape(output_shape)

    @torch.no_grad()
    def text_op_before_attention(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.attention_norm_text(x)
        xq = self.attention_text.wq(x.view_as(x))
        xk = self.attention_text.wk(x.view_as(x))
        xv = self.attention_text.wv(x.view_as(x))
        return xq, xk, xv

    @torch.no_grad()
    def text_op_after_attention(
        self, pre_x: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        h = pre_x + self.attention_text.wo(h)
        out = h + self.feed_forward_text(self.ffn_norm_text(h))
        return out

    def visual_op_before_attention(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.attention_norm_visual(x)
        xq = self.attention_visual.wq(x.view_as(x))
        xk = self.attention_visual.wk(x.view_as(x))
        xv = self.attention_visual.wv(x.view_as(x))
        return xq, xk, xv

    def visual_op_after_attention(
        self, pre_x: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        h = pre_x + self.attention_visual.wo(h)
        out = h + self.feed_forward_visual(self.ffn_norm_visual(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention_text.reset_parameters(init_std, factor)
        self.attention_norm_text.reset_parameters()
        self.attention_visual.reset_parameters(init_std, factor)
        self.attention_norm_visual.reset_parameters()

        self.feed_forward_text.reset_parameters(init_std, factor)
        self.ffn_norm_text.reset_parameters()
        self.feed_forward_visual.reset_parameters(init_std, factor)
        self.ffn_norm_visual.reset_parameters()


class PlanTransformer(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(PlanTransformerBlock(args))

    def get_attention_mask(self, seq_len, text_l):
        """
        Generates an attention mask where:
        - Tokens before `text_l` use causal (autoregressive) attention.
        - Tokens from `text_l` onwards use bidirectional attention.

        Args:
            seq_len (int): Total sequence length.
            text_l (int): The index separating causal and bidirectional attention.

        Returns:
            torch.Tensor: The attention mask of shape (seq_len, seq_len).
        """
        # Causal attention mask (lower triangular)
        causal_mask = torch.tril(torch.ones(text_l, text_l))

        # Bidirectional attention mask (all ones)
        bidirectional_mask = torch.ones(seq_len - text_l, seq_len - text_l)

        # Zero mask for cross attention (causal tokens cannot see bidirectional tokens)
        cross_mask = torch.zeros(text_l, seq_len - text_l)

        # Concatenating the masks
        upper_part = torch.cat([causal_mask, cross_mask], dim=1)
        lower_part = torch.cat(
            [torch.ones(seq_len - text_l, text_l), bidirectional_mask], dim=1
        )

        # Final mask
        attn_mask = torch.cat([upper_part, lower_part], dim=0)

        return attn_mask

    def forward(self, h, text_l, visual_l, freq_cis, attn_impl: str = "sdpa"):
        assert text_l + visual_l == h.size(1)
        mask = self.get_attention_mask(h.size(1), text_l).to(h.device)
        for layer in self.layers:
            h = layer(
                h,
                text_l=text_l,
                visual_l=visual_l,
                freq_cis=freq_cis,
                mask=mask,
                attn_impl=attn_impl,
            )
        return h

    def init_weights(self, pre_trained_path: Optional[str] = None, from_llama=False):
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

            if from_llama:
                filtered_state_dict = {}
                for k, v in target_state_dict.items():
                    if "_text" in k:
                        k_remove = k.replace("_text", "")
                        if (
                            k_remove in ckpt_state_dict
                            and v.shape == ckpt_state_dict[k_remove].shape
                        ):
                            filtered_state_dict[k] = ckpt_state_dict[k_remove]
                    elif "_visual" in k:
                        k_remove = k.replace("_visual", "")
                        if (
                            k_remove in ckpt_state_dict
                            and v.shape == ckpt_state_dict[k_remove].shape
                        ):
                            filtered_state_dict[k] = ckpt_state_dict[k_remove]
                    else:
                        if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                            filtered_state_dict[k] = ckpt_state_dict[k]
            else:
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
            unexpected_keys = set(filtered_state_dict.keys()) - set(
                target_state_dict.keys()
            )
            logger.info(f"Load the checkpoints from {pre_trained_path}")
            logger.warning(f"Missing keys: {missing_keys}")
            logger.warning(f"Unexpected keys: {unexpected_keys}")
            logger.warning(f"Model loaded keys Num: {len(filtered_state_dict.keys())}")
            for name, param in self.named_parameters():
                if "_text" in name:
                    param.requires_grad = False
            self.tok_embeddings.requires_grad_(False)


class Latent_Pollux_Plan(nn.Module):

    VERSION: str = "v1.0"
    DESCRIPTION: str = (
        "The planning model, similar to LMFusion model, the text branch is freezed and the visual branch is learnable."
    )

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.patchify_size = args.latent_projector.patchify_size
        # assert self.patchify_size == 1, "Patchify size must be 1 for 16x16x8 TVAE."
        self.latent_projector = nn.Linear(
            self.patchify_size**2 * args.latent_projector.latent_dim,
            args.latent_projector.output_dim,
            bias=False,
        )

        # additional embeddings
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, args.latent_projector.output_dim)
        )
        self.vision_boi_emb = nn.Parameter(
            torch.zeros(1, args.latent_projector.output_dim)
        )

        self.rope_embeddings_conditions = RotaryEmbedding1D(
            theta=args.llm.rope_theta,
            head_dim=args.llm.head_dim or args.llm.dim // args.llm.n_heads,
            max_seqlen=args.llm.condition_seqlen,
        )

        # rope embedding
        self.rope_embeddings_image = RotaryEmbedding2D(
            theta=args.llm.rope_theta,
            head_dim=args.llm.head_dim or args.llm.dim // args.llm.n_heads,
            max_seqlen=args.llm.gen_seqlen,
        )

        # llama model
        self.llm_tokenizer = LlamaTokenizerFast.from_pretrained(
            args.tokenizer.model_name
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm = PlanTransformer(args.llm)
        # head
        self.dim = args.llm.dim
        self.norm = RMSNorm(args.llm.dim, eps=args.llm.norm_eps)
        self.latent_head = nn.Linear(
            args.latent_projector.output_dim,
            self.patchify_size**2 * args.codebook_size,
            bias=False,
        )

    def init_weights(self, args: ModelArgs, init_std: Optional[float] = None):
        self.rope_embeddings_image.reset_parameters()
        self.rope_embeddings_conditions.reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.latent_head.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        nn.init.normal_(self.vision_boi_emb, std=0.02)
        nn.init.xavier_uniform_(self.latent_projector.weight)
        nn.init.xavier_uniform_(self.latent_head.weight)
        self.llm.init_weights(args.llm.pre_trained_path, args.llm.from_llama)

    def patchify_and_embed(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        pH = pW = self.patchify_size
        B, C, H, W = x.size()
        x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
        x = self.latent_projector(x)
        x = x.flatten(1, 2)  # [B, H/16*W/16, D]

        # rope embeddings
        freqs_cis = self.rope_embeddings_image.freqs_cis[: H // pH, : W // pW]
        freqs_cis = freqs_cis.flatten(0, 1)
        return x, H, W, freqs_cis

    def unpatchify_image(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = x.size(0)
        pH = pW = self.patchify_size

        x = x.view(B, H // pH, W // pW, pH, pW, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)  # [B,16,H/8,W/8]
        return x

    def process_mask(
        self,
        images_embs: torch.Tensor,
        freqs_cis_img: torch.Tensor,
        mask_strategy: str,
        random_rate: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates attention masks, masked indices, and reordered embeddings.
        """
        device = images_embs.device
        B, L, D = images_embs.shape

        if mask_strategy == "random_mask":
            len_keep = int(L * (1 - random_rate))

            noise = torch.rand(1, L, device=device).repeat(B, 1)
            # sort noise for each sample
            ids_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]  # [16, 204]
            ids_mask = ids_shuffle[:, len_keep:]  # [16, 52]

            images_embs_keep = torch.gather(
                images_embs, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D)
            )  # [16, 204, 3072]

            freqs_cis_img_keep = torch.index_select(
                freqs_cis_img, 0, ids_keep[0]
            )  # [204, 64, 2, 2]
            # generate the binary mask: 0 is keep, 1 is remove
            img_mask = torch.ones([B, L], device=device)
            img_mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            img_mask = torch.gather(img_mask, dim=1, index=ids_restore)  # [16, 256]

            # padded mask tokens
            images_embs_pad = self.mask_token.expand(
                B, L - len_keep, D
            )  # [16, 52, 3072]
            freqs_cis_img_mask = torch.index_select(
                freqs_cis_img, 0, ids_mask[0]
            )  # [52, 64, 2, 2]

            # concatenate the keep and pad tokens
            images_embs = torch.cat(
                [images_embs_keep, images_embs_pad], dim=1
            )  # [B, L, D] [16, 256, 3072]

            freqs_cis_img = torch.cat(
                [freqs_cis_img_keep, freqs_cis_img_mask], dim=0
            )  # [256, 64, 2, 2]

        else:
            raise ValueError(f"Invalid mask strategy: {mask_strategy}")

        return images_embs, freqs_cis_img, ids_restore, img_mask

    def cfg_drop(self, data, cfg_ratio, place_holder=""):
        return [place_holder if random.random() < cfg_ratio else x for x in data]

    def forward(
        self,
        batch: dict[str, any],
        mask_strategy: str = "random_mask",
        attn_impl: str = "sdpa",
    ) -> Tuple[dict[str, any], torch.Tensor]:
        # images = batch["image"]
        captions = batch["caption"]
        if isinstance(batch["caption"][0], tuple):
            captions = [x[0] for x in batch["caption"]]
        vae_latent = batch["latent_code"]

        # vae_indices_size = vae_indices.size()
        # [B, 1, H/16, W/16], [B, 6, 1, H/16, W/16]
        vae_embs, H_, W_, freqs_cis_img = self.patchify_and_embed(vae_latent)
        # [B, H/16 * W/16, D]

        if self.args.random_rate is None:
            mask_rate = random.random()
        else:
            mask_rate = self.args.random_rate
        vae_embs, freqs_cis_img, ids_restore, img_mask = self.process_mask(
            vae_embs,
            freqs_cis_img,
            mask_strategy=mask_strategy,
            random_rate=(
                mask_rate if random.random() > self.args.image_cfg_ratio else 1.0
            ),
        )
        captions = self.cfg_drop(
            captions, cfg_ratio=self.args.text_cfg_ratio, place_holder=""
        )
        # Text Embedding
        tokenizer_output = self.llm_tokenizer(
            captions,
            max_length=self.args.llm.text_seqlen,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = tokenizer_output["input_ids"].to(vae_embs.device)  # [B, L]
        text_embs = self.llm.tok_embeddings(text_input_ids)  # [B, L, D]
        # Concat
        boi_emb = self.vision_boi_emb.unsqueeze(0).expand(vae_embs.size(0), -1, -1)
        # eoi_emb = self.vision_eoi_emb.unsqueeze(0).expand(vae_embs.size(0), -1, -1)
        mm_embs = torch.cat(
            [
                text_embs,
                boi_emb,
                vae_embs,
                # eoi_emb,
            ],
            dim=1,
        )
        vae_start_idx = text_embs.size(1) + 1

        # rope freq
        freqs_cis_text = self.rope_embeddings_conditions.freqs_cis[:vae_start_idx]
        freqs_cis_text = freqs_cis_text.to(mm_embs.device)
        freqs_cis_img = freqs_cis_img.to(mm_embs.device)
        freqs_cis = torch.cat([freqs_cis_text, freqs_cis_img], dim=0)

        # LLM Forward
        h = self.llm(
            h=mm_embs,
            text_l=vae_start_idx,
            visual_l=mm_embs.size(1) - vae_start_idx,
            freq_cis=freqs_cis,
            attn_impl=attn_impl,
        )

        # Latent Head
        latent_hidden = h[:, vae_start_idx : vae_start_idx + vae_embs.size(1), :]
        pred_latent = self.norm(latent_hidden)
        pred_latent = torch.gather(
            pred_latent,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, pred_latent.shape[2]),
        )
        batch["plan_embedding"] = pred_latent.clone()
        pred_latent = self.latent_head(pred_latent)  # [B,M,D]
        # restore the order of the latent codes
        pred_latent = self.unpatchify_image(pred_latent, H_, W_)

        # compute loss
        pred_loss = F.mse_loss(pred_latent, vae_latent)
        return batch, pred_loss

    def set_train(self):
        self.latent_projector.train()
        self.llm.train()

    def set_eval(self):
        self.latent_projector.eval()
        self.llm.eval()

    def to(self, device=None, dtype=None, non_blocking=False):
        super().to(device, dtype, non_blocking)
        return self


def get_no_recompute_ops():
    return None


def build_fsdp_grouping_plan(model_args: ModelArgs, model: nn.Module):
    group_plan = []
    logger.info("\nModel structure:")
    for name, module in model.named_modules():
        logger.info(f"- {name}: {module.__class__.__name__}")

    # llama
    for i in range(model_args.llm.n_layers):
        group_plan.append((f"llm.layers.{i}", False))

    group_plan.append(("latent_head", True))

    logger.info(f"The `group_plan` for FSDP (layer-level granularity):\n{group_plan}")
    return group_plan


def tp_parallelize(
    model, tp_mesh, model_args: ModelArgs, distributed_args
):  # we not use tp_paralle, it is a place holder
    projecter_plan = {
        "latent_projector": RowwiseParallel(output_layouts=Shard(1)),
    }
    parallelize_module(model, tp_mesh, projecter_plan)
    logger.info("`latent_projector` parallelized.")

    if hasattr(model, "llm") and model.llm is not None:
        llama_plan = {
            # Attention components
            "model.layers.*.self_attn.q_proj": ColwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            # MLP components
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        }

        # Apply the parallelization plan to LlamaForCausalLM
        parallelize_module(model.llm, tp_mesh, llama_plan)
        for layer in model.llm.model.layers:
            layer.self_attn.num_heads //= distributed_args.tp_size
            layer.self_attn.head_dim //= distributed_args.tp_size

        logger.info(
            f"LlamaForCausalLM layers parallelized with TP size {distributed_args.tp_size}"
        )

    logger.info("Tensor Parallelization complete.")
