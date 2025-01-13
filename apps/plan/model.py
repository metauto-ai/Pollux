from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)
from transformers import LlamaTokenizerFast
from apps.main.modules.vae import LatentVideoVAE, LatentVideoVAEArgs
from apps.main.modules.embedder import LabelEmbedder
from apps.main.modules.gen_transformer import (
    RotaryEmbedding1D,
    RotaryEmbedding2D,
)
from lingua.transformer import TransformerBlock, RMSNorm, InitStdFactor, cross_entropy

logger = logging.getLogger()


@dataclass
class LatentProjecterArgs:
    latent_dim: int = 16
    output_dim: int = 3072
    patchify_size: int = 1


@dataclass
class TokenizerArgs:
    model_name: str = "meta-llama/Llama-3.2-3B"


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
    vocab_size: int = 128256
    pre_trained_path: Optional[str] = None


@dataclass
class ModelArgs:
    vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    latent_projector: LatentProjecterArgs = field(default_factory=LatentProjecterArgs)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    llm: LlamaArgs = field(default_factory=LlamaArgs)
    use_vision_boi: bool = True
    text_cfg_ratio: float = 0.1
    image_cfg_ratio: float = 0.1
    codebook_size: int = 512
    num_classes: int = 1000


class LlamaTransformer(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

    def forward(self, h, freq_cis, mask: str = "causal", attn_impl: str = "sdpa"):
        for layer in self.layers:
            h = layer(h, freq_cis, mask=mask, attn_impl=attn_impl)
        return h

    def init_weights(self, pre_trained_path: Optional[str] = None):
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


class Pollux(nn.Module):

    VERSION: str = "v0.8.2"
    DESCRIPTION: str = (
        "The planning model, basically an MLLM for predicting the long visual latent codes."
    )

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.tvae = LatentVideoVAE(args.vae)
        self.patchify_size = args.latent_projector.patchify_size
        assert self.patchify_size == 1, "Patchify size must be 1 for 16x16x8 TVAE."
        self.latent_projector = nn.Linear(
            self.patchify_size**2 * args.latent_projector.latent_dim,
            args.latent_projector.output_dim,
            bias=False,
        )

        # additional embeddings
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, args.latent_projector.output_dim)
        )
        self.vision_cls_emb = LabelEmbedder(
            num_classes=args.num_classes,
            hidden_size=args.llm.dim,
            dropout_prob=args.image_cfg_ratio,
        )
        self.vision_boi_emb = nn.Parameter(
            torch.zeros(1, args.latent_projector.output_dim)
        )
        # self.vision_eoi_emb = nn.Parameter(
        #     torch.zeros(1, args.latent_projector.output_dim)
        # )
        # we do not use eoi for now

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

        self.llm = LlamaTransformer(args.llm)
        self.llm.init_weights(args.llm.pre_trained_path)

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
        # nn.init.normal_(self.vision_eoi_emb, std=0.02)
        nn.init.xavier_uniform_(self.latent_projector.weight)
        nn.init.xavier_uniform_(self.latent_head.weight)
        # nn.init.zeros_(self.latent_head.bias)
        self.vision_cls_emb.reset_parameters()

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
            noise = torch.rand(B, L, device=device)

            # sort noise for each sample
            ids_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            images_embs_keep = torch.gather(
                images_embs, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D)
            )

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, L], device=device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # padded mask tokens
            images_embs_pad = self.mask_token.expand(B, L - len_keep, D)

            # concatenate the keep and pad tokens
            images_embs = torch.cat(
                [images_embs_keep, images_embs_pad], dim=1
            )  # [B, L, D]

        elif mask_strategy == "full_mask":
            raise NotImplementedError("Full mask is not implemented yet.")

        else:
            raise ValueError(f"Invalid mask strategy: {mask_strategy}")
        return images_embs, ids_restore

    def forward(
        self,
        batch: dict[str, any],
        mask_strategy: str = "random_mask",
        random_rate: float = 0.3,
        attn_impl: str = "sdpa",
    ) -> Tuple[dict[str, any], torch.Tensor]:

        images = batch["image"]
        labels = batch["label"]
        captions = batch["caption"]

        # Class Embedding
        cls_emb = self.vision_cls_emb(labels, train=True).unsqueeze(1)  # [B, 1, D]

        # Vision Discrete Latent Codes
        self.tvae.vae.vae._enc_model.to(images.device)
        vae_indices, vae_latent = self.tvae.encode(images)
        # [B, 1, H/16, W/16], [B, 6, 1, H/16, W/16]
        vae_embs, H_, W_, freqs_cis_img = self.patchify_and_embed(vae_latent.squeeze(2))
        # [B, H/16 * W/16, D]

        # apply masking
        original_vae_embs = vae_embs.clone()
        vae_embs, ids_restore = self.process_mask(
            vae_embs,
            mask_strategy=mask_strategy,
            random_rate=random_rate,
        )

        # Text Embedding
        tokenizer_output = self.llm_tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = tokenizer_output["input_ids"].to(vae_embs.device)  # [B, L]
        text_embs = self.llm.tok_embeddings(text_input_ids)  # [B, L, D]
        # attn_mask = tokenizer_output["attention_mask"].to(vae_embs.device)  # [B, L]
        # attn_mask: left padding, invalid is 0, valid is 1. we do not need it now.

        # Concat
        boi_emb = self.vision_boi_emb.unsqueeze(0).expand(vae_embs.size(0), -1, -1)
        # eoi_emb = self.vision_eoi_emb.unsqueeze(0).expand(vae_embs.size(0), -1, -1)
        mm_embs = torch.cat(
            [
                text_embs,
                cls_emb,
                boi_emb,
                vae_embs,
                # eoi_emb,
            ],
            dim=1,
        )
        vae_start_idx = text_embs.size(1) + 2

        # rope freq
        freqs_cis_text = self.rope_embeddings_conditions.freqs_cis[:vae_start_idx]
        freqs_cis_text = freqs_cis_text.to(mm_embs.device)
        freqs_cis_img = freqs_cis_img.to(mm_embs.device)
        freqs_cis = torch.cat([freqs_cis_text, freqs_cis_img], dim=0)

        # LLM Forward
        h = self.llm(mm_embs, freqs_cis, attn_impl=attn_impl)

        # Latent Head
        latent_hidden = h[:, vae_start_idx : vae_start_idx + vae_embs.size(1), :]
        pred_latent = self.latent_head(self.norm(latent_hidden))  # [B,M,D]
        # pred_latent = self.unpatchify_image(pred_latent, H_, W_)

        # restore the order of the latent codes
        pred_latent = torch.gather(
            pred_latent,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, pred_latent.shape[2]),
        )

        # compute loss
        # pred_loss = F.mse_loss(pred_latent, vae_latent.squeeze(2))
        # TODO: either next-token-prediction or reconstruction token prediction
        # vae_indices: [B, 1, H/16, W/16], pred_latent: [B, M, codebook_size]
        vae_indices = vae_indices.squeeze(1).flatten(1).long()  # [B, H/16*W/16]
        # pred_loss = F.cross_entropy(pred_latent[:, :-1].permute(0, 2, 1), vae_indices[:, 1:])
        pred_loss = cross_entropy(
            pred_latent[:, :-1].flatten(0, 1),
            vae_indices[:, 1:].flatten(0, 1),
            reduction="mean",
        )

        # batch["latent_target"] = latent_target
        # batch["pred_latent"] = pred_latent
        return batch, pred_loss

    def set_train(self):
        self.latent_projector.train()
        self.vision_cls_emb.train()
        self.llm.train()

    def set_eval(self):
        self.latent_projector.eval()
        self.vision_cls_emb.eval()
        self.llm.eval()


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
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

    # llama_model = getattr(model, "llm", None)
    # if llama_model and hasattr(llama_model, "model"):
    #     logger.info("LlamaForCausalLM has `model` attribute. Building group plan...")
    #     for idx, block in enumerate(llama_model.model.layers):
    #         group_plan.append((f"llm.model.layers.{idx}", False))

    # NOTE: Hunyuan
    # vae_encoder = getattr(model.vae.vae.encoder, "down_blocks", None)
    # if vae_encoder:
    #     for i in range(len(vae_encoder)):
    #         group_plan.append((f"vae.vae.encoder.down_blocks.{i}", False))
    # else:
    #     logger.warning("VAE encoder does not have `down_blocks` attribute.")

    # COSMOS
    # vae_encoder = getattr(model.vae.vae.vae._enc_model, "encoder", None)
    # if vae_encoder and hasattr(vae_encoder, "down"):
    #     for i in range(len(vae_encoder.down)):
    #         group_plan.append((f"vae.vae.vae._enc_model.encoder.down.{i}", False))
    # elif vae_encoder and hasattr(vae_encoder, "mid"):
    #     group_plan.append((f"vae.vae.vae._enc_model.encoder.mid.block_1", False))
    #     group_plan.append((f"vae.vae.vae._enc_model.encoder.mid.attn_1", False))
    #     group_plan.append((f"vae.vae.vae._enc_model.encoder.mid.block_2", False))

    logger.info(f"The `group_plan` for FSDP (layer-level granularity):\n{group_plan}")
    return group_plan


def tp_parallelize(model, tp_mesh, model_args: ModelArgs, distributed_args):
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
