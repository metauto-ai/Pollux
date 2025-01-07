from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging
import random
import numpy as np
from PIL import Image
import math

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import log_softmax, nll_loss
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

from transformers.configuration_utils import PretrainedConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm


from apps.main.modules.tokenizer import Tokenizer, TokenizerArgs
from apps.main.modules.schedulers import RectifiedFlow, SchedulerArgs
from apps.main.modules.vision_tower import VisionTower, VisionTowerArgs
from apps.main.modules.vae import LatentVideoVAE, LatentVideoVAEArgs
from apps.main.modules.preprocess import random_mask_images
from apps.main.modules.embedder import LabelEmbedder


logger = logging.getLogger()


def nll_loss_bf16(log_probs, target):

    selected_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(
        -1
    )
    loss = -selected_log_probs.mean()
    return loss


@dataclass
class LLMArgs:
    model_name: str


@dataclass
class VisionProjecterArgs:
    latent_dim: int = 1024
    output_dim: int = 3072


@dataclass
class ModelArgs:

    vision_tower: VisionTowerArgs = field(default_factory=VisionTowerArgs)
    vision_projecter: VisionProjecterArgs = field(default_factory=VisionProjecterArgs)
    llm: Optional[LLMArgs] = None
    vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    use_vision_boi: bool = True
    text_cfg_ratio: float = 0.1
    image_cfg_ratio: float = 0.1
    patch_size: int = 16
    mask_patch: int = 16
    num_classes: int = 1000
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)


class Pollux(nn.Module):

    VERSION: str = "v0.8.1"
    DESCRIPTION: str = (
        "The planning model, basically an MLLM for predicting the long visual latent codes."
    )

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # vision encoder
        self.vae = LatentVideoVAE(args.vae)
        self.vision_projecter = nn.Linear(
            args.vision_projecter.latent_dim,
            args.vision_projecter.output_dim,
            bias=False,
        )

        # vision boi embedding
        if args.use_vision_boi:
            self.vision_boi_emb = nn.Parameter(
                torch.zeros(1, args.vision_projecter.output_dim)
            )

        # llm
        self.llm_tokenizer = LlamaTokenizerFast.from_pretrained(args.llm.model_name)
        self.llm = LlamaForCausalLM.from_pretrained(args.llm.model_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # head
        # latent_dim = (
        #     args.patch_size * args.patch_size * self.vae.vae.config.latent_channels
        # )
        self.latent_head = nn.Linear(
            args.vision_projecter.output_dim, args.vision_projecter.latent_dim
        )

        # backup
        # self.vision_tower = VisionTower(args.vision_tower)
        # self.scheduler = RectifiedFlow(args.scheduler)

    # def patchify(self, images: torch.Tensor, patch_size: int) -> torch.Tensor:

    #     B, C, H, W = images.shape
    #     assert (
    #         H % patch_size == 0 and W % patch_size == 0
    #     ), "Image dimensions must be divisible by patch_size"
    #     patches = images.unfold(2, patch_size, patch_size).unfold(
    #         3, patch_size, patch_size
    #     )
    #     patches = patches.permute(
    #         0, 2, 3, 1, 4, 5
    #     ).contiguous()  # [B, num_patches_h, num_patches_w, C, patch_size, patch_size]
    #     patches = patches.view(B, -1, C, patch_size, patch_size)
    #     return patches

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

    # def compute_2d_rope(self, freq_dim: int, h: int, w: int, scale: float = 10000.0):

    #     freq_h = 1.0 / (scale ** (torch.arange(0, freq_dim, 2).float() / freq_dim))
    #     freq_w = 1.0 / (scale ** (torch.arange(0, freq_dim, 2).float() / freq_dim))

    #     h_positions = torch.arange(h).unsqueeze(1) * freq_h.unsqueeze(
    #         0
    #     )  # (h, freq_dim/2)
    #     w_positions = torch.arange(w).unsqueeze(1) * freq_w.unsqueeze(
    #         0
    #     )  # (w, freq_dim/2)

    #     rope_h = torch.cat([torch.sin(h_positions), torch.cos(h_positions)], dim=-1)
    #     rope_w = torch.cat([torch.sin(w_positions), torch.cos(w_positions)], dim=-1)

    #     rope_2d = torch.cat(
    #         [
    #             rope_h.unsqueeze(1).expand(-1, w, -1),
    #             rope_w.unsqueeze(0).expand(h, -1, -1),
    #         ],
    #         dim=-1,
    #     )
    #     return rope_2d.view(h * w, -1)

    # def process_input(self, input_ids, input_imgs, **kwargs):

    #     if input_imgs.ndim == 4:
    #         # NOTE: single image should unsqueeze to [B, 1, C, H, W]
    #         input_imgs = input_imgs.unsqueeze(1)

    #     batch_size, num_images, channels, height, width = input_imgs.shape
    #     images = rearrange(input_imgs, "b n c h w -> (b n) c h w")

    #     # images_embs = self.vision_tower(images)
    #     images_embs = self.vision_projecter(images_embs)
    #     boi_embs = self.vision_boi_emb.unsqueeze(0).expand(images_embs.size(0), -1, -1)

    #     images_embs = torch.cat([boi_embs, images_embs], dim=1)
    #     images_embs = rearrange(
    #         images_embs, "(b n) t d -> b (n t) d", b=batch_size, n=num_images
    #     )

    #     text_embs = self.llm.get_input_embeddings()(input_ids)

    #     return text_embs, images_embs

    def process_mask(
        self, input_ids, images_embs, mask_strategy: str, random_rate: float = 0.15
    ) -> torch.Tensor:

        text_seq_len = input_ids.shape[1]
        visual_seq_len = images_embs.shape[1]

        if mask_strategy == "full_mask":
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (input_ids.shape[0], text_seq_len), device=input_ids.device
                    ),
                    torch.zeros(
                        (input_ids.shape[0], visual_seq_len), device=input_ids.device
                    ),
                ],
                dim=1,
            )
        elif mask_strategy == "random_mask":
            random_mask = (
                torch.rand(
                    (input_ids.shape[0], visual_seq_len), device=input_ids.device
                )
                < random_rate
            )
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (input_ids.shape[0], text_seq_len), device=input_ids.device
                    ),
                    (~random_mask).float(),
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Invalid mask strategy: {mask_strategy}")

        return attention_mask

    def forward(
        self,
        batch: dict[str, any],
        mask_strategy: str = "random_mask",
        random_rate: float = 0.3,
    ) -> Tuple[dict[str, any], torch.Tensor]:

        images = batch["image"]  # Shape: [B, C, H, W]
        captions = batch["caption"]

        # vision vae
        print("before_vae", torch.cuda.max_memory_allocated() / 1024 / 1024)
        vae_latent = self.vae.encode(images)  # [B,16,H/8,W/8]
        vae_latent = vae_latent.flatten(2, 3).permute(0, 2, 1)  # [B,H/8*W/8,16]
   
        print("after_vae", torch.cuda.max_memory_allocated() / 1024 / 1024)
        vae_embed = self.vision_projecter(vae_latent)  # [B,H/8*W/8,D]
        # print("after_projector", torch.cuda.max_memory_allocated() / 1024 / 1024)

        # text embed
        input_ids = self.llm_tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"].to(self.llm.device)
        text_embs = self.llm.get_input_embeddings()(input_ids)  # [B,T,D]
        print("after_llm", torch.cuda.max_memory_allocated() / 1024 / 1024)

        # concate embeddings
        boi_embs = self.vision_boi_emb.unsqueeze(0).expand(vae_embed.size(0), -1, -1)
        mm_embs = torch.cat(
            [text_embs, boi_embs, vae_embed], dim=1
        )  # multi-modal embeddings
        vae_start_idx, vae_end_idx = text_embs.size(1) + 1, text_embs.size(
            1
        ) + 1 + vae_embed.size(1)

        # ====> text_embs shape: torch.Size([12, 10, 3072])
        # ====> vae_embed shape: torch.Size([12, 1024, 3072])
        # ====> mm_embs shape: torch.Size([12, 1035, 3072])

        # llm forward
        output = self.llm(
            inputs_embeds=mm_embs,
            # attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        final_hidden = output.hidden_states[-1]
        # ====> final_hidden shape: torch.Size([12, 1049, 3072])

        pred_latent = self.latent_head(final_hidden[:, vae_start_idx:, :])  # [B,M,D]
        # ====> pred_latent shape: torch.Size([12, 1024, 16]), vae_latent shape: torch.Size([12, 1024, 16])

        # compute loss
        print("vae shape ", vae_latent.shape)
        vae_latent = torch.rand([12, 1,16]).to(vae_latent.device)
        pred_latent = torch.zeros_like(vae_latent, requires_grad=True)
        pred_loss = F.mse_loss(pred_latent, vae_latent)
        # pred_loss = torch.zeros(images.shape[0], device=images.device)
        a = input()

        # batch["latent_target"] = latent_target
        # batch["pred_latent"] = pred_latent

        return batch, pred_loss


    def set_train(self):
        # self.vision_tower.train()
        self.vision_projecter.train()
        # self.classifier.train()

    def set_eval(self):
        # self.vision_tower.eval()
        self.vision_projecter.eval()
        # self.classifier.eval()

    # def init_weights(self, args: ModelArgs):
    #     self.vision_tower.init_weights(args.vision_tower.pre_trained_path)
    #     # self.llm.init_weights(args.llm.pre_trained_path)
    def init_weights(self, args: ModelArgs):

        # if args.vision_tower.pre_trained_path:
        #     self.vision_tower.init_weights(args.vision_tower.pre_trained_path)
        # else:
        #     for module in self.vision_tower.modules():
        #         if isinstance(module, nn.Linear):
        #             nn.init.xavier_uniform_(module.weight)
        #             if module.bias is not None:
        #                 nn.init.zeros_(module.bias)
        #         elif isinstance(module, nn.Conv2d):
        #             nn.init.kaiming_normal_(
        #                 module.weight, mode="fan_out", nonlinearity="relu"
        #             )
        #         elif isinstance(module, nn.LayerNorm) or isinstance(
        #             module, nn.BatchNorm2d
        #         ):
        #             nn.init.ones_(module.weight)
        #             nn.init.zeros_(module.bias)

        if args.llm and args.llm.model_name:
            pretrained_state_dict = LlamaForCausalLM.from_pretrained(
                args.llm.model_name
            ).state_dict()
            self.llm.load_state_dict(pretrained_state_dict)

        nn.init.xavier_uniform_(self.vision_projecter.weight)
        # nn.init.zeros_(self.vision_projecter.bias)
        nn.init.normal_(self.vision_boi_emb, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


def build_fsdp_grouping_plan(model_args: ModelArgs, model: nn.Module):
    group_plan = []
    logger.info("\nModel structure:")
    for name, module in model.named_modules():
        logger.info(f"- {name}: {module.__class__.__name__}")

    # vision_transformer = getattr(model.vision_tower, "vision_tower", None)

    # logger.info("VisionTransformer has `blocks` attribute. Building group plan...")
    # for idx, block in enumerate(vision_transformer.blocks):
    #     group_plan.append((f"vision_tower.vision_tower.blocks.{idx}", True))

    group_plan.append(("vision_projecter", False))

    llama_model = getattr(model, "llm", None)
    if llama_model and hasattr(llama_model, "model"):
        logger.info("LlamaForCausalLM has `model` attribute. Building group plan...")
        for idx, block in enumerate(llama_model.model.layers):
            group_plan.append((f"llm.model.layers.{idx}", False))

    vae_encoder = getattr(model.vae.vae.encoder, "down_blocks", None)
    if vae_encoder:
        for i in range(len(vae_encoder)):
            group_plan.append((f"vae.vae.encoder.down_blocks.{i}", False))
    else:
        logger.warning("VAE encoder does not have `down_blocks` attribute.")

    logger.info(f"The `group_plan` for FSDP (layer-level granularity):\n{group_plan}")
    return group_plan


def tp_parallelize(model, tp_mesh, model_args: ModelArgs, distributed_args):

    # Parallelize `vision_tower` (VisionTransformer)
    # vision_tower = model.vision_tower.vision_tower

    # transformer_plan = {
    #     # Attention components
    #     "blocks.*.attn.qkv": ColwiseParallel(
    #         input_layouts=Replicate(), output_layouts=Shard(1)
    #     ),
    #     "blocks.*.attn.proj": RowwiseParallel(output_layouts=Shard(1)),
    #     # MLP components
    #     "blocks.*.mlp.fc1": ColwiseParallel(),
    #     "blocks.*.mlp.fc2": RowwiseParallel(output_layouts=Shard(1)),
    # }

    # parallelize_module(vision_tower, tp_mesh, transformer_plan)

    # for block in vision_tower.blocks:
    #     block.attn.num_heads //= distributed_args.tp_size
    #     block.attn.head_dim //= distributed_args.tp_size
    logger.info(
        f"VisionTransformer blocks parallelized with TP size {distributed_args.tp_size}"
    )

    projecter_plan = {
        "vision_projecter": RowwiseParallel(output_layouts=Shard(1)),
    }
    parallelize_module(model, tp_mesh, projecter_plan)
    logger.info("`vision_projecter` parallelized.")

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
