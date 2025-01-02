from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging
import random
import numpy as np
from PIL import Image

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

    selected_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    loss = -selected_log_probs.mean() 
    return loss



@dataclass
class ModelArgs:
    # gen_transformer: GenTransformerArgs = field(default_factory=GenTransformerArgs)
    # plan_transformer: PlanTransformerArgs = field(default_factory=PlanTransformerArgs)
    vision_tower: VisionTowerArgs = field(default_factory=VisionTowerArgs)
    # vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    # llm: # TODO: Add LlamaArgs
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)
    # tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs) #TODO: Add TokenizerArgs
    text_cfg_ratio: float = 0.1
    image_cfg_ratio: float = 0.1
    mask_patch: int = 16
    num_classes: int = 1000


class Pollux(nn.Module):

    VERSION: str = "v0.7"
    DESCRIPTION: str = (
        "Latent Diffusion Transformer for VideoGen: (1) currently we only support class conditional image generation for debugging."
    )

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.vision_tower = VisionTower(args.vision_tower)  
        self.vision_projecter = nn.Linear(1024, 2048, bias=True) 
        self.vision_boi_emb = nn.Parameter(torch.zeros(1, 1024))
        
        # TODO: Now we keep HF original usage. Will specify in the future.
        self.lang_tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-3.2-3B")
        self.llm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

        #self.vae = LatentVideoVAE(args.vae)
        #self.scheduler = RectifiedFlow(args.scheduler)

    def patchify(self, images: torch.Tensor, patch_size: int) -> torch.Tensor:

        B, C, H, W = images.shape
        assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch_size"
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, num_patches_h, num_patches_w, C, patch_size, patch_size]
        patches = patches.view(B, -1, C * patch_size * patch_size)  # Flatten patches
        return patches


    def process_input(self, input_ids, input_imgs, mask_strategy: str = "full_mask", random_rate: float = 0.15, **kwargs):

        if input_imgs.ndim == 4:  
             # NOTE: single image should unsqueeze to [B, 1, C, H, W]
            images = input_imgs.unsqueeze(1)

        batch_size, n = images.shape[:2]
        images = rearrange(images, "b n c h w -> (b n) c h w")
        image_patches = self.patchify(images, patch_size=16)

        images_embs = self.vision_tower(image_patches)
        images_embs = self.vision_projecter(images_embs)
        
        # NOTE: add BOI token for each visual sequence 
        boi_emb = self.vision_boi_emb[0].detach().clone()
        images_embs = torch.cat(
            [
                boi_emb.view(1, 1, -1).repeat(images_embs.shape[0], 1, 1),
                images_embs,
            ],
            dim=1,
        )

        text_embs = self.llm.get_input_embeddings()(input_ids)
        images_embs = rearrange(images_embs, "(b n) t d -> b (n t) d", b=batch_size, n=n)        
        
        return text_embs, images_embs

    def process_mask(self, input_ids, images_embs, mask_strategy: str, random_rate: float = 0.15) -> torch.Tensor:
 
        text_seq_len = input_ids.shape[1]
        visual_seq_len = images_embs.shape[1]

        if mask_strategy == "full_mask":
            attention_mask = torch.cat(
                [torch.ones((input_ids.shape[0], text_seq_len), device=input_ids.device),
                torch.zeros((input_ids.shape[0], visual_seq_len), device=input_ids.device)],
                dim=1,
            )
        elif mask_strategy == "random_mask":
            random_mask = torch.rand((input_ids.shape[0], visual_seq_len), device=input_ids.device) < random_rate
            attention_mask = torch.cat(
                [torch.ones((input_ids.shape[0], text_seq_len), device=input_ids.device),
                (~random_mask).float()],
                dim=1,
            )
        else:
            raise ValueError(f"Invalid mask strategy: {mask_strategy}")

        return attention_mask


    def contrastive_loss(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        logits = torch.cat([positive_scores, negative_scores], dim=1) / temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss


    def forward(self, batch: dict[str, any], mask_strategy: str = "random_mask", random_rate: float = 0.3) -> Tuple[dict[str, any], torch.Tensor]:

        images = batch["image"]  # Shape: [B, C, H, W]
        captions = batch["caption"]

        input_ids = self.lang_tokenizer(captions, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(self.llm.device)

        text_embs, images_embs = self.process_input(input_ids, images)
        combined_embs = torch.cat([text_embs, images_embs], dim=1)
        attention_mask = self.process_mask(input_ids, images_embs, mask_strategy, random_rate)

        outputs = self.llm(
            inputs_embeds=combined_embs,
            attention_mask=attention_mask,
        )
        predicted_tokens = outputs.sequences

        visual_start_idx = input_ids.shape[1] 
        visual_attention_mask = attention_mask[:, visual_start_idx:].bool()
        _visual_tokens = combined_embs[:, visual_start_idx:][visual_attention_mask]

        masked_indices = ~visual_attention_mask
        masked_embs = combined_embs[:, visual_start_idx:][masked_indices].view(-1, combined_embs.size(-1))
        original_embs = images_embs[masked_indices]

        positive_scores = F.cosine_similarity(original_embs, masked_embs, dim=-1).unsqueeze(1)
        negative_scores = F.cosine_similarity(
            original_embs.unsqueeze(1),  
            masked_embs.unsqueeze(0),  
            dim=-1
        )
        negative_scores = negative_scores.masked_fill(torch.eye(negative_scores.size(0), device=negative_scores.device).bool(), 0)
        contrastive_loss = self.contrastive_loss(positive_scores, negative_scores)

        generation_loss = F.cross_entropy(predicted_tokens.view(-1, predicted_tokens.size(-1)), _visual_tokens.view(-1))
        total_loss = contrastive_loss + generation_loss

        batch["prediction"] = predicted_tokens

        return batch, total_loss

    def set_train(self):
        self.vision_tower.train()
        self.vision_projecter.train()
        # self.classifier.train()
 
    def set_eval(self):
        self.vision_tower.eval()
        self.vision_projecter.eval()
        # self.classifier.eval()

    def init_weights(self, args: ModelArgs):
        self.vision_tower.init_weights(args.vision_tower.pre_trained_path)

# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


def build_fsdp_grouping_plan(model_args: ModelArgs, model: nn.Module):
    group_plan = []
    logger.info("\nModel structure:")
    for name, module in model.named_modules():
        logger.info(f"- {name}: {module.__class__.__name__}")

    vision_transformer = getattr(model.vision_tower, "vision_tower", None)

    logger.info("VisionTransformer has `blocks` attribute. Building group plan...")
    for idx, block in enumerate(vision_transformer.blocks):
        group_plan.append((f"vision_tower.vision_tower.blocks.{idx}", True)) 

    group_plan.append(("vision_projecter", False))
    # group_plan.append(("classifier", True))

    logger.info(f"The `group_plan` for FSDP (layer-level granularity):\n{group_plan}")
    return group_plan


def tp_parallelize(model, tp_mesh, model_args: ModelArgs, distributed_args):

    # Parallelize `vision_tower` (VisionTransformer)
    vision_tower = model.vision_tower.vision_tower

    # Define the parallelization plan for VisionTransformer blocks
    transformer_plan = {
        # Attention components
        "blocks.*.attn.qkv": ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
        "blocks.*.attn.proj": RowwiseParallel(output_layouts=Shard(1)),
        # MLP components
        "blocks.*.mlp.fc1": ColwiseParallel(),
        "blocks.*.mlp.fc2": RowwiseParallel(output_layouts=Shard(1)),
    }

    # Apply the parallelization plan to VisionTransformer
    parallelize_module(vision_tower, tp_mesh, transformer_plan)

    # Update attributes for attention heads based on TP size
    for block in vision_tower.blocks:
        block.attn.num_heads //= distributed_args.tp_size
        block.attn.head_dim //= distributed_args.tp_size

    logger.info(f"VisionTransformer blocks parallelized with TP size {distributed_args.tp_size}")

    # Parallelize `vision_projecter` (Linear layer)
    projecter_plan = {
        "vision_projecter": RowwiseParallel(output_layouts=Shard(1)),
    }
    parallelize_module(model, tp_mesh, projecter_plan)
    logger.info("`vision_projecter` parallelized.")

    # Parallelize `classifier` (Linear layer)
    # classifier_plan = {
    #     "classifier": RowwiseParallel(output_layouts=Shard(1)),
    # }
    # parallelize_module(model, tp_mesh, classifier_plan)
    # logger.info("`classifier` parallelized.")

    logger.info("Tensor Parallelization complete.")
