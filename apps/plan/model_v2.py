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


        batch_size, n = images.shapep[:2]
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
        
        # text_seq_len = text_embs.shape[1]
        # visual_seq_len = images_embs.shape[1]

        # NOTE: for this version we only mask images. 
        if mask_strategy == "full_mask":
            noise_embs = torch.randn_like(images_embs)
            images_embs = noise_embs
        elif mask_strategy == "random_mask":
            random_mask = torch.rand(images_embs.shape[:2], device=images_embs.device) < random_rate
            noise_embs = torch.randn_like(images_embs)
            images_embs[random_mask] = noise_embs[random_mask]

        text_embs = self.llm.get_input_embeddings()(input_ids)
        combined_embs = torch.cat([text_embs, images_embs], dim=1)
        return combined_embs

    def forward(self, batch: dict[str:any]) -> dict[str:any]:

        # NOTE: currently only a single image but the following func support multiple for future
        image = batch["image"] 
        label = batch["label"]
        caption = batch["caption"]

        input_ids = self.tokenizer(caption)
        input_embs = self.process_input(input_ids, image, mask_strategy="random_mask", random_rate=0.15)
  
        # NOTE: develop to here. Will finish tonight
        breakpoint()

        batch["prediction"] = output
        label = batch["label"]

        # TODO: remove this
        label = label.to(output.device)
        loss = F.cross_entropy(output, label)

        return batch, loss

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
