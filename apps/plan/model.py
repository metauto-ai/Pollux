from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging
import random
import torch
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

# from transformers.configuration_utils import PretrainedConfig
# from transformers import (
#     AutoConfig,
#     AutoModelForCausalLM,
#     PreTrainedModel,
#     LlamaConfig,
#     LlamaForCausalLM,
# )
# from transformers.models.llama.modeling_llama import LlamaRMSNorm


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
class ModelArgs:
    # gen_transformer: GenTransformerArgs = field(default_factory=GenTransformerArgs)
    # plan_transformer: PlanTransformerArgs = field(default_factory=PlanTransformerArgs)
    vision_tower: VisionTowerArgs = field(default_factory=VisionTowerArgs)
    # vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    # llm: # TODO: Add LlamaArgs
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
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
        self.vision_aligner = nn.Linear(1024, 2048, bias=True)
        self.classifier = nn.Linear(2048, args.num_classes, bias=True)
        # self.llm = LlamaForCausalLM(LlamaConfig)

        # self.cond_provider =
        # self.cond_boe_emb = nn.Parameter(torch.zeros(1, 1024))
        # self.generator

    def forward(self, batch: dict[str:any]) -> dict[str:any]:

        image = batch["image"]
        label = batch["label"]

        tower_output = self.vision_tower(image)  # [48, 256, 1024]
        aligned_output = self.vision_aligner(tower_output)  # [48, 256, 2048]
        aligned_output = aligned_output.mean(dim=1)
        output = self.classifier(aligned_output)  # [48, 1000]

        batch["prediction"] = output
        label = batch["label"]

        # TODO: remove this
        label = label.to(output.device)
        loss = F.cross_entropy(output, label)

        return batch, loss

    def set_train(self):
        self.vision_tower.train()
        self.vision_aligner.train()
        self.classifier.train()

    def set_eval(self):
        self.vision_tower.eval()
        self.vision_aligner.eval()
        self.classifier.eval()

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

    group_plan.append(("vision_aligner", False))
    group_plan.append(("classifier", True))

    logger.info(f"The `group_plan` for FSDP (layer-level granularity):\n{group_plan}")
    return group_plan


def tp_parallelize(model, tp_mesh, model_args: ModelArgs, distributed_args):

    # Parallelize `vision_tower` (VisionTransformer)
    vision_tower = model.vision_tower.vision_tower

    # Define the parallelization plan for VisionTransformer blocks
    transformer_plan = {
        # Attention components
        "blocks.*.attn.qkv": ColwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1)
        ),
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

    logger.info(
        f"VisionTransformer blocks parallelized with TP size {distributed_args.tp_size}"
    )

    # Parallelize `vision_aligner` (Linear layer)
    aligner_plan = {
        "vision_aligner": RowwiseParallel(output_layouts=Shard(1)),
    }
    parallelize_module(model, tp_mesh, aligner_plan)
    logger.info("`vision_aligner` parallelized.")

    # Parallelize `classifier` (Linear layer)
    classifier_plan = {
        "classifier": RowwiseParallel(output_layouts=Shard(1)),
    }
    parallelize_module(model, tp_mesh, classifier_plan)
    logger.info("`classifier` parallelized.")

    logger.info("Tensor Parallelization complete.")
