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
from torch.nn.functional import log_softmax
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


@dataclass
class LLMArgs:
    model_name: str
    hidden_size: int


@dataclass
class LatentProjecterArgs:
    latent_dim: int = 16
    output_dim: int = 3072
    patchify_size: int = 1


@dataclass
class ModelArgs:

    vision_tower: VisionTowerArgs = field(default_factory=VisionTowerArgs)
    latent_projector: LatentProjecterArgs = field(default_factory=LatentProjecterArgs)
    llm: Optional[LLMArgs] = None
    vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    use_vision_boi: bool = True
    text_cfg_ratio: float = 0.1
    image_cfg_ratio: float = 0.1
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)
    num_classes: int = 1000


class Pollux(nn.Module):

    VERSION: str = "v0.8.2"
    DESCRIPTION: str = (
        "The planning model, basically an MLLM for predicting the long visual latent codes."
    )

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.vae = LatentVideoVAE(args.vae)
        self.patchify_size = args.latent_projector.patchify_size
        self.latent_projector = nn.Linear(
            self.patchify_size**2 * args.latent_projector.latent_dim,
            args.latent_projector.output_dim,
            bias=False,
        )

        self.vision_cls_emb = LabelEmbedder(
            num_classes=args.num_classes,
            hidden_size=args.llm.hidden_size,
            dropout_prob=args.image_cfg_ratio,
        )

        self.vision_boi_emb = nn.Parameter(torch.zeros(1, args.latent_projector.output_dim))
        self.vision_eoi_emb = nn.Parameter(torch.zeros(1, args.latent_projector.output_dim))

        self.llm_tokenizer = LlamaTokenizerFast.from_pretrained(args.llm.model_name)
        self.llm = LlamaForCausalLM.from_pretrained(args.llm.model_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.latent_head = nn.Linear(
            args.latent_projector.output_dim,
            self.patchify_size**2 * args.latent_projector.latent_dim,
        )


    
    def patchify_and_embed(self, x: torch.Tensor):
        pH = pW = self.patchify_size
        B, C, H, W = x.size()
        x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3)
        x = self.latent_projector(x)
        x = x.flatten(1, 2)  # [B, H/16*W/16, D]
        return x, H, W

    def unpatchify_image(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = x.size(0)
        pH = pW = self.patchify_size

        x = x.view(B, H // pH, W // pW, pH, pW, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)  # [B,16,H/8,W/8]
        return x

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

        images = batch["image"]
        labels = batch["label"]
        captions = batch["caption"]

        # Class Embedding
        cls_emb = self.vision_cls_emb(labels, train=True).unsqueeze(1)  # [B, 1, D]

        # Vision Discrete Latent Codes
        vae_indices, vae_latent = self.vae.encode(images)  # [B, 1, H/16, W/16], [B, 6, 1, H/16, W/16]
        vae_embs, H_, W_ = self.patchify_and_embed(vae_latent.squeeze(2)) # [B, H/16*W/16, D]

        # Text Embedding
        input_ids = self.llm_tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"].to(self.llm.device)
        text_embs = self.llm.get_input_embeddings()(input_ids)  # [B,T,D]

        # Concat
        boi_emb = self.vision_boi_emb.unsqueeze(0).expand(vae_embs.size(0), -1, -1)
        eoi_emb = self.vision_eoi_emb.unsqueeze(0).expand(vae_embs.size(0), -1, -1)

        mm_embs = torch.cat([text_embs, cls_emb, boi_emb, vae_embs, eoi_emb], dim=1)
        vae_start_idx = text_embs.size(1) + 2

        # LLM Forward
        output = self.llm(
            inputs_embeds=mm_embs,
            # attention_mask=attention_mask,
            output_hidden_states=True,
            # labels=input_ids,
            return_dict=True,
        )

        # Latent Head
        latent_hidden = output.hidden_states[-1][:, vae_start_idx:vae_start_idx+vae_embs.size(1), :]  # [B,M,D]
        pred_latent = self.latent_head(latent_hidden)
        pred_latent = self.unpatchify_image(pred_latent, H_, W_) 

        # compute loss
        pred_loss = F.mse_loss(pred_latent, vae_latent.squeeze(2))

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

    def init_weights(self, args: ModelArgs):

        if args.llm and args.llm.model_name:
            pretrained_state_dict = LlamaForCausalLM.from_pretrained(
                args.llm.model_name
            ).state_dict()
            self.llm.load_state_dict(pretrained_state_dict)

        nn.init.xavier_uniform_(self.latent_projector.weight)
        nn.init.normal_(self.vision_boi_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.vision_eoi_emb, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)

        self.vision_cls_emb.reset_parameters()

# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


def build_fsdp_grouping_plan(model_args: ModelArgs, model: nn.Module):
    group_plan = []
    logger.info("\nModel structure:")
    for name, module in model.named_modules():
        logger.info(f"- {name}: {module.__class__.__name__}")

    llama_model = getattr(model, "llm", None)
    if llama_model and hasattr(llama_model, "model"):
        logger.info("LlamaForCausalLM has `model` attribute. Building group plan...")
        for idx, block in enumerate(llama_model.model.layers):
            group_plan.append((f"llm.model.layers.{idx}", False))

    # NOTE: Hunyuan
    # vae_encoder = getattr(model.vae.vae.encoder, "down_blocks", None)
    # if vae_encoder:
    #     for i in range(len(vae_encoder)):
    #         group_plan.append((f"vae.vae.encoder.down_blocks.{i}", False))
    # else:
    #     logger.warning("VAE encoder does not have `down_blocks` attribute.")

    # NOTE: COSMOS
    # TODO: We need to add the FSDP, but currently I failed to find the correct module name.
    # Maybe this is because the torch.jit or RecurrentScriptModule is used.

    # print("==================================")
    # for name, module in model.vae.named_modules():
    #     print(name, module)
    # print("==================================")

    # vae_encoder = getattr(model.vae._enc_model, "encoder", None)
    # if vae_encoder and hasattr(vae_encoder, "down"):
    #     for i in range(len(vae_encoder.down)):
    #         group_plan.append((f"vae._enc_model.encoder.down.{i}", False))
    # else:
    #     logger.warning("COSMOS-DV encoder does not have `down` attribute.")

    modules = recursive_list_modules(model.vae)
    print("Modules found:\n", "\n".join(modules))

    for module_path in modules:
        if "encoder.down" in module_path:
            group_plan.append((module_path, False))
        elif "encoder.mid" in module_path:
            group_plan.append((module_path, False))

    logger.info(f"The `group_plan` for FSDP (layer-level granularity):\n{group_plan}")
    return group_plan


def recursive_list_modules(module, prefix=""):
    modules = []
    if isinstance(module, (torch.nn.Module, torch.jit.RecursiveScriptModule)):
        modules.append(prefix.strip("."))
        for name in dir(module):
            if not name.startswith("_"):
                try:
                    child = getattr(module, name)
                    if isinstance(
                        child, (torch.nn.Module, torch.jit.RecursiveScriptModule)
                    ):
                        modules.extend(
                            recursive_list_modules(child, prefix=f"{prefix}.{name}")
                        )
                except AttributeError:
                    pass
    return modules


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
