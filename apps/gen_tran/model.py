# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import logging
import random
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
from apps.main.modules.tokenizer import Tokenizer, TokenizerArgs
from apps.main.modules.schedulers import RectifiedFlow, SchedulerArgs
from apps.main.modules.gen_transformer import (
    BaseDiffusionTransformer,
    RotaryEmbedding2D,
    RotaryEmbedding1D,
    GenTransformerArgs,
)
from apps.main.modules.plan_transformer import (
    PlanTransformerArgs,
)
from apps.main.modules.vae import LatentVideoVAE, LatentVideoVAEArgs
from apps.main.modules.preprocess import random_mask_images
import os
from apps.main.modules.embedder import LabelEmbedder, ImageEmbedder, TimestepEmbedder
from apps.main.modules.ops import AdaLN as AdaLNModulation, modulate
from lingua.transformer import (
    BaseTransformerArgs,
    TransformerBlock,
    RMSNorm,
    InitStdFactor,
)
from apps.main.modules.ops import create_causal_mask

logger = logging.getLogger()


@dataclass
class PlanTransformerArgs(BaseTransformerArgs):

    seed: int = 42
    in_channels: int = 3
    pre_trained_path: Optional[str] = None
    text_seqlen: int = 256
    vocab_size: int = -1


@dataclass
class ModelArgs:
    type: str = "pollux"  # "pollux" or "latent_pollux"
    gen_transformer: GenTransformerArgs = field(default_factory=GenTransformerArgs)
    plan_transformer: PlanTransformerArgs = field(default_factory=PlanTransformerArgs)
    vae: LatentVideoVAEArgs = field(default_factory=LatentVideoVAEArgs)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    text_cfg_ratio: float = 0.1
    image_cfg_ratio: float = 0.1
    mask_patch: int = 16
    num_classes: int = 1


class GenTransformer(BaseDiffusionTransformer):
    """
    Diffusion Transformer capable of handling both images and video sequences (in the future).
    Uses patchify for images and a similar approach for video (flattening spatial and temporal dims).
    """

    def __init__(self, args: GenTransformerArgs):
        super().__init__(args)
        self.patch_size = args.patch_size
        self.out_channels = args.out_channels
        self.in_channels = args.in_channels
        self.tmb_embed = TimestepEmbedder(
            hidden_size=args.ada_dim, time_embedding_size=args.tmb_size
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
        self.ada_dim = args.ada_dim
        self.dim = args.dim
        self.cos_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.coe_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

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

    def forward(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        condition: torch.Tensor,
        attn_impl: str = "sdpa",
    ):
        x, img_size, freqs_cis_img = self.patchify_and_embed_image(x)
        x_l = x.size(1)
        if len(condition.shape) == 2:
            modulation_signal = condition + self.tmb_embed(time_steps)
        else:
            modulation_signal = torch.mean(
                condition, dim=1, keepdim=False
            ) + self.tmb_embed(time_steps)
        condition_ = torch.cat(
            [
                self.cos_token.repeat(len(condition), 1, 1),
                condition.unsqueeze(1) if len(condition.shape) == 2 else condition,
                self.coe_token.repeat(len(condition), 1, 1),
            ],
            dim=1,
        )

        c_l = condition_.size(1)

        freqs_cis_img = freqs_cis_img.to(x.device)
        freqs_cis_cond = self.rope_embeddings_conditions.freqs_cis[:c_l].to(x.device)
        x = torch.cat([condition_, x], dim=1)
        freqs_cis = torch.cat([freqs_cis_cond, freqs_cis_img], dim=0)

        h = super().forward(x, freqs_cis, modulation_signal, attn_impl=attn_impl)

        h = h[:, -x_l:, :]

        out = self.img_output(self.norm(h))

        x = self.unpatchify_image(out, img_size)

        return x

    def unpatchify_image(
        self, x: torch.Tensor, img_size: Tuple[int, int]
    ) -> torch.Tensor:
        pH = pW = self.patch_size
        H, W = img_size
        B = x.size(0)
        L = (H // pH) * (W // pW)
        x = x[:, :L].view(B, H // pH, W // pW, pH, pW, self.out_channels)
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
        nn.init.normal_(self.cos_token, std=0.02)
        nn.init.normal_(self.coe_token, std=0.02)


class BasePlanTransformer(nn.Module):
    def __init__(self, args: PlanTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.layers = nn.ModuleList()
        assert not (args.n_layers % 2 != 0)
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        h,
        freqs_cis,
        attn_impl: str = "sdpa",
    ):
        seq_len = h.size(1)
        mask = create_causal_mask(seq_len, attn_impl)
        for idx, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, mask=mask, attn_impl=attn_impl)
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


class PlanTransformer(
    BasePlanTransformer
):  # TODO As planning model is not finished, we use a pure LLAMA model here, we will update this with the latest planning  model
    def __init__(self, args: PlanTransformerArgs):
        super().__init__(args)
        self.text_seqlen = args.text_seqlen
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.rope_embeddings_cap = RotaryEmbedding1D(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.text_seqlen,
        )
        self.dim = args.dim
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        batch: dict[str:any],
        attn_impl: str = "sdpa",
    ):
        x_cap = self.tok_embeddings(batch["cap_token"])

        freqs_cis = self.rope_embeddings_cap.freqs_cis[: x_cap.size(1)]
        h = super().forward(x_cap, freqs_cis, attn_impl=attn_impl)

        return self.norm(h)

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        self.rope_embeddings_cap.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )


class Pollux(nn.Module):
    """
    Latent Diffusion Transformer Model.
    This model integrates a VAE for latent compression, a transformer for temporal and spatial token mixing,
    and a custom scheduler for diffusion steps.
    """

    VERSION: str = "v0.7"
    DESCRIPTION: str = (
        "Latent Diffusion Transformer for VideoGen: (1) currently we only support class conditional image generation for debugging."
    )

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.compressor = LatentVideoVAE(args.vae)
        self.scheduler = RectifiedFlow(args.scheduler)
        self.gen_transformer = GenTransformer(args.gen_transformer)
        self.tokenizer = Tokenizer(model_path=args.tokenizer.model_path)

        assert args.plan_transformer.vocab_size == self.tokenizer.n_words

        self.plan_transformer = PlanTransformer(args.plan_transformer)
        self.text_seqlen = self.plan_transformer.text_seqlen
        self.text_cfg_ratio = args.text_cfg_ratio
        self.token_proj = nn.Linear(
            in_features=args.plan_transformer.dim,
            out_features=args.gen_transformer.dim,
            bias=False,
        )
        self.negative_token = nn.Parameter(torch.zeros(1, 1, args.plan_transformer.dim))
        init_std = self.gen_transformer.dim ** (-0.5)
        nn.init.trunc_normal_(
            self.token_proj.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        nn.init.normal_(self.negative_token, std=0.02)

    def cap_tokenize(self, batch: dict[str:any]) -> dict[str:any]:
        batch["cap_token"] = [
            self.tokenizer.encode(x, bos=True, eos=False) for x in batch["caption"]
        ]
        pad_id = self.tokenizer.pad_id
        bsz = len(batch["cap_token"])
        tokens = torch.full(
            (bsz, self.plan_transformer.text_seqlen),
            pad_id,
            dtype=torch.long,
        ).cuda()
        for k, t in enumerate(batch["cap_token"]):
            if len(t) < tokens.size(1):
                tokens[k, : len(t)] = torch.tensor(
                    t[:], dtype=torch.long, device="cuda"
                )
            else:
                tokens[k, :] = torch.tensor(
                    t[: tokens.size(1)], dtype=torch.long, device="cuda"
                )
        batch["cap_token"] = tokens
        return batch

    def prepare_negative_context(self, batch: dict[str:any]) -> dict[str:any]:
        batch["negative_token"] = self.negative_token.repeat(
            batch["text_embedding"].size(0), batch["text_embedding"].size(1), 1
        )
        return batch["negative_token"]

    def forward(self, batch: dict[str:any]) -> dict[str:any]:

        image = batch["image"]
        batch = self.cap_tokenize(batch)
        with torch.no_grad():
            batch["text_embedding"] = self.plan_transformer(batch)
        if random.random() > self.text_cfg_ratio:
            conditional_signal = batch["text_embedding"]
        else:
            conditional_signal = self.prepare_negative_context(batch)
        with torch.no_grad():
            latent_code = self.compressor.encode(image)
        conditional_signal = self.token_proj(conditional_signal)
        noised_x, t, target = self.scheduler.sample_noised_input(latent_code)
        output = self.gen_transformer(
            x=noised_x, time_steps=t, condition=conditional_signal
        )
        batch["prediction"] = output
        batch["target"] = target
        target = target.to(output.dtype)
        loss = F.mse_loss(output, target)

        return batch, loss

    def set_train(self):
        self.gen_transformer.train()

    def set_eval(self):
        self.gen_transformer.eval()

    def init_weights(self, args: ModelArgs):
        self.gen_transformer.init_weights(args.gen_transformer.pre_trained_path)
        self.plan_transformer.init_weights(args.plan_transformer.pre_trained_path)
        self.plan_transformer = self.plan_transformer.requires_grad_(False)
        self.plan_transformer.eval()


class LatentPollux(nn.Module):
    """
    Latent Diffusion Transformer Model.
    This model drops VAE for latent compression, as it is already pre-processed, a transformer for temporal and spatial token mixing,
    and a custom scheduler for diffusion steps.
    """

    VERSION: str = "v0.7"
    DESCRIPTION: str = (
        "Latent Diffusion Transformer for VideoGen: (1) currently we only support class conditional image generation for debugging."
    )

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.scheduler = RectifiedFlow(args.scheduler)
        self.gen_transformer = GenTransformer(args.gen_transformer)
        self.text_cfg_ratio = args.text_cfg_ratio
        self.token_proj = nn.Linear(
            in_features=args.plan_transformer.dim,
            out_features=args.gen_transformer.dim,
            bias=False,
        )
        init_std = self.gen_transformer.dim ** (-0.5)
        nn.init.trunc_normal_(
            self.token_proj.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        self.negative_token = nn.Parameter(torch.zeros(1, 1, args.plan_transformer.dim))
        nn.init.normal_(self.negative_token, std=0.02)

    def forward(self, batch: dict[str:any]) -> dict[str:any]:
        if random.random() > self.text_cfg_ratio:
            conditional_signal = batch["text_embedding"]
        else:
            conditional_signal = self.negative_token.repeat(
                batch["text_embedding"].size(0), batch["text_embedding"].size(1), 1
            )
        latent_code = batch["latent_code"]
        conditional_signal = self.token_proj(conditional_signal)
        noised_x, t, target = self.scheduler.sample_noised_input(latent_code)
        output = self.gen_transformer(
            x=noised_x, time_steps=t, condition=conditional_signal
        )
        batch["prediction"] = output
        batch["target"] = target
        target = target.to(output.dtype)
        loss = F.mse_loss(output, target)

        return batch, loss

    def set_train(self):
        self.gen_transformer.train()

    def set_eval(self):
        self.gen_transformer.eval()

    def init_weights(self, args: ModelArgs):
        self.gen_transformer.init_weights(args.gen_transformer.pre_trained_path)


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan_pollux(model_args: ModelArgs, vae_config: dict):
    group_plan: Tuple[int, bool] = []

    for i in range(len(vae_config.down_block_types)):
        group_plan.append((f"compressor.vae.encoder.down_blocks.{i}", False))

    for i in range(model_args.plan_transformer.n_layers):
        group_plan.append((f"plan_transformer.layers.{i}", False))

    for i in range(model_args.gen_transformer.n_layers):
        group_plan.append((f"gen_transformer.layers.{i}", False))

    group_plan.append(("gen_transformer.img_output", True))
    logger.info(f"The `group_plan` for fsdp is:\n{group_plan}")

    return group_plan


def build_fsdp_grouping_plan_latent_pollux(model_args: ModelArgs):
    group_plan: Tuple[int, bool] = []

    for i in range(model_args.gen_transformer.n_layers):
        group_plan.append((f"gen_transformer.layers.{i}", False))

    group_plan.append(("gen_transformer.img_output", True))
    logger.info(f"The `group_plan` for fsdp is:\n{group_plan}")

    return group_plan


def tp_parallelize(model, tp_mesh, model_args: ModelArgs, distributed_args):

    # assert model_args.plan_transformer.dim % distributed_args.tp_size == 0
    # assert model_args.plan_transformer.vocab_size % distributed_args.tp_size == 0
    # assert model_args.plan_transformer.n_heads % distributed_args.tp_size == 0
    # assert (model_args.plan_transformer.n_kv_heads or 0) % distributed_args.tp_size == 0
    # assert model_args.plan_transformer.n_heads % (model_args.n_kv_heads or 1) == 0

    assert model_args.gen_transformer.dim % distributed_args.tp_size == 0
    assert model_args.gen_transformer.vocab_size % distributed_args.tp_size == 0
    assert model_args.gen_transformer.n_heads % distributed_args.tp_size == 0
    assert (model_args.gen_transformer.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.gen_transformer.n_heads % (model_args.n_kv_heads or 1) == 0

    main_plan = {}
    main_plan["norm"] = SequenceParallel()
    main_plan["img_output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model.gen_transformer,
        tp_mesh,
        main_plan,
    )

    # TODO: Adding plan_transformer Modules
    for layer in model.gen_transformer.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers TP
        # Feedforward layers TP
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
