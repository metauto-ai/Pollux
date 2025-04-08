# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Tuple
import logging
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    CLIPTokenizer,
    CLIPModel,
)

logger = logging.getLogger()


@dataclass
class TextEncoderArgs:
    config_name: str = "ViT-B/32"
    dtype: str = "bf16"
    text_seqlen: int = 77
    model_path: str = ""


class BaseTextEncoder:
    def __init__(self, args: TextEncoderArgs):
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[args.dtype]
        self.text_seqlen = args.text_seqlen

    # TODO: use this to get the dimension of the text encoder for transformer
    def dim(self) -> int:
        raise NotImplementedError

    def __call__(self, batch: dict[str:any]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class CLIP(BaseTextEncoder):
    def __init__(self, args: TextEncoderArgs):
        super().__init__(args)

        self.clip_model = CLIPModel.from_pretrained(
            (
                "openai/clip-vit-large-patch14"
                if args.model_path == ""
                else args.model_path
            ),
            torch_dtype=self.dtype,
        ).cuda()
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            (
                "openai/clip-vit-large-patch14"
                if args.model_path == ""
                else args.model_path
            ),
        )

    def dim(self) -> int:
        return self.clip_model.config.hidden_size

    def __call__(self, batch: dict[str:any]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert "caption" in batch
        if isinstance(batch["caption"][0], tuple):
            batch["caption"] = [x[0] for x in batch["caption"]]
        for idx, x in enumerate(batch["caption"]):
            if not isinstance(x, str):
                logger.warning(f"Expected string but got {type(x)}: {x}")
                batch["caption"][idx] = ""
        with torch.no_grad():
            inputs = self.tokenizer(
                batch["caption"], return_tensors="pt", padding=True, truncation=True
            ).to(self.clip_model.device)
            outputs = self.clip_model.text_model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs.attention_mask
        return last_hidden_state, attention_mask


class Qwen2_5_VL(BaseTextEncoder):
    def __init__(self, args: TextEncoderArgs):
        super().__init__(args)
        self.model = AutoModel.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct" if args.model_path == "" else args.model_path,
            torch_dtype=self.dtype,
        ).cuda()
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct" if args.model_path == "" else args.model_path,
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct" if args.model_path == "" else args.model_path,
        )

    def dim(self) -> int:
        return self.model.config.hidden_size

    def _convert_caption_to_messages(self, caption: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an assistant designed to generate high-quality images based on user prompts.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption},
                ],
            },
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, batch: dict[str:any]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert "caption" in batch
        if isinstance(batch["caption"][0], tuple):
            batch["caption"] = [x[0] for x in batch["caption"]]
        with torch.no_grad():
            messages = [
                self._convert_caption_to_messages(caption)
                for caption in batch["caption"]
            ]
            inputs = self.processor(
                text=messages,
                padding=True,
                return_tensors="pt",
                max_length=self.text_seqlen,
                truncation=True,
            ).to(self.model.device)

            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs.attention_mask

        return last_hidden_state, attention_mask


def create_text_encoder(args: TextEncoderArgs) -> BaseTextEncoder:
    if args.config_name == "ViT-B/32":
        return CLIP(args)
    elif args.config_name == "Qwen/Qwen2.5-VL-3B-Instruct":
        return Qwen2_5_VL(args)
    else:
        raise ValueError(f"Unknown text encoder: {args.config_name}")
