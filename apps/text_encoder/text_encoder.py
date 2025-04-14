import ftfy
import torch
import html
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, UMT5EncoderModel



"""
Lumina-next:  "gemma-2b"
SANA:         "gemma-2-2b-it-sana"
WAN:          "umt5-xxl"
"""
class TextEncoder():
    def __init__(self, model_name, device='cuda', dtype=torch.float32):

        self.text_encoder_dict = {
            "gemma-2b": "google/gemma-2b", 
            "gemma-2b-it": "google/gemma-2b-it",
            "gemma-2-2b": "google/gemma-2-2b",
            "gemma-2-2b-it": "google/gemma-2-2b-it",
            "gemma-2-9b": "google/gemma-2-9b",
            "gemma-2-9b-it": "google/gemma-2-9b-it",
            "gemma-2-2b-it-sana": "/mnt/pollux/checkpoints/gemma-2-2b-it",
            "umt5-small":"google/umt5-small",
            "umt5-base":"google/umt5-base",
            "umt5-xl":"google/umt5-xl",
            "umt5-xxl": "/mnt/pollux/checkpoints/umt5-xxl",
        }

        self.model_name = model_name
        self.device=device
        self.dtype = dtype
        self.tokenizer, self.text_encoder = self.build_text_encoder()


    def build_text_encoder(self):
        if 'gemma' in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_dict[self.model_name])
            tokenizer.padding_side = "right"
            text_encoder =AutoModelForCausalLM.from_pretrained(
                                               self.text_encoder_dict[self.model_name], 
                                               torch_dtype=self.dtype
                                               ).to(self.device)

        elif "umt5" in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_dict[self.model_name])
            text_encoder = UMT5EncoderModel.from_pretrained(
                                                 self.text_encoder_dict[self.model_name], 
                                                 torch_dtype=self.dtype
                                                 ).to(self.device)
        else:
            print("error load text encoder")
            exit()
        return tokenizer, text_encoder

    def prompt_clean(self, text):
        # basic_clean
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = text.strip()
        # whitespace_clean
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text


    def get_t5_prompt_embeds(
        self, 
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [self.prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=max_sequence_length,
                        truncation=True,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    ).to(device=self.device)

        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids, mask)
        if 'gemma' in self.model_name:
            prompt_embeds = prompt_embeds[0]
        elif 'umt5' in self.model_name:
            prompt_embeds = prompt_embeds.last_hidden_state

        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds



if __name__ == "__main__":
    prompt = 'Write a hello world program'
    name = 'gemma-2-2b-it-sana'
    text_encoder = TextEncoder(name)
    prompt_embeds = text_encoder.get_t5_prompt_embeds(prompt)
    print (prompt_embeds.shape) # [1, 512, 256000]

    name = 'umt5-xxl'
    text_encoder = TextEncoder(name)
    prompt_embeds = text_encoder.get_t5_prompt_embeds(prompt)
    print (prompt_embeds.shape) # [1, 512, 4096]
