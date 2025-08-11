from dataclasses import dataclass
import torch.nn as nn
import torch
from transformers import AutoModel, Siglip2VisionModel
from typing import Dict, List, Tuple, Union


@dataclass
class VisionEncoderArgs:
    encoder_name: str = 'siglip2'
    weight_path: str = 'google/siglip2-so400m-patch16-naflex'
    projection_hidden_dim: int = 1024
    dtype: str = "bfloat16"


class BaseVisionEncoder:
    def __init__(self, args: VisionEncoderArgs):
        self.args = args
        self.dtype = dict(fp32=torch.float32, bfloat16=torch.bfloat16)[args.dtype]

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        raise NotImplementedError
    
    @property
    def dim(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def extract_image_representations(self, batch: dict[str:any], flops_meter= None) -> Union[torch.Tensor, List[torch.Tensor]]:
        images = batch["image"]
        if isinstance(images, torch.Tensor):
            if flops_meter is not None:
                flops_meter.log_vision_encoder_flops(images.shape)
            return self.forward(images)
        elif isinstance(images, list):
            grouped_images: Dict[Tuple[int, int], List[Tuple[int, torch.Tensor]]] = {}
            for i, img in enumerate(images):
                resolution = (img.shape[-2], img.shape[-1])
                if resolution not in grouped_images:
                    grouped_images[resolution] = []
                img_with_batch = img if img.dim() == 4 else img.unsqueeze(0)
                grouped_images[resolution].append((i, img_with_batch))

            results = [None] * len(images)
            for resolution, indexed_tensors in grouped_images.items():
                indices = [item[0] for item in indexed_tensors]
                tensors = [item[1] for item in indexed_tensors]

                input_batch = torch.cat(tensors, dim=0)

                if flops_meter is not None:
                    flops_meter.log_vision_encoder_flops(input_batch.shape)

                feature_batch = self.forward(input_batch)

                for i, features in enumerate(feature_batch):
                    original_index = indices[i]
                    results[original_index] = features

            return results
        else:
            raise TypeError(f"Unsupported type for batch['image']: {type(images)}")


class DINOv2VisionEncoder(BaseVisionEncoder):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__(args)
        self.model = AutoModel.from_pretrained(
            args.weight_path if args.weight_path else "facebook/dinov2-base", 
            use_flash_attention_2=True,
            torch_dtype=self.dtype
        ).requires_grad_(False).cuda()
        self.model = self.model.eval()
        self.model = torch.compile(self.model)

    @property
    def dim(self):
        return self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        '''
        Expects image resized by factor of 224/256 to get same number of patches as siglip2 and vaes
        '''
        image = image.to(self.model.device, dtype=self.model.dtype)
        inputs = {
            "pixel_values": image,
        }

        outputs = self.model(**inputs)
        # ignore the cls token
        return outputs.last_hidden_state[:, 1:, :]
    

class Siglip2VisionEncoder(BaseVisionEncoder):
    def __init__(self, args: VisionEncoderArgs):
        super().__init__(args)
        self.model = Siglip2VisionModel.from_pretrained(
            args.weight_path if args.weight_path else "google/siglip2-so400m-patch16-naflex", 
            attn_implementation="flash_attention_2",
            torch_dtype=self.dtype
        ).requires_grad_(False).cuda()
        self.model = self.model.eval()
        self.model = torch.compile(self.model)

    @property
    def dim(self):
        return self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        b, c, h, w = image.shape
        image = image.to(self.model.device, dtype=self.dtype)
        patched_image = image.view(b, c, h // 16, 16, w // 16, 16)           # [B, C, H/16, 16, W/16, 16]
        patched_image = patched_image.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // 16) * (w // 16), -1)   # [B, L, D]
        
        inputs = {
            "pixel_values": patched_image,
            "pixel_attention_mask": torch.ones(b, (h // 16) * (w // 16), dtype=torch.int32, device=self.model.device),
            "spatial_shapes": torch.tensor([[h // 16, w // 16]]).repeat(b, 1).to(self.model.device, dtype=torch.int64)
        }

        outputs = self.model(**inputs)
        return outputs.last_hidden_state


encoder_dict = {
    "siglip2": Siglip2VisionEncoder,
    "dinov2-base": DINOv2VisionEncoder,
}


def create_vision_encoder(args: VisionEncoderArgs):
    if args.encoder_name not in encoder_dict:
        raise ValueError(f"Unsupported encoder: {args.encoder_name}, supported encoders: {encoder_dict.keys()}")
    return encoder_dict[args.encoder_name](args)
