from dataclasses import dataclass
import torch.nn as nn
import torch
from transformers import Siglip2VisionModel
from typing import Dict, List, Tuple, Union


@dataclass
class VisionEncoderArgs:
    weight_path: str = 'google/siglip2-so400m-patch16-naflex'
    align_hidden_dim: int = 1024
    align_encoder_dim: int = 768


class VisionEncoder:
    def __init__(self, args: VisionEncoderArgs):
        self.model = Siglip2VisionModel.from_pretrained(
            args.weight_path, 
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16
        ).requires_grad_(False).cuda()

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        b, c, h, w = image.shape
        image = image.to(self.model.device, dtype=self.model.dtype)
        patched_image = image.view(b, c, h // 16, 16, w // 16, 16)           # [B, C, H/16, 16, W/16, 16]
        patched_image = patched_image.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // 16) * (w // 16), -1)   # [B, L, D]
        
        inputs = {
            "pixel_values": patched_image,
            "pixel_attention_mask": torch.ones(b, (h // 16) * (w // 16), dtype=torch.int32, device=self.model.device),
            "spatial_shapes": torch.tensor([[h // 16, w // 16]]).repeat(b, 1).to(self.model.device, dtype=torch.int64)
        }

        outputs = self.model(**inputs)
        # ignore the cls token
        return outputs.last_hidden_state
    
    def get_features(self, batch: dict[str:any]) -> Union[torch.Tensor, List[torch.Tensor]]:
        images = batch["image"]
        if isinstance(images, torch.Tensor):
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

                feature_batch = self.forward(input_batch)

                for i, features in enumerate(feature_batch):
                    original_index = indices[i]
                    results[original_index] = features

            return results
        else:
            raise TypeError(f"Unsupported type for batch['image_cond']: {type(images)}")


class VisionEncoder_Proj(nn.Module):
    def __init__(self, dim:int, args: VisionEncoderArgs):
        super(VisionEncoder_Proj, self).__init__()
        
        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(dim, args.align_hidden_dim),
                nn.SiLU(),
                nn.Linear(args.align_hidden_dim, args.align_hidden_dim),
                nn.SiLU(),
                nn.Linear(args.align_hidden_dim, args.align_encoder_dim),
            )
        )
        
    def forward(self, x):
        x = self.proj(x)
        return x
