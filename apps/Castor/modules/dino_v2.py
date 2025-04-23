from dataclasses import dataclass
import torch.nn as nn
import torch
from transformers import AutoModel
from typing import Dict, List, Tuple, Union


@dataclass
class DINOv2Args:
    weight_path: str = 'facebook/dinov2-base'
    align_hidden_dim: int = 1024
    align_encoder_dim: int = 768


class DINOv2:
    def __init__(self, args: DINOv2Args):
        self.model = AutoModel.from_pretrained(
            args.weight_path, 
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16
        ).requires_grad_(False).cuda()

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        image = image.to(self.model.device, dtype=self.model.dtype)
        inputs = {
            "pixel_values": image,
        }

        outputs = self.model(**inputs)
        # ignore the cls token
        return outputs.last_hidden_state[:, 1:, :]
    
    def get_features(self, batch: dict[str:any]) -> Union[torch.Tensor, List[torch.Tensor]]:
        images = batch["image_cond"]
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


class DINOv2_Proj(nn.Module):
    def __init__(self, dim:int, args: DINOv2Args):
        super(DINOv2_Proj, self).__init__()
        
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
