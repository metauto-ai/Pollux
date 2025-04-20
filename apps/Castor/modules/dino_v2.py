from dataclasses import dataclass
import torch.nn as nn
import torch
from transformers import AutoImageProcessor, AutoModel


@dataclass
class DINOv2Args:
    weight_path: str = 'facebook/dinov2-base'
    align_hidden_dim: int = 1024
    align_encoder_dim: int = 768
    align_layer: int = 12


class DINOv2(nn.Module):
    def __init__(self, args: DINOv2Args):
        super(DINOv2, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(args.weight_path)
        self.model = AutoModel.from_pretrained(args.weight_path, use_flash_attention_2=True).requires_grad_(False)

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        # denormalize the image from PolluxImageProcessor
        image = image.to(torch.float32) * 0.5 + 0.5

        b, c, h, w = image.shape
        inputs = self.processor(
            images=image, 
            return_tensors="pt", 
            size={'height': h * 224 // 256, 'width': w * 224 // 256}, 
            do_normalize=True,
            do_resize=True,

            # PolluxImageProcessor already does the center crop and rescale
            do_center_crop=False,
            do_rescale=False,
        ).to(self.model.device)

        outputs = self.model(**inputs)
        # ignore the cls token
        return outputs.last_hidden_state[:, 1:, :]  # [B, H/pH * W/pW, C]


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
