import argparse
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
from dataclasses import dataclass, field

# Registry to store all available filter classes
FILTER_REGISTRY = {}

def register_filter(name):
    """
    Decorator to register a filter class in the FILTER_REGISTRY.
    """
    def decorator(cls):
        FILTER_REGISTRY[name] = cls
        return cls
    return decorator


@dataclass
class ImgFilterArgs:
    model_name: str = "WaterMarkFilter"  # ["CLIPFilter"]
    pretrained_model_name_or_path: str = (
        "/jfs/checkpoints/data_preprocessing/watermark_model_v1.pt"
    )


class BaseFilter(nn.Module):
    def __init__(self):
        super().__init__()

    def predict(self, batch_frame_tensor: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError


@register_filter("WaterMarkFilter")
class WaterMarkFilter(BaseFilter):
    def __init__(self, args: ImgFilterArgs):
        self.cfg=args

        model = timm.create_model("efficientnet_b3a", pretrained=True, num_classes=2)

        model.classifier = nn.Sequential(
            # 1536 is the orginal in_features
            nn.Linear(in_features=1536, out_features=625),
            nn.ReLU(),  # ReLu to be the activation function
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
        )

        state_dict = torch.load(self.cfg.pretrained_model_name_or_path)

        model.load_state_dict(state_dict)
        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        self.preprocessing = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.model = model

    @torch.no_grad()
    def predict(self, batch_frame_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_frame_tensor = self.preprocessing(batch_frame_tensor)
        pred = self.model(batch_frame_tensor)
        syms = F.softmax(pred, dim=1)
        score = syms[:, 1].cpu()  # Probability of 'clear_sym' wo watermark
        return score


@register_filter("CLIPFilter")
class CLIPFilter(BaseFilter):
    def __init__(self, args: ImgFilterArgs):
        pass


def build_filter(args: ImgFilterArgs) -> BaseFilter:
    """
    Factory function to build the appropriate filter based on the model name using the registry.
    """
    if args.model_name in FILTER_REGISTRY:
        return FILTER_REGISTRY[args.model_name](args)
    else:
        raise ValueError(
            f"Unsupported filter model: {args.model_name}. Available filters: {list(FILTER_REGISTRY.keys())}"
        )
