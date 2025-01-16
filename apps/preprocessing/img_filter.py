import argparse
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
from dataclasses import dataclass, field
from torchmetrics.multimodal import CLIPScore
from typing import Literal


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
    model_name: Literal["WaterMarkFilter", "CLIPFilter"] = "WaterMarkFilter"
    pretrained_model_name_or_path: Literal[
        "/jfs/checkpoints/data_preprocessing/watermark_model_v1.pt",
        "/jfs/checkpoints/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3/",
    ] = "/jfs/checkpoints/data_preprocessing/watermark_model_v1.pt"


class BaseFilter(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def predict(self, image: Image.Image, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError


@register_filter("WaterMarkFilter")
class WaterMarkFilter(BaseFilter):
    def __init__(self, args: ImgFilterArgs):
        super().__init__()
        self.cfg=args

        model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=2)

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
    def predict(self, image: Image.Image, **kwargs) -> torch.FloatTensor:
        """
        Predict the watermark probability for a single PIL Image.

        Args:
            image (Image.Image): A PIL Image object.

        Returns:
            torch.FloatTensor: Probability of 'clear' (no watermark).
        """
        # Apply preprocessing to the input image
        processed_image = self.preprocessing(image).unsqueeze(0).cuda()  
        # Add batch dimension

        # Predict the scores
        pred = self.model(processed_image)

        # Apply softmax to get probabilities
        syms = F.softmax(pred, dim=1)

        # Extract the probability for 'clear_sym'
        score = syms[0, 1].cpu()  # Scalar probability for 'clear' class
        return score


@register_filter("CLIPFilter")
class CLIPFilter(BaseFilter):
    # Code reference: https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html
    def __init__(self, args: ImgFilterArgs):
        super().__init__()
        self.cfg = args
        self.clip_score_metric = CLIPScore(model_name_or_path=self.cfg.pretrained_model_name_or_path).cuda()
        self.preprocessing = T.Compose(
            [
                T.ToTensor(),
                T.Lambda(lambda x: (x * 255).to(torch.uint8)),
            ]
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, **kwargs) -> torch.FloatTensor:
        """
        Predict the CLIP score for a single PIL Image using the CLIPScore metric.

        Args:
            image (Image.Image): A PIL Image object.
            **kwargs: Additional keyword arguments (e.g., 'prompts').

        Returns:
            torch.FloatTensor: The CLIP score as a scalar tensor.
        """
        prompt = kwargs.get("prompt")
        if prompt is None:
            raise ValueError("The 'prompt' parameter is required for CLIPFilter.")

        prompt=[prompt]

        # Convert PIL Image to PyTorch tensor
        image_tensor = self.preprocessing(image).unsqueeze(0).cuda()  # Add batch dimension

        # Calculate CLIP score using the metric
        clip_score_value = self.clip_score_metric(image_tensor, prompt)

        return clip_score_value.cpu().detach().round() # Return the score as a tensor on CPU


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
