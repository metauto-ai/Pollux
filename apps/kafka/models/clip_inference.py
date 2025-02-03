import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

class CLIPInference:
    def __init__(self, model_name="openai/clip-vit-base-patch16", device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def compute_clip_score(self, batch):
        """
        Compute the CLIP score for a batch of images and captions.

        Args:
            batch (dict): A dictionary with keys "input_ids", "attention_mask", "pixel_values"

        Returns:
            torch.Tensor: A tensor of shape (batch_size,) containing the CLIP scores.
        """

        # Move batch to device
        for key in batch:
            batch[key] = batch[key].to(self.device)

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**batch)

        # Normalize features
        image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, dim=-1)

        # Compute cosine similarity directly between corresponding pairs
        # Using einsum for efficient pairwise multiplication
        logits = 100 * torch.einsum('bd,bd->b', image_embeds, text_embeds)

        return logits
