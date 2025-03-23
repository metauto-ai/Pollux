import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


# Dummy Dataset Class for diffusion
class DummyDataLoad(Dataset):
    def __init__(
        self,
        num_samples=1000,
        num_classes=10,
        image_size=(3, 64, 64),
        word_count=256,
    ):
        """
        Initialize the dummy dataset.
        Args:
            num_samples (int): Number of samples in the dataset.
            num_classes (int): Number of distinct classes.
            image_size (tuple): Shape of the dummy images (C, H, W).
            word_count: Number of the word in this caption
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.word_count = word_count

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a random sample.
        Returns:
            class_idx (int): A random class index.
            time_step (int): A random time step for the diffusion process.
            image (torch.Tensor): A random image tensor.
        """
        class_idx = np.random.randint(0, self.num_classes)
        image = torch.randn(self.image_size)  # Random image tensor
        caption = generate_random_text()
        batch = {"label": class_idx, "caption": caption, "image": image}
        return batch
