"""Dataset for cached latent + mask .pt files."""

import os
import glob

import torch
from torch.utils.data import Dataset


class LatentMaskDataset(Dataset):
    """Load cached (latent, mask) pairs from .pt files.
    
    Each .pt contains:
        latent: (4, 64, 64) fp16
        mask:   (C, 64, 64) fp16
    """

    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if not self.files:
            raise ValueError(f"No .pt files found in {cache_dir}")
        print(f"LatentMaskDataset: {len(self.files)} samples from {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        latent = data["latent"].float()  # (4, 64, 64)
        mask = data["mask"].float()      # (C, 64, 64)
        return latent, mask