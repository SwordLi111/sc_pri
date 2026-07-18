"""Visualization tools for sanity-checking cached data."""
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sc_pri.vae import load_vae, decode


def visualize_cached_samples(cache_dir, out_dir, vae, n_samples=20,
                              scale_factor=0.18215):
    """For each cached sample, decode latent and overlay masks; save side-by-side PNG.
    
    Color coding (adapt if you change class order):
      - class 0 (face)  → GREEN overlay
      - class 1 (plate) → RED overlay
    """
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[:n_samples]
    
    if not files:
        print(f"No .pt files found in {cache_dir}")
        return
    
    device = next(vae.parameters()).device
    
    for i, f in enumerate(files):
        data = torch.load(f, map_location="cpu")
        latent = data["latent"].float().unsqueeze(0).to(device)  # (1,4,64,64)
        mask = data["mask"].float()  # (C, 64, 64)
        
        img = decode(vae, latent, scale_factor=scale_factor)
        img = ((img.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = img.astype(np.uint8)  # (H, W, 3)
        H = img.shape[0]
        
        # Upsample mask to image resolution
        mask_up = F.interpolate(
            mask.unsqueeze(0), size=H,
            mode="bilinear", align_corners=False
        ).squeeze(0).clamp(0, 1).numpy()  # (C, H, H)
        
        overlay = img.copy().astype(np.float32)
        if mask_up.shape[0] >= 1:
            # face → green
            overlay[..., 1] = np.clip(overlay[..., 1] + mask_up[0] * 180, 0, 255)
        if mask_up.shape[0] >= 2:
            # plate → red
            overlay[..., 0] = np.clip(overlay[..., 0] + mask_up[1] * 180, 0, 255)
        overlay = overlay.astype(np.uint8)
        
        # Also show original for reference
        orig = Image.open(data["filepath"]).convert("RGB")
        orig_np = np.array(orig.resize((H, H)))
        
        # 3-panel: orig | decoded | overlay
        combined = np.concatenate([orig_np, img, overlay], axis=1)
        Image.fromarray(combined).save(os.path.join(out_dir, f"vis_{i:03d}.png"))
    
    print(f"Wrote {len(files)} visualizations to {out_dir}")
    print("Panels (left→right): original | VAE-decoded | mask overlay")
    print("Check:")
    print("  - GREEN overlay should be on faces")
    print("  - RED overlay should be on license plates")
    print("  - No misalignment, no all-black or all-covered masks")