"""Stage 4: Verify latent-space DDIM inpainting.

For a few cached samples:
  1. Load latent z_0 and mask
  2. Run DDIM repaint inpainting (mask region → generated, outside → original)
  3. Save 4-panel comparison:
     original | decoded_original | mask_overlay | inpainted

Usage:
    python scripts/04_verify_inpaint.py
"""

import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sc_pri.utils import load_config
from sc_pri.vae import load_vae, decode
from sc_pri.diffusion import load_sd_components, ddim_inpaint


def run_verification(cache_dir, out_dir, vae, sd_components, cfg,
                     n_samples=10, num_steps=50, strength=1.0,
                     prompt="a blurred face", guidance_scale=7.5):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[:n_samples]

    if not files:
        print(f"No .pt files found in {cache_dir}")
        return

    device = sd_components["device"]
    scale_factor = cfg["latent"]["scale_factor"]

    for i, f in enumerate(files):
        data = torch.load(f, map_location="cpu")
        latent = data["latent"].float().unsqueeze(0).to(device)  # (1,4,64,64)
        mask = data["mask"].float()  # (C, 64, 64)

        # Combine all class masks into one binary mask (1 = replace)
        # Any class with mask > 0.3 counts as "sensitive"
        combined_mask = (mask.sum(dim=0, keepdim=True) > 0.3).float()  # (1, 64, 64)
        combined_mask = combined_mask.unsqueeze(0).to(device)  # (1, 1, 64, 64)

        # Skip if no mask content
        if combined_mask.sum() < 1:
            print(f"  [{i}] No mask content, skipping")
            continue

        print(f"  [{i}] Mask coverage: {combined_mask.mean().item()*100:.1f}%, running inpaint...")

        # Run DDIM inpainting
        z_inpainted = ddim_inpaint(
            z_0=latent,
            mask_64=combined_mask,
            sd_components=sd_components,
            num_inference_steps=num_steps,
            strength=strength,
            prompt=prompt,
            guidance_scale=guidance_scale,
            seed=42 + i,
        )

        # Decode both original and inpainted
        img_orig = decode(vae, latent, scale_factor=scale_factor)
        img_orig = ((img_orig.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0)
        img_orig = img_orig.cpu().numpy().astype(np.uint8)

        img_inpaint = decode(vae, z_inpainted, scale_factor=scale_factor)
        img_inpaint = ((img_inpaint.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0)
        img_inpaint = img_inpaint.cpu().numpy().astype(np.uint8)

        H = img_orig.shape[0]

        # Mask overlay on original
        mask_up = F.interpolate(
            combined_mask, size=H, mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        overlay = img_orig.copy().astype(np.float32)
        overlay[..., 1] = np.clip(overlay[..., 1] + mask_up * 180, 0, 255)
        overlay = overlay.astype(np.uint8)

        # Original photo (if available)
        try:
            orig_photo = Image.open(data["filepath"]).convert("RGB")
            orig_photo = np.array(orig_photo.resize((H, H)))
        except Exception:
            orig_photo = np.zeros_like(img_orig)

        # 4-panel: photo | decoded_original | mask_overlay | inpainted
        combined = np.concatenate([orig_photo, img_orig, overlay, img_inpaint], axis=1)
        out_path = os.path.join(out_dir, f"inpaint_{i:03d}.png")
        Image.fromarray(combined).save(out_path)
        print(f"  [{i}] Saved {out_path}")

    print(f"\nWrote visualizations to {out_dir}")
    print("Panels (left→right): photo | decoded | mask | inpainted")
    print("Check:")
    print("  - Mask region should have different/generated content")
    print("  - Non-mask region should be identical to decoded original")


def main():
    cfg = load_config("configs/data.yaml")
    device = cfg["vae"]["device"]

    # Load VAE
    print("Loading VAE...")
    vae = load_vae(cfg["vae"]["model_id"], device=device)

    # Load SD components (UNet, scheduler, text encoder)
    print("Loading SD 1.5 components...")
    sd_components = load_sd_components(device=device)

    cache_dir = os.path.join(cfg["cache"]["root"], cfg["cache"]["train_subdir"])
    out_dir = "debug/vis_inpaint"

    run_verification(
        cache_dir, out_dir, vae, sd_components, cfg,
        n_samples=10,
        num_steps=50,
        strength=1.0,
        prompt="a blurred face",
        guidance_scale=7.5,
    )


if __name__ == "__main__":
    main()