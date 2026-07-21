"""Stage 6: End-to-end pipeline verification.

Full pipeline:
  z_0 → AWGN(SNR) → z_noisy → MaskDecoder → predicted mask → DDIM inpaint → output

Compares GT mask vs predicted mask inpainting at multiple SNRs.

Usage:
    python scripts/06_e2e_verify.py
"""

import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sc_pri.utils import load_config
from sc_pri.vae import load_vae, decode
from sc_pri.channel import Channel
from sc_pri.models.mask_decoder import MaskDecoder
from sc_pri.diffusion import load_sd_components, ddim_inpaint


def decode_to_numpy(vae, latent, scale_factor):
    img = decode(vae, latent, scale_factor=scale_factor)
    img = ((img.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0)
    return img.cpu().numpy().astype(np.uint8)


def mask_to_overlay(img, mask_64, H):
    """Green overlay from (1,1,64,64) mask onto (H,W,3) image."""
    mask_up = F.interpolate(mask_64, size=H, mode="bilinear", align_corners=False)
    mask_up = mask_up.squeeze().cpu().numpy()
    overlay = img.copy().astype(np.float32)
    overlay[..., 1] = np.clip(overlay[..., 1] + mask_up * 180, 0, 255)
    return overlay.astype(np.uint8)


def run_e2e(cache_dir, out_dir, vae, sd_components, model, cfg,
            snr_list=(20, 10, 5, 0), n_samples=5, num_steps=50,
            prompt="a blurred face", guidance_scale=7.5):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[:n_samples]

    if not files:
        print(f"No .pt files found in {cache_dir}")
        return

    device = sd_components["device"]
    scale_factor = cfg["latent"]["scale_factor"]
    channel = Channel()
    model.eval()

    for i, f in enumerate(files):
        data = torch.load(f, map_location="cpu")
        latent = data["latent"].float().unsqueeze(0).to(device)
        gt_mask = data["mask"].float()

        # GT combined mask
        gt_combined = (gt_mask.sum(dim=0, keepdim=True) > 0.3).float()
        gt_combined = gt_combined.unsqueeze(0).to(device)  # (1,1,64,64)

        if gt_combined.sum() < 1:
            continue

        # Clean decoded
        img_clean = decode_to_numpy(vae, latent, scale_factor)
        H = img_clean.shape[0]

        for snr_db in snr_list:
            print(f"  [{i}] SNR={snr_db} dB")

            # Channel noise
            z_noisy, _ = channel(latent, snr_db=float(snr_db))
            img_noisy = decode_to_numpy(vae, z_noisy, scale_factor)

            # MaskDecoder prediction
            with torch.no_grad():
                pred_logits = model(z_noisy)
                pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
                pred_combined = (pred_mask.sum(dim=1, keepdim=True) > 0).float()

            # Inpaint with GT mask
            z_inpaint_gt = ddim_inpaint(
                z_0=z_noisy, mask_64=gt_combined,
                sd_components=sd_components,
                num_inference_steps=num_steps, strength=1.0,
                prompt=prompt, guidance_scale=guidance_scale,
                seed=42 + i,
            )
            img_inpaint_gt = decode_to_numpy(vae, z_inpaint_gt, scale_factor)

            # Inpaint with predicted mask
            z_inpaint_pred = ddim_inpaint(
                z_0=z_noisy, mask_64=pred_combined,
                sd_components=sd_components,
                num_inference_steps=num_steps, strength=1.0,
                prompt=prompt, guidance_scale=guidance_scale,
                seed=42 + i,
            )
            img_inpaint_pred = decode_to_numpy(vae, z_inpaint_pred, scale_factor)

            # Overlays
            gt_overlay = mask_to_overlay(img_noisy, gt_combined, H)
            pred_overlay = mask_to_overlay(img_noisy, pred_combined, H)

            # 6-panel: clean | noisy | gt_mask | gt_inpaint | pred_mask | pred_inpaint
            panels = np.concatenate([
                img_clean, img_noisy,
                gt_overlay, img_inpaint_gt,
                pred_overlay, img_inpaint_pred,
            ], axis=1)

            out_path = os.path.join(out_dir, f"e2e_{i:03d}_snr{snr_db}.png")
            Image.fromarray(panels).save(out_path)

    print(f"\nPanels: clean | noisy | GT_mask | GT_inpaint | pred_mask | pred_inpaint")
    print(f"Wrote to {out_dir}")


def main():
    cfg = load_config("configs/data.yaml")
    device = cfg["vae"]["device"]
    num_classes = len(cfg["classes"])

    print("Loading VAE...")
    vae = load_vae(cfg["vae"]["model_id"], device=device)

    print("Loading SD components...")
    sd_components = load_sd_components(device=device)

    print("Loading MaskDecoder...")
    model = MaskDecoder(in_channels=4, out_channels=num_classes, base=64).to(device)
    ckpt_path = "checkpoints/mask_decoder_oi/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt['epoch']}, "
          f"IoU {ckpt['best_mean_iou']:.4f})")

    cache_dir = os.path.join(cfg["cache"]["root"], cfg["cache"]["train_subdir"])
    out_dir = "debug/vis_e2e"

    run_e2e(
        cache_dir, out_dir, vae, sd_components, model, cfg,
        snr_list=[20, 10, 5, 0],
        n_samples=5,
        num_steps=50,
    )


if __name__ == "__main__":
    main()