"""Stage 6 (v2): End-to-end two-stage pipeline verification.

Pipeline (relay-side privacy gateway):
    z0 -> AWGN(SNR) -> y1
       -> MaskDecoder                          [detect on y1 by default]
       -> ddim_denoise (t*-mapping + LoRA)     [restoration]
       -> ddim_inpaint on the DENOISED latent  [sanitization]
       -> decode

Changes vs v1:
    - denoising stage inserted; inpaint context is z_hat, not y1
    - prompt "" (was "a blurred face"), guidance 3.0 (was 7.5)
    - predicted mask dilated 1 latent pixel before repaint
    - LoRA auto-loaded from checkpoints/lora_domain/best if present
    - reads from val cache, prefers samples with actual privacy content

Panels per (sample, SNR), 6 columns left to right:
    clean | noisy | denoised | pred-mask overlay | inpaint(pred) | inpaint(GT)

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
from sc_pri.diffusion import (
    load_sd_components, ddim_denoise, ddim_inpaint,
    estimate_sigma_c2_from_snr,
)


SNR_LIST = [20, 10, 5, 0]
N_SAMPLES = 5
NUM_DDIM_STEPS = 50
INPAINT_STEPS = 50
INPAINT_PROMPT = ""
INPAINT_GUIDANCE = 3.0
DETECT_ON = "noisy"                       # "noisy" or "denoised"
MASKDEC_CKPT = "checkpoints/mask_decoder_oi/best.pt"
LORA_CANDIDATES = [
    "checkpoints/lora_domain/best",
    "checkpoints/lora_denoise/best",
]
OUT_DIR = "debug/vis_e2e_v2"
DEVICE = "cuda"
SEED = 42


def decode_to_numpy(vae, latent, scale_factor):
    img = decode(vae, latent, scale_factor=scale_factor)
    img = ((img.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0)
    return img.cpu().numpy().astype(np.uint8)


def mask_to_overlay(img, mask_64, H):
    mask_up = F.interpolate(mask_64, size=H, mode="bilinear",
                            align_corners=False)
    mask_up = mask_up.squeeze().cpu().numpy()
    overlay = img.copy().astype(np.float32)
    overlay[..., 1] = np.clip(overlay[..., 1] + mask_up * 180, 0, 255)
    return overlay.astype(np.uint8)


def dilate(mask_1, ks=3):
    return F.max_pool2d(mask_1, kernel_size=ks, stride=1, padding=ks // 2)


def load_lora_if_present(sd):
    first_step_only = False
    for lora_dir in LORA_CANDIDATES:
        if os.path.isdir(lora_dir):
            sd["unet"].load_lora_adapter(
                lora_dir,
                weight_name="pytorch_lora_weights.safetensors",
            )
            info_path = os.path.join(os.path.dirname(lora_dir),
                                     "best_info.txt")
            mode = "full"
            if os.path.exists(info_path):
                with open(info_path) as f:
                    if "mode=first" in f.read():
                        mode = "first"
            first_step_only = (mode == "first")
            print(f"LoRA loaded from {lora_dir} (mode={mode})")
            return first_step_only
    print("No LoRA found; denoising uses the frozen UNet")
    return first_step_only


def main():
    torch.manual_seed(SEED)
    cfg = load_config("configs/data.yaml")
    scale_factor = cfg["latent"]["scale_factor"]
    num_classes = len(cfg["classes"])

    print("Loading VAE...")
    vae = load_vae(cfg["vae"]["model_id"], device=DEVICE)
    print("Loading SD components...")
    sd = load_sd_components(device=DEVICE)
    first_step_only = load_lora_if_present(sd)

    print("Loading MaskDecoder...")
    model = MaskDecoder(in_channels=4, out_channels=num_classes,
                        base=64).to(DEVICE).eval()
    ckpt = torch.load(MASKDEC_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"MaskDecoder from {MASKDEC_CKPT} "
          f"(epoch {ckpt['epoch']}, IoU {ckpt.get('best_mean_iou', -1):.4f})")

    cache_dir = os.path.join(cfg["cache"]["root"], cfg["cache"]["val_subdir"])
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    picked = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        if (d["mask"].float() > 0.3).sum() > 8:
            picked.append(f)
        if len(picked) >= N_SAMPLES:
            break
    if not picked:
        raise SystemExit("no val samples with privacy content found")
    print(f"Selected {len(picked)} val samples with privacy content")

    os.makedirs(OUT_DIR, exist_ok=True)
    channel = Channel()

    for i, f in enumerate(picked):
        data = torch.load(f, map_location="cpu")
        z0 = data["latent"].float().unsqueeze(0).to(DEVICE)
        gt_mask = data["mask"].float().unsqueeze(0).to(DEVICE)
        gt_combined = (gt_mask.sum(dim=1, keepdim=True) > 0.3).float()

        img_clean = decode_to_numpy(vae, z0, scale_factor)
        H = img_clean.shape[0]

        for snr_db in SNR_LIST:
            print(f"  [{i}] SNR={snr_db} dB")

            y1, _ = channel(z0, snr_db=float(snr_db))
            img_noisy = decode_to_numpy(vae, y1, scale_factor)

            s2 = estimate_sigma_c2_from_snr(y1, snr_db)[0].item()
            z_hat, _ = ddim_denoise(
                y1, s2, sd, num_inference_steps=NUM_DDIM_STEPS,
                prompt="", guidance_scale=1.0,
                adapter_first_step_only=first_step_only)
            img_deno = decode_to_numpy(vae, z_hat, scale_factor)

            det_input = y1 if DETECT_ON == "noisy" else z_hat
            with torch.no_grad():
                logits = model(det_input)
                pred = (torch.sigmoid(logits) > 0.5).float()
                pred_combined = (pred.sum(dim=1, keepdim=True) > 0).float()
            pred_dilated = dilate(pred_combined, ks=3)

            z_san_pred = ddim_inpaint(
                z_0=z_hat, mask_64=pred_dilated, sd_components=sd,
                num_inference_steps=INPAINT_STEPS, strength=1.0,
                prompt=INPAINT_PROMPT, guidance_scale=INPAINT_GUIDANCE,
                seed=SEED + i)
            img_san_pred = decode_to_numpy(vae, z_san_pred, scale_factor)

            z_san_gt = ddim_inpaint(
                z_0=z_hat, mask_64=dilate(gt_combined, ks=3),
                sd_components=sd,
                num_inference_steps=INPAINT_STEPS, strength=1.0,
                prompt=INPAINT_PROMPT, guidance_scale=INPAINT_GUIDANCE,
                seed=SEED + i)
            img_san_gt = decode_to_numpy(vae, z_san_gt, scale_factor)

            overlay = mask_to_overlay(img_deno, pred_dilated, H)

            panels = np.concatenate([
                img_clean, img_noisy, img_deno,
                overlay, img_san_pred, img_san_gt,
            ], axis=1)
            Image.fromarray(panels).save(
                os.path.join(OUT_DIR, f"e2e_{i:03d}_snr{snr_db:02d}.png"))

    print(f"\nPanels: clean | noisy | denoised | pred-mask | "
          f"inpaint(pred) | inpaint(GT)")
    print(f"Wrote to {OUT_DIR}  (DETECT_ON={DETECT_ON})")


if __name__ == "__main__":
    main()