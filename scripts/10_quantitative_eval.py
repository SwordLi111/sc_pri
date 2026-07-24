"""Stage 10: Paper-grade quantitative evaluation of restoration quality.

Methods compared at each SNR in EVAL_SNRS, on N_SAMPLES val images:
    noisy      -- no processing (decode y1 directly)
    tfree      -- training-free t*-mapping denoise (frozen UNet)
    lora       -- + trained LoRA (skipped if checkpoint absent)

Metrics (decoded-image domain, vs. decode(z0)):
    PSNR (dB), SSIM, LPIPS (AlexNet)
plus latent-domain MSE vs z0.

Output: one table per metric, methods as columns -- paste-ready for the paper.

Requires:  pip install lpips scikit-image
Usage:     python scripts/10_quantitative_eval.py
"""

import os
import glob

import numpy as np
import torch

import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn

from sc_pri.utils import load_config
from sc_pri.vae import load_vae, decode
from sc_pri.channel import Channel
from sc_pri.diffusion import (
    load_sd_components, ddim_denoise, estimate_sigma_c2_from_snr,
)


EVAL_SNRS = [0, 5, 10, 15, 20]
N_SAMPLES = 200
NUM_DDIM_STEPS = 50
LORA_DIR = "checkpoints/lora_denoise/best"
LORA_INFO = "checkpoints/lora_denoise/best_info.txt"
DEVICE = "cuda"
SEED = 42


def decode_img(vae, latent, scale_factor):
    """-> HxWx3 uint8"""
    img = decode(vae, latent, scale_factor=scale_factor)
    img = ((img.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0)
    return img.cpu().numpy().astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def to_lpips_tensor(img_u8, device):
    """uint8 HWC -> (1,3,H,W) in [-1,1]"""
    t = torch.from_numpy(img_u8).float().permute(2, 0, 1).unsqueeze(0)
    return (t / 127.5 - 1.0).to(device)


def main():
    torch.manual_seed(SEED)
    cfg = load_config("configs/data.yaml")
    scale_factor = cfg["latent"]["scale_factor"]

    print("Loading VAE...")
    vae = load_vae(cfg["vae"]["model_id"], device=DEVICE)
    print("Loading SD components...")
    sd = load_sd_components(device=DEVICE)
    print("Loading LPIPS (AlexNet)...")
    lpips_model = lpips_lib.LPIPS(net="alex").to(DEVICE).eval()

    has_lora = os.path.isdir(LORA_DIR)
    lora_first = False
    if has_lora:
        sd["unet"].load_lora_adapter(LORA_DIR)
        if os.path.exists(LORA_INFO):
            with open(LORA_INFO) as f:
                lora_first = "mode=first" in f.read()
        sd["unet"].disable_adapters()   # start disabled; enable per-method
        print(f"LoRA available (first_step_only={lora_first})")
    methods = ["noisy", "tfree"] + (["lora"] if has_lora else [])

    cache_dir = os.path.join(cfg["cache"]["root"], cfg["cache"]["val_subdir"])
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[:N_SAMPLES]
    print(f"{len(files)} val samples, SNRs {EVAL_SNRS}, methods {methods}\n")

    channel = Channel()

    # res[metric][method][snr] = list of values
    metrics = ["mse_latent", "psnr", "ssim", "lpips"]
    res = {m: {meth: {s: [] for s in EVAL_SNRS} for meth in methods}
           for m in metrics}

    for i, f in enumerate(files):
        data = torch.load(f, map_location="cpu")
        z0 = data["latent"].float().unsqueeze(0).to(DEVICE)
        img_clean = decode_img(vae, z0, scale_factor)
        clean_lp = to_lpips_tensor(img_clean, DEVICE)

        for snr in EVAL_SNRS:
            y1, _ = channel(z0, snr_db=float(snr))
            s2 = estimate_sigma_c2_from_snr(y1, snr)[0].item()

            outputs = {"noisy": y1}

            if has_lora:
                sd["unet"].disable_adapters()
            z_tf, _ = ddim_denoise(y1, s2, sd,
                                   num_inference_steps=NUM_DDIM_STEPS,
                                   prompt="", guidance_scale=1.0)
            outputs["tfree"] = z_tf

            if has_lora:
                sd["unet"].enable_adapters()
                z_lo, _ = ddim_denoise(y1, s2, sd,
                                       num_inference_steps=NUM_DDIM_STEPS,
                                       prompt="", guidance_scale=1.0,
                                       adapter_first_step_only=lora_first)
                outputs["lora"] = z_lo

            for meth, z in outputs.items():
                res["mse_latent"][meth][snr].append(
                    (z - z0).pow(2).mean().item())
                img = decode_img(vae, z, scale_factor)
                res["psnr"][meth][snr].append(psnr(img, img_clean))
                res["ssim"][meth][snr].append(
                    ssim_fn(img_clean, img, channel_axis=2, data_range=255))
                with torch.no_grad():
                    d = lpips_model(clean_lp,
                                    to_lpips_tensor(img, DEVICE)).item()
                res["lpips"][meth][snr].append(d)

        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(files)}")

    # ---------------- tables ----------------
    for m in metrics:
        print(f"\n=== {m} ===")
        hdr = f"{'SNR':>4}"
        for meth in methods:
            hdr += f" {meth:>10}"
        print(hdr)
        for snr in EVAL_SNRS:
            line = f"{snr:>4}"
            for meth in methods:
                v = res[m][meth][snr]
                line += f" {sum(v)/len(v):>10.4f}"
            print(line)

    print("\nReading guide: PSNR/SSIM higher is better; LPIPS/MSE lower is "
          "better.\nExpect tfree >> noisy everywhere; lora vs tfree gap "
          "largest at SNR 15-20 (grid-mismatch correction).")


if __name__ == "__main__":
    main()