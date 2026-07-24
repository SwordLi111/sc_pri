"""Stage 7: Verify training-free channel denoising via SNR -> t* mapping.

For each cached sample and each SNR in EVAL_SNRS:
  1. y1 = Channel(z0, snr)
  2. sigma_c2 two ways:
       (a) ORACLE:   mean((y1 - z0)^2)            -- ground truth, offline only
       (b) DEPLOYED: P_y1 / (1 + SNR_linear)      -- what the relay can compute
     Report their agreement (they must match closely; this validates that
     deployment needs no access to z0).
  3. z_hat = ddim_denoise(y1, sigma_c2_deployed, ...)
  4. Metrics:
       latent MSE:  MSE(y1, z0)  vs  MSE(z_hat, z0)   -> denoising gain
       image PSNR:  decode(y1)   vs  decode(z_hat), both against decode(z0)
  5. Save 3-panel PNG per (sample, SNR): clean | noisy | denoised

Success criterion: MSE(z_hat, z0) < MSE(y1, z0) and PSNR up, at every SNR.
Expect the gain to grow as SNR drops (more noise to remove); at SNR=20 dB
the mapping may land below the DDIM discretization and change little.

Usage:
    python scripts/07_verify_denoise.py
"""

import os
import glob

import numpy as np
import torch
from PIL import Image

from sc_pri.utils import load_config
from sc_pri.vae import load_vae, decode
from sc_pri.channel import Channel
from sc_pri.diffusion import (
    load_sd_components, ddim_denoise,
    estimate_sigma_c2_from_snr, snr_to_timestep,
)


EVAL_SNRS = [0, 5, 10, 15, 20]
N_SAMPLES = 8
NUM_STEPS = 50          # DDIM discretization of the full schedule
OUT_DIR = "debug/vis_denoise"
SEED = 42


def decode_to_numpy(vae, latent, scale_factor):
    img = decode(vae, latent, scale_factor=scale_factor)
    img = ((img.clamp(-1, 1) + 1) * 127.5).squeeze(0).permute(1, 2, 0)
    return img.cpu().numpy().astype(np.uint8)


def psnr(a, b):
    """PSNR between two uint8 HxWx3 images."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    torch.manual_seed(SEED)
    cfg = load_config("configs/data.yaml")
    device = cfg["vae"]["device"]
    scale_factor = cfg["latent"]["scale_factor"]

    print("Loading VAE...")
    vae = load_vae(cfg["vae"]["model_id"], device=device)
    print("Loading SD components...")
    sd = load_sd_components(device=device)
    LORA_DIR = "/home/jli038/code/SC_PRI_clean/checkpoints/lora_domain/best"
    LORA_WEIGHT = "pytorch_lora_weights.safetensors"    
    print(f"Loading domain LoRA: {LORA_DIR}/{LORA_WEIGHT}")
    sd["unet"].load_lora_adapter(
        LORA_DIR,
        weight_name=LORA_WEIGHT,
        adapter_name="lora_domain",
        prefix=None,
    )
    sd["unet"].set_adapters(
    ["lora_domain"],
    adapter_weights=[1.0],
    )
    sd["unet"].eval()
    
    assert hasattr(sd["unet"], "peft_config")
    assert "lora_domain" in sd["unet"].peft_config
    print("LoRA adapters:", list(sd["unet"].peft_config.keys()))


    cache_dir = os.path.join(cfg["cache"]["root"], cfg["cache"]["val_subdir"])
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[:N_SAMPLES]
    if not files:
        raise SystemExit(f"No .pt files in {cache_dir}")

    os.makedirs(OUT_DIR, exist_ok=True)
    channel = Channel()

    # results[snr] = list of dicts per sample
    results = {snr: [] for snr in EVAL_SNRS}

    for i, f in enumerate(files):
        data = torch.load(f, map_location="cpu")
        z0 = data["latent"].float().unsqueeze(0).to(device)  # (1,4,64,64)
        img_clean = decode_to_numpy(vae, z0, scale_factor)

        for snr_db in EVAL_SNRS:
            y1, _ = channel(z0, snr_db=float(snr_db))

            # --- sigma_c2: oracle vs deployed ---
            sigma_oracle = (y1 - z0).pow(2).mean().item()
            sigma_deployed = estimate_sigma_c2_from_snr(y1, snr_db)[0].item()
            rel_err = abs(sigma_deployed - sigma_oracle) / max(sigma_oracle, 1e-12)

            t_star, abar = snr_to_timestep(sigma_deployed, sd["scheduler"])

            # --- denoise (deployed estimate, as the relay would) ---
            z_hat, _ = ddim_denoise(
                y1, sigma_deployed, sd,
                num_inference_steps=NUM_STEPS,
                prompt="", guidance_scale=1.0,
            )

            # --- latent-domain metrics ---
            mse_noisy = (y1 - z0).pow(2).mean().item()
            mse_denoised = (z_hat - z0).pow(2).mean().item()

            # --- image-domain metrics ---
            img_noisy = decode_to_numpy(vae, y1, scale_factor)
            img_denoised = decode_to_numpy(vae, z_hat, scale_factor)
            psnr_noisy = psnr(img_noisy, img_clean)
            psnr_denoised = psnr(img_denoised, img_clean)

            results[snr_db].append({
                "sigma_rel_err": rel_err,
                "t_star": t_star,
                "mse_noisy": mse_noisy,
                "mse_denoised": mse_denoised,
                "psnr_noisy": psnr_noisy,
                "psnr_denoised": psnr_denoised,
            })

            # --- 3-panel: clean | noisy | denoised ---
            panels = np.concatenate([img_clean, img_noisy, img_denoised], axis=1)
            Image.fromarray(panels).save(
                os.path.join(OUT_DIR, f"denoise_{i:03d}_snr{snr_db:02d}.png"))

            print(f"[{i}] SNR={snr_db:2d}dB  t*={t_star:3d}  "
                  f"sigma_err={rel_err*100:5.2f}%  "
                  f"MSE {mse_noisy:.5f}->{mse_denoised:.5f}  "
                  f"PSNR {psnr_noisy:5.2f}->{psnr_denoised:5.2f} dB")

    # ---------------- summary ----------------
    print("\n" + "=" * 78)
    print(f"{'SNR':>4} {'t* (avg)':>9} {'sig err%':>9} "
          f"{'MSE noisy':>11} {'MSE denoise':>12} "
          f"{'PSNR noisy':>11} {'PSNR denoise':>13} {'gain dB':>8}")
    for snr_db in EVAL_SNRS:
        r = results[snr_db]
        avg = lambda k: sum(x[k] for x in r) / len(r)
        gain = avg("psnr_denoised") - avg("psnr_noisy")
        print(f"{snr_db:>4} {avg('t_star'):>9.0f} {avg('sigma_rel_err')*100:>9.2f} "
              f"{avg('mse_noisy'):>11.5f} {avg('mse_denoised'):>12.5f} "
              f"{avg('psnr_noisy'):>11.2f} {avg('psnr_denoised'):>13.2f} "
              f"{gain:>+8.2f}")
    print("=" * 78)
    print(f"\nPanels (left->right): clean | noisy | denoised  ->  {OUT_DIR}/")
    print("Checks:")
    print("  - sig err% small (<~2%): SNR-only estimate is deployment-viable")
    print("  - MSE denoised < MSE noisy at every SNR")
    print("  - PSNR gain largest at low SNR; near zero at 20 dB is expected")


if __name__ == "__main__":
    main()