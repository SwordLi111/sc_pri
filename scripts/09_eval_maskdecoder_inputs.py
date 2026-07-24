"""Stage 9: MaskDecoder IoU on three inputs -- clean z0 / noisy y1 / denoised z_hat.

Answers the pipeline-ordering question: should the relay detect BEFORE or
AFTER denoising?  Produces a per-SNR, per-class IoU table for:
    clean    -- upper bound (SNR-independent, computed once)
    noisy    -- detect-first ordering  (y1 -> MaskDecoder)
    denoised -- denoise-first ordering (y1 -> ddim_denoise -> MaskDecoder)

Denoised variant uses the trained LoRA (checkpoints/lora_denoise/best) if
present, in the adapter mode recorded in best_info.txt; falls back to the
frozen UNet otherwise.

IoU is aggregated dataset-wide (sum of intersections / sum of unions), not
batch-averaged, so empty-mask samples do not distort the numbers.

Usage:
    python scripts/09_eval_maskdecoder_inputs.py
"""

import os
import glob

import torch

from sc_pri.utils import load_config
from sc_pri.channel import Channel
from sc_pri.models.mask_decoder import MaskDecoder
from sc_pri.diffusion import (
    load_sd_components, ddim_denoise, estimate_sigma_c2_from_snr,
)


EVAL_SNRS = [0, 5, 10, 15, 20]
N_SAMPLES = 200
NUM_DDIM_STEPS = 50
MASKDEC_CKPT = "checkpoints/mask_decoder_oi/best.pt"
LORA_DIR = "checkpoints/lora_denoise/best"
LORA_INFO = "checkpoints/lora_denoise/best_info.txt"
DEVICE = "cuda"
SEED = 42


def load_lora_if_present(sd):
    """Attach trained LoRA to sd['unet'] if available. Returns adapter mode."""
    first_step_only = False
    if os.path.isdir(LORA_DIR):
        sd["unet"].load_lora_adapter(LORA_DIR)
        mode = "full"
        if os.path.exists(LORA_INFO):
            with open(LORA_INFO) as f:
                txt = f.read()
            if "mode=first" in txt:
                mode = "first"
        first_step_only = (mode == "first")
        print(f"LoRA loaded from {LORA_DIR} (mode={mode})")
    else:
        print("No LoRA found; denoised variant uses the frozen UNet")
    return first_step_only


class IoUAccum:
    """Dataset-wide per-class IoU: sum(inter) / sum(union)."""

    def __init__(self, num_classes):
        self.inter = torch.zeros(num_classes)
        self.union = torch.zeros(num_classes)

    def add(self, pred_bin, gt_bin):
        dims = (0, 2, 3)
        inter = (pred_bin * gt_bin).sum(dims).cpu()
        union = pred_bin.sum(dims).cpu() + gt_bin.sum(dims).cpu() - inter
        self.inter += inter
        self.union += union

    def value(self):
        return (self.inter / self.union.clamp(min=1e-8)).tolist()


@torch.no_grad()
def predict(model, z):
    logits = model(z)
    return (torch.sigmoid(logits) > 0.5).float()


def main():
    torch.manual_seed(SEED)
    cfg = load_config("configs/data.yaml")
    num_classes = len(cfg["classes"])
    class_names = [c["name"] for c in cfg["classes"]]

    # MaskDecoder
    model = MaskDecoder(in_channels=4, out_channels=num_classes,
                        base=64).to(DEVICE).eval()
    ckpt = torch.load(MASKDEC_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"MaskDecoder from {MASKDEC_CKPT} "
          f"(epoch {ckpt['epoch']}, IoU {ckpt.get('best_mean_iou', -1):.4f})")

    # SD + optional LoRA
    sd = load_sd_components(device=DEVICE)
    first_step_only = load_lora_if_present(sd)

    cache_dir = os.path.join(cfg["cache"]["root"], cfg["cache"]["val_subdir"])
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))[:N_SAMPLES]
    print(f"{len(files)} val samples, SNRs {EVAL_SNRS}\n")

    channel = Channel()

    acc_clean = IoUAccum(num_classes)
    acc_noisy = {s: IoUAccum(num_classes) for s in EVAL_SNRS}
    acc_deno = {s: IoUAccum(num_classes) for s in EVAL_SNRS}

    for i, f in enumerate(files):
        data = torch.load(f, map_location="cpu")
        z0 = data["latent"].float().unsqueeze(0).to(DEVICE)
        gt = (data["mask"].float().unsqueeze(0).to(DEVICE) > 0.5).float()

        acc_clean.add(predict(model, z0), gt)

        for snr in EVAL_SNRS:
            y1, _ = channel(z0, snr_db=float(snr))
            acc_noisy[snr].add(predict(model, y1), gt)

            s2 = estimate_sigma_c2_from_snr(y1, snr)[0].item()
            z_hat, _ = ddim_denoise(
                y1, s2, sd, num_inference_steps=NUM_DDIM_STEPS,
                prompt="", guidance_scale=1.0,
                adapter_first_step_only=first_step_only)
            acc_deno[snr].add(predict(model, z_hat), gt)

        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(files)}")

    # ---------------- report ----------------
    print("\n" + "=" * 74)
    hdr = f"{'SNR':>4} {'input':>9}"
    for n in class_names:
        hdr += f" {n[:14]:>15}"
    hdr += f" {'mean':>8}"
    print(hdr)

    def row(label, snr_label, vals):
        line = f"{snr_label:>4} {label:>9}"
        for v in vals:
            line += f" {v:>15.4f}"
        line += f" {sum(vals)/len(vals):>8.4f}"
        print(line)

    row("clean", "-", acc_clean.value())
    print("-" * 74)
    for snr in EVAL_SNRS:
        row("noisy", str(snr), acc_noisy[snr].value())
        row("denoised", "", acc_deno[snr].value())
    print("=" * 74)
    print("\nDecision rule: if 'denoised' beats 'noisy' at low SNR, the relay")
    print("pipeline is denoise -> detect; otherwise detect on y1 directly.")


if __name__ == "__main__":
    main()