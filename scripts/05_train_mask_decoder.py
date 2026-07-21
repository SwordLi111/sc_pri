"""Train MaskDecoder on cached OI latents with random SNR channel noise.

Each batch:
  1. Load clean latents + GT masks
  2. Sample random SNR per sample, add AWGN
  3. Feed noisy latents to MaskDecoder
  4. Supervise with BCE + Dice loss against GT masks

Validates at fixed SNR levels [0, 5, 10, 15, 20] dB.

Usage:
    python scripts/05_train_mask_decoder.py
"""

import os
import time

import torch
from torch.utils.data import DataLoader

from sc_pri.utils import load_config, bce_dice_loss, iou
from sc_pri.channel import Channel
from sc_pri.models.mask_decoder import MaskDecoder
from sc_pri.data.latent_dataset import LatentMaskDataset


# ---------- Config ----------
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 1e-3
SNR_RANGE = (0.0, 20.0)
EVAL_SNRS = [0, 5, 10, 15, 20]
SAVE_DIR = "checkpoints/mask_decoder_oi"
DEVICE = "cuda"


def train_one_epoch(model, loader, optimizer, channel, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for latent, mask in loader:
        latent = latent.to(device)  # (B, 4, 64, 64)
        mask = mask.to(device)      # (B, C, 64, 64)

        # Add random SNR channel noise
        noisy_latent, snr_used = channel(latent)

        # Forward
        pred_logits = model(noisy_latent)  # (B, C, 64, 64)
        loss = bce_dice_loss(pred_logits, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, channel, device, eval_snrs):
    model.eval()
    results = {}

    for snr_db in eval_snrs:
        total_iou = None
        n_batches = 0

        for latent, mask in loader:
            latent = latent.to(device)
            mask = mask.to(device)

            noisy_latent, _ = channel(latent, snr_db=float(snr_db))
            pred_logits = model(noisy_latent)
            pred_binary = (torch.sigmoid(pred_logits) > 0.5).float()

            batch_iou = iou(pred_binary, (mask > 0.5).float())  # (C,)
            if total_iou is None:
                total_iou = batch_iou
            else:
                total_iou = total_iou + batch_iou
            n_batches += 1

        mean_iou = total_iou / n_batches  # (C,)
        results[snr_db] = mean_iou.cpu()

    return results


def main():
    cfg = load_config("configs/data.yaml")
    num_classes = len(cfg["classes"])
    class_names = [c["name"] for c in cfg["classes"]]

    cache_root = cfg["cache"]["root"]
    train_dir = os.path.join(cache_root, cfg["cache"]["train_subdir"])
    val_dir = os.path.join(cache_root, cfg["cache"]["val_subdir"])

    # Datasets
    train_ds = LatentMaskDataset(train_dir)
    val_ds = LatentMaskDataset(val_dir)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = MaskDecoder(in_channels=4, out_channels=num_classes, base=64).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MaskDecoder: {total_params:.1f}M params, {num_classes} classes")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    channel = Channel(snr_db_range=SNR_RANGE)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_mean_iou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, channel, DEVICE)
        val_results = validate(model, val_loader, channel, DEVICE, EVAL_SNRS)

        scheduler.step()

        # Print results
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} ({elapsed:.1f}s)  loss={train_loss:.4f}")
        print(f"  {'SNR':>5s}", end="")
        for name in class_names:
            print(f"  {name:>15s}", end="")
        print(f"  {'mean':>8s}")

        epoch_ious = []
        for snr_db in EVAL_SNRS:
            iou_per_class = val_results[snr_db]
            print(f"  {snr_db:>3d}dB", end="")
            for c in range(num_classes):
                print(f"  {iou_per_class[c].item():>15.4f}", end="")
            mean = iou_per_class.mean().item()
            print(f"  {mean:>8.4f}")
            epoch_ious.append(mean)

        overall_mean = sum(epoch_ious) / len(epoch_ious)
        print(f"  Overall mean IoU across SNRs: {overall_mean:.4f}")

        # Save best
        if overall_mean > best_mean_iou:
            best_mean_iou = overall_mean
            save_path = os.path.join(SAVE_DIR, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mean_iou": best_mean_iou,
            }, save_path)
            print(f"  -> New best! Saved to {save_path}")

        # Save periodic checkpoint
        if epoch % 20 == 0:
            save_path = os.path.join(SAVE_DIR, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)

    print(f"\nTraining complete. Best mean IoU: {best_mean_iou:.4f}")


if __name__ == "__main__":
    main()