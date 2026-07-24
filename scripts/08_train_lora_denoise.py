"""Stage 12: Domain-adaptation LoRA -- standard diffusion fine-tuning with
channel-entry mixing and x0-space loss reweighting.

Rationale: stage-8 trained only at the trajectory ENTRY step, where the
epsilon objective is saturated by the frozen prior at low SNR. Here the LoRA
is trained the standard way -- random timestep t over the FULL schedule --
so every step the reverse trajectory passes through has been adapted to the
OI face/plate domain. This is legitimate precisely because the t* mapping
converts channel noise into standard diffusion noise at inference.

Batch composition (per sample):
    P_CHANNEL = 0.3 : channel-entry sample (stage-8 construction) -- keeps
                      the high-SNR grid-mismatch correction (-17%)
    else            : standard DSM: t ~ U{0..999}, eps ~ N(0, I)

Loss:
    per-sample weight  w = 1 + LAMBDA * min((1 - abar_t) / abar_t, W_CAP)
    (the exact reweighting induced by an x0-space MSE, numerically capped;
     pushes emphasis toward large t == low SNR)
    loss = mask-outside weighted MSE(eps_pred, eps_target) * w

Validation: identical protocol to stage 8 (fixed noise pack, live frozen
baseline, full-trajectory ddim_denoise) -- numbers directly comparable.
Success criterion: SNR0 MSE_safe below baseline within ~10 epochs.

Requires:  pip install peft
Usage:     python scripts/12_train_lora_domain.py
"""

import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from peft import LoraConfig

from sc_pri.utils import load_config
from sc_pri.channel import Channel
from sc_pri.data.latent_dataset import LatentMaskDataset
from sc_pri.diffusion import (
    load_sd_components, ddim_denoise, estimate_sigma_c2_from_snr,
    get_text_embedding,
)


# ---------- Config ----------
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4
LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
P_CHANNEL = 0.5              # fraction of channel-entry samples per batch
LAMBDA = 0.5                 # strength of x0-space reweighting
W_CAP = 20.0                 # cap on (1-abar)/abar before scaling
SNR_RANGE = (5.0, 15.0)
VAL_SNRS = [0, 10, 20]
N_VAL_SAMPLES = 16
NUM_DDIM_STEPS = 50
MASK_DILATE_KS = 5
SAVE_DIR = "checkpoints/lora_domain"
DEVICE = "cuda"
SEED = 42


def build_keep_mask(mask):
    privacy = (mask.sum(dim=1, keepdim=True) > 0.3).float()
    privacy = F.max_pool2d(privacy, kernel_size=MASK_DILATE_KS,
                           stride=1, padding=MASK_DILATE_KS // 2)
    return 1.0 - privacy


def grid_entry_batch(y1, snr_db, grid_ts, grid_abars):
    B = y1.shape[0]
    snr_lin = 10.0 ** (snr_db / 10.0)
    p_y = y1.reshape(B, -1).pow(2).mean(dim=1)
    sigma2 = p_y / (1.0 + snr_lin)
    target_abar = 1.0 / (1.0 + sigma2)
    idx = (grid_abars.unsqueeze(0) - target_abar.unsqueeze(1)).abs().argmin(dim=1)
    return grid_ts[idx], grid_abars[idx]


def train_one_epoch(unet, loader, optimizer, channel,
                    grid_ts, grid_abars, abars_full, text_emb, device):
    unet.train()
    total_loss, n_batches = 0.0, 0

    for latent, mask in loader:
        z0 = latent.to(device)
        mask = mask.to(device)
        B = z0.shape[0]

        # ---- standard DSM branch (all samples first) ----
        t_std = torch.randint(0, abars_full.shape[0], (B,), device=device)
        abar_std = abars_full[t_std]
        eps_std = torch.randn_like(z0)
        sa = abar_std.sqrt().view(B, 1, 1, 1)
        s1 = (1.0 - abar_std).clamp(min=1e-8).sqrt().view(B, 1, 1, 1)
        x_std = sa * z0 + s1 * eps_std

        # ---- channel-entry branch ----
        y1, snr_db = channel(z0)
        t_ch, abar_ch = grid_entry_batch(y1, snr_db, grid_ts, grid_abars)
        sa_c = abar_ch.sqrt().view(B, 1, 1, 1)
        s1_c = (1.0 - abar_ch).clamp(min=1e-8).sqrt().view(B, 1, 1, 1)
        x_ch = sa_c * y1
        eps_ch = (x_ch - sa_c * z0) / s1_c

        # ---- per-sample mix ----
        use_ch = (torch.rand(B, device=device) < P_CHANNEL)
        sel = use_ch.view(B, 1, 1, 1).float()
        x_t = sel * x_ch + (1 - sel) * x_std
        eps_target = sel * eps_ch + (1 - sel) * eps_std
        t_used = torch.where(use_ch, t_ch, t_std)
        abar_used = torch.where(use_ch, abar_ch, abar_std)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eps_pred = unet(x_t, t_used,
                            encoder_hidden_states=text_emb.expand(B, -1, -1)
                            ).sample

        # x0-space reweighting (capped), pushes emphasis to large t / low SNR
        ratio = ((1.0 - abar_used) / abar_used.clamp(min=1e-8)).clamp(max=W_CAP)
        w = (1.0 + LAMBDA * ratio).view(B, 1, 1, 1)

        keep = build_keep_mask(mask)
        err2 = (eps_pred.float() - eps_target).pow(2)
        weighted = err2 * keep * w
        norm = (keep * w).sum() * z0.shape[1] + 1e-8
        loss = weighted.sum() / norm

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in unet.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def build_val_pack(val_ds, channel, device, seed):
    torch.manual_seed(seed)
    pack = {snr: [] for snr in VAL_SNRS}
    n = min(N_VAL_SAMPLES, len(val_ds))
    for i in range(n):
        z0, mask = val_ds[i]
        z0 = z0.unsqueeze(0).to(device)
        keep = build_keep_mask(mask.unsqueeze(0).to(device))
        for snr in VAL_SNRS:
            y1, _ = channel(z0, snr_db=float(snr))
            s2 = estimate_sigma_c2_from_snr(y1, snr)[0].item()
            pack[snr].append((z0, y1, s2, keep))
    return pack


@torch.no_grad()
def validate(sd, val_pack):
    sd["unet"].eval()
    out = {}
    for snr, items in val_pack.items():
        sums = torch.zeros(3)
        counts = torch.zeros(3)
        for z0, y1, s2, keep in items:
            z_hat, _ = ddim_denoise(y1, s2, sd,
                                    num_inference_steps=NUM_DDIM_STEPS,
                                    prompt="", guidance_scale=1.0)
            err2 = (z_hat - z0).pow(2)
            priv = 1.0 - keep
            sums[0] += err2.mean().item(); counts[0] += 1
            mse_safe = (err2 * keep).sum() / (keep.sum() * z0.shape[1] + 1e-8)
            sums[1] += mse_safe.item(); counts[1] += 1
            if priv.sum() > 0:
                mse_priv = (err2 * priv).sum() / (
                    priv.sum() * z0.shape[1] + 1e-8)
                sums[2] += mse_priv.item(); counts[2] += 1
        out[snr] = (sums / counts.clamp(min=1)).tolist()
    return out


def fmt(res, base=None):
    parts = []
    for snr, (a, s, p) in res.items():
        line = f"SNR{snr}: all={a:.5f} safe={s:.5f} priv={p:.5f}"
        if base is not None:
            bs = base[snr][1]
            line += f" ({'-' if s < bs else '+'}{abs(s-bs)/bs*100:.1f}% safe)"
        parts.append(line)
    return "\n  ".join(parts)


def main():
    torch.manual_seed(SEED)
    cfg = load_config("configs/data.yaml")

    cache_root = cfg["cache"]["root"]
    train_ds = LatentMaskDataset(
        os.path.join(cache_root, cfg["cache"]["train_subdir"]))
    val_ds = LatentMaskDataset(
        os.path.join(cache_root, cfg["cache"]["val_subdir"]))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=1, pin_memory=True, drop_last=True)

    sd = load_sd_components(device=DEVICE)
    unet, scheduler = sd["unet"], sd["scheduler"]
    assert scheduler.config.prediction_type == "epsilon"

    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        init_lora_weights="gaussian", target_modules=LORA_TARGETS))
    for name, p in unet.named_parameters():
        if p.requires_grad:
            assert "lora" in name.lower(), f"unexpected trainable: {name}"
    trainable = [p for p in unet.parameters() if p.requires_grad]
    print(f"Domain LoRA rank {LORA_RANK} (P_channel={P_CHANNEL}, "
          f"lambda={LAMBDA}, cap={W_CAP}): "
          f"{sum(p.numel() for p in trainable)/1e6:.2f}M trainable")

    scheduler.set_timesteps(NUM_DDIM_STEPS, device=DEVICE)
    grid_ts = scheduler.timesteps.long().clone()
    abars_full = scheduler.alphas_cumprod.to(DEVICE).clone()
    grid_abars = abars_full[grid_ts].clone()

    text_emb = get_text_embedding(sd["tokenizer"], sd["text_encoder"],
                                  "", DEVICE)

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-2)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS)
    channel = Channel(snr_db_range=SNR_RANGE)
    os.makedirs(SAVE_DIR, exist_ok=True)

    val_pack = build_val_pack(val_ds, channel, DEVICE, seed=SEED + 1)
    unet.disable_adapters()
    baseline = validate(sd, val_pack)
    unet.enable_adapters()
    print("\nFrozen-UNet baseline (fixed val noise):\n  " + fmt(baseline))

    best_safe = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(unet, train_loader, optimizer, channel,
                                     grid_ts, grid_abars, abars_full,
                                     text_emb, DEVICE)
        scheduler_lr.step()
        res = validate(sd, val_pack)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} ({time.time()-t0:.0f}s)  "
              f"loss={train_loss:.5f}\n  " + fmt(res, baseline))

        mean_safe = sum(v[1] for v in res.values()) / len(res)
        unet.save_lora_adapter(os.path.join(SAVE_DIR, f"epoch_{epoch:03d}"))
        if mean_safe < best_safe:
            best_safe = mean_safe
            unet.save_lora_adapter(os.path.join(SAVE_DIR, "best"))
            with open(os.path.join(SAVE_DIR, "best_info.txt"), "w") as f:
                f.write(f"epoch={epoch} mode=full "
                        f"mean_mse_safe={best_safe:.6f}\n")
            print(f"  -> New best: mean MSE_safe={best_safe:.5f}")

    print(f"\nDone. Best mean MSE_safe: {best_safe:.5f}")


if __name__ == "__main__":
    main()