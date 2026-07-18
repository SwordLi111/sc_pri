"""Cache SD-VAE latents + multi-class soft masks for OI subset.

For each sample:
- Load image
- Build per-class binary mask in ORIGINAL pixel space:
    * If seg mask available for that instance, use it
    * Else fall back to bbox rectangle
- Letterbox both image and mask to configured size
- Encode image → latent (4, 64, 64)
- Downsample mask → soft mask (C, 64, 64) via avg_pool8
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import fiftyone as fo

from sc_pri.utils import (
    letterbox_image, letterbox_binary_mask,
    transform_bbox_letterbox,
)
from sc_pri.vae import load_vae, encode




def _should_skip(det, filter_cfg):
    """Check whether a detection should be filtered out."""
    if filter_cfg.get("is_group_of", True) and getattr(det, "IsGroupOf", False):
        return True
    if filter_cfg.get("is_depiction", True) and getattr(det, "IsDepiction", False):
        return True
    return False


def build_multiclass_mask_original(sample, img_h, img_w, class_to_idx, filter_cfg):
    """Build (C, H, W) binary mask in ORIGINAL image pixel space from bboxes."""
    C = len(class_to_idx)
    pixel_mask = np.zeros((C, img_h, img_w), dtype=np.uint8)

    det_field = sample.ground_truth
    if det_field is None or not det_field.detections:
        return pixel_mask

    for det in det_field.detections:
        if det.label not in class_to_idx:
            continue
        if _should_skip(det, filter_cfg):
            continue
        c = class_to_idx[det.label]
        x, y, w, h = det.bounding_box
        x1 = max(0, int(round(x * img_w)))
        y1 = max(0, int(round(y * img_h)))
        x2 = min(img_w, int(round((x + w) * img_w)))
        y2 = min(img_h, int(round((y + h) * img_h)))
        if x2 > x1 and y2 > y1:
            pixel_mask[c, y1:y2, x1:x2] = 1

    return pixel_mask


def process_sample(sample, vae, cfg, class_to_idx):
    """Process one fiftyone sample → dict with latent + mask."""
    img = Image.open(sample.filepath).convert("RGB")
    orig_w, orig_h = img.size
    img_np = np.array(img)
    
    # Build masks in original space
    pixel_mask = build_multiclass_mask_original(
        sample, orig_h, orig_w, class_to_idx, cfg["filter"]
    )
    
    # Letterbox image
    img_size = cfg["image"]["size"]
    pad_value = cfg["image"]["pad_value"]
    img_lb, _, _, _ = letterbox_image(img_np, new_size=img_size, pad_value=pad_value)
    
    # Letterbox each class's mask
    C = pixel_mask.shape[0]
    mask_lb = np.zeros((C, img_size, img_size), dtype=np.uint8)
    for c in range(C):
        if pixel_mask[c].sum() > 0:
            mask_lb[c] = letterbox_binary_mask(pixel_mask[c], new_size=img_size)
    
    # Encode image → latent
    scale = cfg["latent"]["scale_factor"]
    latent = encode(vae, img_lb, scale_factor=scale)  # (1, 4, 64, 64) fp32
    latent = latent.squeeze(0).cpu().half()           # (4, 64, 64) fp16
    
    # Downsample mask → latent-space soft mask
    downsample = cfg["latent"]["downsample"]
    mask_tensor = torch.from_numpy(mask_lb).float().unsqueeze(0)  # (1, C, H, W)
    soft_mask = F.avg_pool2d(mask_tensor, kernel_size=downsample, stride=downsample)
    soft_mask = soft_mask.squeeze(0).half()  # (C, 64, 64)
    
    return {
        "latent": latent,
        "mask": soft_mask,
        "filepath": sample.filepath,
        "orig_size": (orig_h, orig_w),
    }


def process_split(fo_dataset_name, out_dir, vae, cfg, class_to_idx):
    os.makedirs(out_dir, exist_ok=True)
    ds = fo.load_dataset(fo_dataset_name)
    n_saved, n_skipped = 0, 0
    first_errors = []
    
    for i, sample in enumerate(tqdm(ds, desc=f"caching {fo_dataset_name}")):
        try:
            result = process_sample(sample, vae, cfg, class_to_idx)
            torch.save(result, os.path.join(out_dir, f"{i:07d}.pt"))
            n_saved += 1
        except Exception as e:
            n_skipped += 1
            if len(first_errors) < 5:
                first_errors.append((sample.filepath, str(e)))
    
    print(f"\n[{fo_dataset_name}] saved={n_saved} skipped={n_skipped}")
    if first_errors:
        print("First few errors:")
        for fp, err in first_errors:
            print(f"  {fp}: {err}")