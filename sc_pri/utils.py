"""Dependency-free utilities. No imports from other sc_pri modules."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def letterbox_image(img_np, new_size=512, pad_value=114):
    """Resize keeping aspect ratio, pad to square.
    
    Args:
        img_np: HxWx3 uint8 numpy array
        new_size: target square size
        pad_value: pixel value for padding
    
    Returns:
        (padded_img, scale, pad_x, pad_y)
    """
    h, w = img_np.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = np.array(Image.fromarray(img_np).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((new_size, new_size, 3), pad_value, dtype=np.uint8)
    pad_y = (new_size - nh) // 2
    pad_x = (new_size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def letterbox_binary_mask(mask_np, new_size=512):
    """Letterbox a HxW binary mask using nearest interpolation with zero padding."""
    h, w = mask_np.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = np.array(
        Image.fromarray((mask_np.astype(np.uint8) * 255))
             .resize((nw, nh), Image.NEAREST)
    ) > 127
    canvas = np.zeros((new_size, new_size), dtype=np.uint8)
    pad_y = (new_size - nh) // 2
    pad_x = (new_size - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized.astype(np.uint8)
    return canvas


def transform_bbox_letterbox(x, y, w, h, orig_w, orig_h, new_size=512):
    """Convert relative bbox [0,1] to pixel coords in letterboxed image.
    
    Args:
        x, y, w, h: relative bbox (fiftyone / OI format)
        orig_w, orig_h: original image size
        new_size: letterboxed size
    
    Returns:
        (x1, y1, x2, y2) as clipped ints
    """
    scale = new_size / max(orig_w, orig_h)
    pad_x = (new_size - int(round(orig_w * scale))) // 2
    pad_y = (new_size - int(round(orig_h * scale))) // 2
    x1 = int(round(x * orig_w * scale + pad_x))
    y1 = int(round(y * orig_h * scale + pad_y))
    x2 = int(round((x + w) * orig_w * scale + pad_x))
    y2 = int(round((y + h) * orig_h * scale + pad_y))
    return (max(0, x1), max(0, y1), min(new_size, x2), min(new_size, y2))


# ---------- Loss / metrics ----------

def dice_loss(pred, target, eps=1e-6):
    """Dice loss for binary segmentation. Both pred/target are (B, C, H, W) in [0,1]."""
    pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
    target = target.contiguous().view(target.shape[0], target.shape[1], -1)
    num = 2 * (pred * target).sum(-1) + eps
    den = pred.sum(-1) + target.sum(-1) + eps
    return 1 - (num / den).mean()


def bce_dice_loss(pred_logits, target, dice_weight=1.0):
    """BCE + Dice loss combo for multi-channel mask prediction."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    dice = dice_loss(torch.sigmoid(pred_logits), target)
    return bce + dice_weight * dice


def iou(pred_binary, target_binary, eps=1e-6):
    """Per-class IoU. Inputs are (B, C, H, W) binary. Returns (C,) tensor."""
    dims = (0, 2, 3)
    inter = (pred_binary * target_binary).sum(dims)
    union = pred_binary.sum(dims) + target_binary.sum(dims) - inter
    return (inter + eps) / (union + eps)


# ---------- Config loading ----------

def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)