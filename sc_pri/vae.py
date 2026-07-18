"""SD-VAE encoder/decoder wrapper."""
import torch
from diffusers import AutoencoderKL


def load_vae(model_id="stabilityai/sd-vae-ft-mse", device="cuda"):
    vae = AutoencoderKL.from_pretrained(model_id).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae, img_tensor, scale_factor=0.18215):
    """Encode HWC uint8 numpy (or preprocessed tensor) to latent.
    
    Args:
        vae: loaded AutoencoderKL
        img_tensor: either (H, W, 3) uint8 numpy, or (B, 3, H, W) float tensor in [-1, 1]
    
    Returns:
        latent (B, 4, H/8, W/8) on VAE device, fp32
    """
    if hasattr(img_tensor, "dtype") and img_tensor.dtype == torch.uint8:
        # Assume HWC
        x = img_tensor.float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    elif img_tensor.ndim == 3:  # numpy HWC
        import numpy as np
        assert isinstance(img_tensor, np.ndarray)
        x = torch.from_numpy(img_tensor).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    else:
        x = img_tensor  # already (B, 3, H, W) float in [-1, 1]
    
    device = next(vae.parameters()).device
    x = x.to(device)
    z = vae.encode(x).latent_dist.mean * scale_factor
    return z


@torch.no_grad()
def decode(vae, latent, scale_factor=0.18215):
    """Decode latent to image tensor in [-1, 1]."""
    device = next(vae.parameters()).device
    latent = latent.to(device) / scale_factor
    img = vae.decode(latent).sample
    return img