"""SD 1.5 UNet + DDIM scheduler for latent-space repaint inpainting.

Pipeline:
    1. Forward diffusion: add noise to z_0 up to timestep T
    2. Reverse DDIM: denoise step by step, each step replacing
       mask-outside region with the original noised latent at that timestep.
    Result: mask-inside = generated content, mask-outside = original image.
"""

import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


def load_sd_components(model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                       device="cuda"):
    """Load UNet, scheduler, and text encoder from SD 1.5.
    
    Returns:
        dict with keys: unet, scheduler, text_encoder, tokenizer
    """
    print(f"Loading UNet from {model_id}...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet"
    ).to(device).eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    
    print("Loading DDIM scheduler...")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    print("Loading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder"
    ).to(device).eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    
    return {
        "unet": unet,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "device": device,
    }


def get_text_embedding(tokenizer, text_encoder, prompt, device):
    """Encode a text prompt to CLIP embedding."""
    tokens = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        emb = text_encoder(tokens).last_hidden_state  # (1, 77, 768)
    return emb


@torch.no_grad()
def ddim_inpaint(z_0, mask_64, sd_components,
                 num_inference_steps=50,
                 strength=1.0,
                 prompt="",
                 guidance_scale=7.5,
                 seed=42):
    """DDIM repaint-style inpainting in latent space.
    
    Args:
        z_0: (1, 4, 64, 64) original clean latent (already scaled by 0.18215)
        mask_64: (1, 1, 64, 64) float mask, 1 = region to replace, 0 = keep
        sd_components: dict from load_sd_components()
        num_inference_steps: DDIM steps
        strength: how much of the diffusion process to run (1.0 = full)
        prompt: text prompt for generation (empty = unconditional-ish)
        guidance_scale: classifier-free guidance scale
        seed: random seed
    
    Returns:
        z_inpainted: (1, 4, 64, 64) inpainted latent (scaled)
    """
    unet = sd_components["unet"]
    scheduler = sd_components["scheduler"]
    tokenizer = sd_components["tokenizer"]
    text_encoder = sd_components["text_encoder"]
    device = sd_components["device"]
    
    # Unscale for diffusion (SD UNet expects unscaled latents)
    scale_factor = 0.18215
    z_0_unscaled = z_0 / scale_factor
    
    # Set up scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Determine start timestep based on strength
    init_timestep = int(num_inference_steps * strength)
    t_start_idx = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start_idx:]
    
    # Get text embeddings (conditional + unconditional for CFG)
    cond_emb = get_text_embedding(tokenizer, text_encoder, prompt, device)
    uncond_emb = get_text_embedding(tokenizer, text_encoder, "", device)
    text_emb = torch.cat([uncond_emb, cond_emb])  # (2, 77, 768)
    
    # Forward diffusion: add noise to z_0 up to first timestep
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z_0_unscaled.shape, generator=generator,
                        device=device, dtype=z_0_unscaled.dtype)
    
    # Noised original at the starting timestep
    z_t = scheduler.add_noise(z_0_unscaled, noise, timesteps[:1])
    
    # Reverse DDIM loop
    for i, t in enumerate(timesteps):
        # Classifier-free guidance: predict noise for uncond and cond
        z_t_input = torch.cat([z_t, z_t])  # (2, 4, 64, 64)
        t_input = torch.cat([t.unsqueeze(0)] * 2)
        
        noise_pred = unet(z_t_input, t_input, encoder_hidden_states=text_emb).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # DDIM step
        z_t = scheduler.step(noise_pred, t, z_t).prev_sample
        
        # Repaint: replace mask-outside with original noised latent at this timestep
        if i < len(timesteps) - 1:
            next_t = timesteps[i + 1]
            z_orig_noised = scheduler.add_noise(z_0_unscaled, noise, next_t.unsqueeze(0))
        else:
            # Last step: use clean original
            z_orig_noised = z_0_unscaled
        
        z_t = mask_64 * z_t + (1 - mask_64) * z_orig_noised
    
    # Re-scale for VAE decoding
    z_inpainted = z_t * scale_factor
    return z_inpainted