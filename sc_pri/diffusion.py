"""SD 1.5 UNet + DDIM scheduler for latent-space denoising and repaint inpainting.

IMPORTANT scale convention (bug fix vs. previous version):
    The SD UNet operates in the SCALED latent domain, i.e. on
        z = vae.encode(x).latent_dist.mean * 0.18215
    which is exactly what we cache and what the Channel adds noise to.
    The previous version divided by 0.18215 before the UNet (and multiplied
    back after), feeding the UNet inputs ~5.5x too large. That is removed:
    latents pass through this module UNCHANGED in scale. Only vae.decode()
    (in sc_pri/vae.py) divides by the scale factor.

New in this version:
    - snr_to_timestep(): map channel noise variance -> diffusion timestep t*
      via alpha_bar(t*) = 1 / (1 + sigma_c^2).
    - ddim_denoise():   training-free channel denoising. Treat sqrt(abar)*y1
      as a legitimate diffusion state at t* and run reverse DDIM to 0.
    - estimate_sigma_c2_from_snr(): deployment-realistic noise-variance
      estimate using only the RECEIVED signal + known SNR (no clean z0).
"""

import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


def load_sd_components(model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                       device="cuda"):
    """Load UNet, scheduler, and text encoder from SD 1.5.

    Returns:
        dict with keys: unet, scheduler, text_encoder, tokenizer, device
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


# ---------------------------------------------------------------------------
# SNR <-> diffusion timestep mapping
# ---------------------------------------------------------------------------

def estimate_sigma_c2_from_snr(y1, snr_db):
    """Estimate channel noise variance from the RECEIVED latent only.

    Deployment-realistic: the relay has y1 and knows/estimates the SNR,
    but does NOT have the clean z0.

    Model: y1 = z0 + n, n ~ N(0, sigma_c^2), noise independent of signal.
        P_y1 ~= P_z0 + sigma_c^2,  and  sigma_c^2 = P_z0 / SNR_linear
        =>  sigma_c^2 = P_y1 / (1 + SNR_linear)

    Args:
        y1: (B, C, H, W) received latent (scaled domain, same as Channel)
        snr_db: float or (B,) tensor, SNR in dB

    Returns:
        sigma_c2: (B,) tensor of estimated noise variances
    """
    B = y1.shape[0]
    if isinstance(snr_db, (int, float)):
        snr_db = torch.full((B,), float(snr_db), device=y1.device)
    snr_linear = 10.0 ** (snr_db / 10.0)
    p_y = y1.reshape(B, -1).pow(2).mean(dim=1)  # (B,)
    return p_y / (1.0 + snr_linear)


def snr_to_timestep(sigma_c2, scheduler):
    """Map channel noise variance to the matching diffusion timestep t*.

    Derivation: channel gives y1 = z0 + sigma_c * n. Multiply by sqrt(abar_t):
        sqrt(abar_t) * y1 = sqrt(abar_t) * z0 + sqrt(abar_t) * sigma_c * n
    This matches the forward diffusion form
        x_t = sqrt(abar_t) * z0 + sqrt(1 - abar_t) * eps
    iff  abar_t * sigma_c^2 = 1 - abar_t,  i.e.
        abar_{t*} = 1 / (1 + sigma_c^2).

    Args:
        sigma_c2: float or scalar tensor, noise variance in the UNet's
                  (scaled) latent domain
        scheduler: DDIMScheduler with .alphas_cumprod

    Returns:
        (t_star, abar_star): int training-timestep index in [0, 999],
                             and the scheduler's abar at that index.
    """
    if torch.is_tensor(sigma_c2):
        sigma_c2 = sigma_c2.item()
    target_abar = 1.0 / (1.0 + sigma_c2)
    abars = scheduler.alphas_cumprod  # (num_train_timesteps,), decreasing in t
    t_star = int(torch.argmin((abars - target_abar).abs()).item())
    return t_star, abars[t_star].item()


# ---------------------------------------------------------------------------
# Training-free channel denoising
# ---------------------------------------------------------------------------

def snap_to_ddim_grid(sigma_c2, scheduler, num_inference_steps, device):
    """Map noise variance to the NEAREST timestep on the actual DDIM grid.

    Training/inference consistency: the same grid-aligned (t_g, abar_g) must
    be used both to construct training targets and to scale y1 at inference.
    The residual noise-level mismatch (at most half a grid spacing) is part
    of what the LoRA learns to absorb.

    Returns:
        (grid_idx, t_g, abar_g): index into scheduler.timesteps, the training
        timestep value at that index, and its abar.
    """
    if torch.is_tensor(sigma_c2):
        sigma_c2 = sigma_c2.item()
    scheduler.set_timesteps(num_inference_steps, device=device)
    grid = scheduler.timesteps.long()                       # descending
    grid_abars = scheduler.alphas_cumprod.to(device)[grid]  # (S,)
    target_abar = 1.0 / (1.0 + sigma_c2)
    grid_idx = int((grid_abars - target_abar).abs().argmin().item())
    return grid_idx, int(grid[grid_idx].item()), float(grid_abars[grid_idx])


def _set_adapters(unet, enabled):
    """Enable/disable PEFT adapters if any are attached; no-op otherwise."""
    try:
        if enabled:
            unet.enable_adapters()
        else:
            unet.disable_adapters()
    except (AttributeError, ValueError):
        pass


@torch.no_grad()
def ddim_denoise(y1, sigma_c2, sd_components,
                 num_inference_steps=50,
                 prompt="",
                 guidance_scale=1.0,
                 adapter_first_step_only=False,
                 verbose=False):
    """Remove channel noise by treating y1 as a diffusion state on the grid.

    Steps:
        1. Snap sigma_c2 to the nearest timestep t_g ON the DDIM grid
        2. x_{t_g} = sqrt(abar_{t_g}) * y1
        3. reverse DDIM from t_g down to 0
    No mask, no repaint -- pure restoration of the whole latent.

    Args:
        y1: (1, 4, 64, 64) noisy latent in the SCALED domain (as cached /
            as output by Channel). Passed to the UNet unchanged in scale.
        sigma_c2: channel noise variance in the same domain
        sd_components: dict from load_sd_components()
        num_inference_steps: DDIM discretization of the FULL schedule;
            only the steps at/below t_g are actually run.
        prompt: text prompt. Empty ("") + guidance_scale=1.0 recommended
            for pure denoising (single UNet pass per step, no CFG).
        guidance_scale: CFG scale. 1.0 disables CFG.
        adapter_first_step_only: if True and the UNet carries PEFT adapters
            (LoRA), the adapters are active only on the FIRST reverse step
            (the channel-entry correction) and disabled for the rest of the
            trajectory. No-op for a plain frozen UNet.

    Returns:
        (z_hat, t_g): denoised latent (1, 4, 64, 64) and the grid timestep.
    """
    unet = sd_components["unet"]
    scheduler = sd_components["scheduler"]
    tokenizer = sd_components["tokenizer"]
    text_encoder = sd_components["text_encoder"]
    device = sd_components["device"]

    y1 = y1.to(device)

    # 1-2. Grid-aligned entry point and rescale
    grid_idx, t_g, abar_g = snap_to_ddim_grid(
        sigma_c2, scheduler, num_inference_steps, device)
    x_t = (abar_g ** 0.5) * y1
    run_timesteps = scheduler.timesteps[grid_idx:]

    if verbose:
        print(f"  sigma_c2={float(sigma_c2):.5f}  abar_g={abar_g:.4f}  "
              f"t_g={t_g}  running {len(run_timesteps)} DDIM steps")

    use_cfg = guidance_scale > 1.0
    cond_emb = get_text_embedding(tokenizer, text_encoder, prompt, device)
    if use_cfg:
        uncond_emb = get_text_embedding(tokenizer, text_encoder, "", device)
        text_emb = torch.cat([uncond_emb, cond_emb])
    else:
        text_emb = cond_emb

    for i, t in enumerate(run_timesteps):
        if adapter_first_step_only and i == 1:
            _set_adapters(unet, False)

        if use_cfg:
            x_in = torch.cat([x_t, x_t])
            t_in = torch.cat([t.unsqueeze(0)] * 2)
            noise_pred = unet(x_in, t_in, encoder_hidden_states=text_emb).sample
            n_uncond, n_cond = noise_pred.chunk(2)
            noise_pred = n_uncond + guidance_scale * (n_cond - n_uncond)
        else:
            noise_pred = unet(x_t, t.unsqueeze(0),
                              encoder_hidden_states=text_emb).sample

        x_t = scheduler.step(noise_pred, t, x_t).prev_sample

    if adapter_first_step_only and len(run_timesteps) > 1:
        _set_adapters(unet, True)   # restore for the caller

    return x_t, t_g


# ---------------------------------------------------------------------------
# Repaint inpainting (scale bug fixed: latents used as-is)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_inpaint(z_0, mask_64, sd_components,
                 num_inference_steps=50,
                 strength=1.0,
                 prompt="",
                 guidance_scale=3.0,
                 seed=42):
    """DDIM repaint-style inpainting in latent space.

    Args:
        z_0: (1, 4, 64, 64) context latent in the SCALED domain. For best
            results this should be a CLEAN latent (e.g. output of
            ddim_denoise); repaint copies it into the mask-outside region,
            so any noise it carries is preserved verbatim.
        mask_64: (1, 1, 64, 64) float mask, 1 = region to replace, 0 = keep
        sd_components: dict from load_sd_components()
        num_inference_steps: DDIM steps
        strength: fraction of the diffusion process to run (1.0 = full)
        prompt: text prompt for the generated region
        guidance_scale: CFG scale (default lowered to 3.0; high CFG with
            weak prompts causes artifacts)
        seed: random seed

    Returns:
        z_inpainted: (1, 4, 64, 64) inpainted latent, same scaled domain
    """
    unet = sd_components["unet"]
    scheduler = sd_components["scheduler"]
    tokenizer = sd_components["tokenizer"]
    text_encoder = sd_components["text_encoder"]
    device = sd_components["device"]

    z_0 = z_0.to(device)
    mask_64 = mask_64.to(device)

    scheduler.set_timesteps(num_inference_steps, device=device)
    init_timestep = int(num_inference_steps * strength)
    t_start_idx = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start_idx:]

    cond_emb = get_text_embedding(tokenizer, text_encoder, prompt, device)
    uncond_emb = get_text_embedding(tokenizer, text_encoder, "", device)
    text_emb = torch.cat([uncond_emb, cond_emb])  # (2, 77, 768)

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z_0.shape, generator=generator,
                        device=device, dtype=z_0.dtype)

    # Noised context at the starting timestep
    z_t = scheduler.add_noise(z_0, noise, timesteps[:1])

    for i, t in enumerate(timesteps):
        z_t_input = torch.cat([z_t, z_t])
        t_input = torch.cat([t.unsqueeze(0)] * 2)

        noise_pred = unet(z_t_input, t_input, encoder_hidden_states=text_emb).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond)

        z_t = scheduler.step(noise_pred, t, z_t).prev_sample

        # Repaint: pin mask-outside to the context, noised to the next step
        if i < len(timesteps) - 1:
            next_t = timesteps[i + 1]
            z_ctx = scheduler.add_noise(z_0, noise, next_t.unsqueeze(0))
        else:
            z_ctx = z_0

        z_t = mask_64 * z_t + (1 - mask_64) * z_ctx

    return z_t