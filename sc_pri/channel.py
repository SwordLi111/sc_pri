"""AWGN channel for latent-space semantic communication.

Fixes vs. previous implementation:
- Uses ALL tensor dimensions (C, H, W) for signal power computation, not 3 of 4.
- No spurious /2 factor. That factor was mistakenly borrowed from complex-signal
  formulas; latents are real-valued, so per-sample noise variance = P_signal / SNR_linear.

Verified by test_channel_snr() below.
"""
import torch


class Channel:
    """Additive White Gaussian Noise (AWGN) channel for real-valued latents.
    
    Given a target SNR in dB, adds noise such that:
        SNR_dB = 10 * log10(P_signal / P_noise)
    
    where P_signal is measured per-sample over all latent dimensions.
    """
    
    def __init__(self, snr_db_range=(0.0, 20.0), fixed_snr_db=None):
        """
        Args:
            snr_db_range: (low, high) for random SNR sampling per batch
            fixed_snr_db: if set, ignores range and uses this fixed SNR
        """
        self.snr_db_range = snr_db_range
        self.fixed_snr_db = fixed_snr_db
    
    def sample_snr(self, batch_size, device):
        """Sample SNR (in dB) per sample in the batch."""
        if self.fixed_snr_db is not None:
            return torch.full((batch_size,), self.fixed_snr_db, 
                              dtype=torch.float32, device=device)
        low, high = self.snr_db_range
        return torch.rand(batch_size, device=device) * (high - low) + low
    
    def forward(self, z, snr_db=None):
        """Add AWGN to latent z.
        
        Args:
            z: (B, C, H, W) real-valued latent
            snr_db: (B,) or scalar SNR in dB. If None, sample from configured range.
        
        Returns:
            (noisy_z, snr_db_used)
        """
        B = z.shape[0]
        if snr_db is None:
            snr_db = self.sample_snr(B, z.device)
        elif isinstance(snr_db, (int, float)):
            snr_db = torch.full((B,), float(snr_db), device=z.device)
        
        # Per-sample signal power over ALL dims (C, H, W)
        # z_flat: (B, C*H*W)
        z_flat = z.reshape(B, -1)
        signal_power = z_flat.pow(2).mean(dim=1)  # (B,)
        
        # noise_power = signal_power / SNR_linear
        snr_linear = 10.0 ** (snr_db / 10.0)         # (B,)
        noise_power = signal_power / snr_linear      # (B,)
        noise_std = noise_power.sqrt()               # (B,)
        
        # Broadcast std to z shape
        noise = torch.randn_like(z) * noise_std.view(B, 1, 1, 1)
        return z + noise, snr_db
    
    def __call__(self, z, snr_db=None):
        return self.forward(z, snr_db)

