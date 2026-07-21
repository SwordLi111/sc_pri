"""MaskDecoder: small U-Net for latent-space mask prediction.

Input:  (B, 4, 64, 64) noisy latent
Output: (B, C, 64, 64) logits for C classes (face, plate, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MaskDecoder(nn.Module):
    """3-level U-Net for latent-space segmentation.
    
    Args:
        in_channels: input channels (4 for SD-VAE latent)
        out_channels: number of classes (2 for face + plate)
        base: base feature width
    """

    def __init__(self, in_channels=4, out_channels=2, base=64):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base)       # 64x64
        self.enc2 = DoubleConv(base, base * 2)           # 32x32
        self.enc3 = DoubleConv(base * 2, base * 4)       # 16x16

        # Bottleneck
        self.bottleneck = DoubleConv(base * 4, base * 4)  # 8x8

        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)       # 16x16

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)       # 32x32

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)           # 64x64

        self.head = nn.Conv2d(base, out_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # (B, 64, 64, 64)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 32, 32)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 16, 16)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 256, 8, 8)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # (B, 256, 16, 16)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 128, 32, 32)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 64, 64, 64)

        return self.head(d1)  # (B, C, 64, 64) logits