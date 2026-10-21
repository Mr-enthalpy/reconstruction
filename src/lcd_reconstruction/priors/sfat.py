"""Spatial-frequency aggregation prior used by the D5 HSFAU-style stage."""
import torch
from torch import nn
import torch.nn.functional as F


class SpatialFrequencyBlock(nn.Module):
    def __init__(self, channels=21, width=32):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, channels, 3, padding=1))
        self.frequency = nn.Sequential(
            nn.Conv2d(channels, width, 1), nn.GELU(),
            nn.Conv2d(width, channels, 1))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x):
        spatial = self.spatial(x)
        spectrum = torch.fft.rfft2(x, norm="ortho")
        magnitude = spectrum.abs()
        gain = torch.sigmoid(self.frequency(magnitude))
        frequency = torch.fft.irfft2(spectrum * gain, s=x.shape[-2:], norm="ortho")
        mixed = self.fuse(torch.cat([spatial, frequency], dim=1))
        return x + mixed * self.attention(x)


class SFAT(nn.Module):
    def __init__(self, channels=21, width=32, levels=3):
        super().__init__()
        if levels < 2:
            raise ValueError("at least two levels required")
        self.levels = levels
        self.stem = nn.Conv2d(channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [SpatialFrequencyBlock(channels, width) for _ in range(levels)])
        self.ups = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(levels - 1)])
        self.out = nn.Conv2d(channels, channels, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        features = [self.blocks[0](self.stem(x))]
        for level in range(1, self.levels):
            down = F.avg_pool2d(features[-1], 2, ceil_mode=True)
            features.append(self.blocks[level](down))
        current = features[-1]
        for level in range(self.levels - 2, -1, -1):
            current = F.interpolate(current, size=features[level].shape[-2:],
                                    mode="bilinear", align_corners=False)
            current = self.ups[level](current + features[level])
        return self.out(current)
