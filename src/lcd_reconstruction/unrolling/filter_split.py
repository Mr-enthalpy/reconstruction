"""Explicit per-band convolution and spectral-filter split for D5."""
import torch
from torch import nn
import torch.nn.functional as F


class FilterSplitPhysics(nn.Module):
    """J_b = H_b X_b; y = sum_b w_b J_b, with an explicit adjoint."""
    def __init__(self, kernels, hw, weights=None):
        super().__init__()
        k = torch.as_tensor(kernels)
        if k.ndim != 3 or k.shape[0] != 21 or not torch.isfinite(k).all():
            raise ValueError("finite [21,kh,kw] kernels required")
        self.hw = tuple(hw)
        self.origin = (k.shape[-2] // 2, k.shape[-1] // 2)
        self.fft_hw = tuple(1 << (a + b - 2).bit_length() for a, b in zip(hw, k.shape[-2:]))
        self.register_buffer("kernel_fft", torch.fft.rfft2(k, s=self.fft_hw))
        if weights is None:
            weights = torch.ones(21, 1, 1)
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.ndim == 1:
            weights = weights[:, None, None]
        if weights.shape[0] != 21 or (weights < 0).any() or not torch.isfinite(weights).all():
            raise ValueError("nonnegative finite 21-band weights required")
        self.register_buffer("weights", weights)

    def bands(self, x):
        if tuple(x.shape[-2:]) != self.hw or x.shape[-3] != 21:
            raise ValueError("input shape differs from operator contract")
        full = torch.fft.irfft2(
            torch.fft.rfft2(x, s=self.fft_hw) * self.kernel_fft, s=self.fft_hw)
        oy, ox = self.origin
        h, w = self.hw
        return full[..., oy:oy + h, ox:ox + w]

    def forward(self, x):
        weighted = self.bands(x) * self.weights
        return weighted.sum(-3, keepdim=True)

    def adjoint(self, y):
        if tuple(y.shape[-2:]) != self.hw or y.shape[1] != 1:
            raise ValueError("measurement shape differs from operator contract")
        oy, ox = self.origin
        h, w = self.hw
        fh, fw = self.fft_hw
        weighted = y * self.weights
        padded = F.pad(weighted, (ox, fw - w - ox, oy, fh - h - oy))
        full = torch.fft.irfft2(
            torch.fft.rfft2(padded, s=self.fft_hw) * self.kernel_fft.conj(),
            s=self.fft_hw)
        return full[..., :h, :w]

    def lipschitz_bound(self):
        band = self.kernel_fft.abs().square()
        weight = self.weights.square()
        if weight.ndim == 3:
            weight = weight.amax((-2, -1))
        return (band * weight[:, None, None]).sum(0).max().real

