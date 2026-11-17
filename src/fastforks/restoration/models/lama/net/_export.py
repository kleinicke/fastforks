"""The fixed-size orthonormal Fourier transforms used for LaMa ONNX export."""

from __future__ import annotations

import math

import torch


def _dft_basis(n: int, bins: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal cosine and sine bases with spatial rows and frequency columns."""
    angle = torch.arange(n, device=device, dtype=torch.float64)[:, None] * (2 * math.pi / n)
    angle = angle * torch.arange(bins, device=device, dtype=torch.float64)
    return (angle.cos() / math.sqrt(n)).float(), (angle.sin() / math.sqrt(n)).float()


def real_dft2(x: torch.Tensor) -> torch.Tensor:
    """Real-input 2-D DFT as real matmuls for TensorRT, with real/imaginary last."""
    h, w = x.shape[-2:]
    cos_w, sin_w = _dft_basis(w, w // 2 + 1, x.device)
    cos_h, sin_h = _dft_basis(h, h, x.device)
    re, im = x @ cos_w, -(x @ sin_w)
    return torch.stack((cos_h.T @ re + sin_h.T @ im, cos_h.T @ im - sin_h.T @ re), dim=-1)


def inverse_real_dft2(spectrum: torch.Tensor, width: int) -> torch.Tensor:
    """Inverse real 2-D DFT. Interior horizontal bins represent both signs of the frequency."""
    h, bins = spectrum.shape[-3:-1]
    cos_h, sin_h = _dft_basis(h, h, spectrum.device)
    cos_w, sin_w = _dft_basis(width, bins, spectrum.device)
    k = torch.arange(bins, device=spectrum.device)
    singleton = (k == 0) | (2 * k == width)
    weight = 2 - singleton.to(spectrum.dtype)
    re, im = spectrum.unbind(-1)
    x_re = cos_h @ re - sin_h @ im
    x_im = sin_h @ re + cos_h @ im
    return x_re @ (cos_w * weight).T - x_im @ (sin_w * weight).T
