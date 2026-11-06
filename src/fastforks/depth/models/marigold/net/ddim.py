# fastforks: `diffusers.DDIMScheduler` (huggingface/diffusers, Apache-2.0, see LICENSE-huggingface) re-implemented for
# the configuration Marigold ships (scheduler/scheduler_config.json): scaled-linear betas 0.00085 -> 0.012 over 1000
# steps rescaled to zero terminal SNR, "trailing" timestep spacing, v-prediction, set_alpha_to_one false, eta 0 (no
# noise is added in a step), no clipping / thresholding. Same float32 arithmetic and operation order as diffusers.
from __future__ import annotations

import numpy as np
import torch

TRAIN_STEPS = 1000


def _betas() -> torch.Tensor:
    """`betas_for` "scaled_linear" + `rescale_zero_terminal_snr`."""
    betas = torch.linspace(0.00085**0.5, 0.012**0.5, TRAIN_STEPS, dtype=torch.float32) ** 2
    alphas_bar_sqrt = torch.cumprod(1.0 - betas, dim=0).sqrt()
    s0, st = alphas_bar_sqrt[0].clone(), alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= st
    alphas_bar_sqrt *= s0 / (s0 - st)
    alphas_bar = alphas_bar_sqrt**2
    alphas = torch.cat([alphas_bar[0:1], alphas_bar[1:] / alphas_bar[:-1]])
    return 1 - alphas


class DDIM:
    """Timesteps and the deterministic DDIM update for `steps` denoising steps."""

    def __init__(self, steps: int) -> None:
        self.alphas_cumprod = torch.cumprod(1.0 - _betas(), dim=0)
        self.final_alpha_cumprod = self.alphas_cumprod[0]
        self.steps = steps
        ratio = TRAIN_STEPS / steps
        self.timesteps = (np.round(np.arange(TRAIN_STEPS, 0, -ratio)).astype(np.int64) - 1).tolist()
        self._coef = {t: self._coefficients(t) for t in self.timesteps}

    def _coefficients(self, t: int) -> tuple[torch.Tensor, ...]:
        """`sqrt(a_t)`, `sqrt(1 - a_t)`, `sqrt(a_prev)`, `sqrt(1 - a_prev)` of step `t` (0-d float32 CPU tensors)."""
        prev = t - TRAIN_STEPS // self.steps
        a_t = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod[prev] if prev >= 0 else self.final_alpha_cumprod
        b_t = 1 - a_t
        coef = (a_t**0.5, b_t**0.5, a_prev**0.5, (1 - a_prev - 0.0**2) ** 0.5)  # std_dev_t = eta * variance ** 0.5 = 0
        return coef

    def step(self, v: torch.Tensor, t: int, sample: torch.Tensor) -> torch.Tensor:
        """x_t -> x_{t - 1000 // steps} from the v-prediction `v` (diffusers `DDIMScheduler.step`, eta = 0).

        The coefficients are 0-d CPU tensors computed once per timestep, as diffusers computes them per step (same
        results): Inductor fused the select from the CPU `alphas_cumprod` into a CUDA kernel (invalid pointer).
        """
        sa, sb, sp, d = self._coef.get(t) or self._coefficients(t)
        x0 = sa * sample - sb * v
        eps = sa * v + sb * sample
        return sp * x0 + d * eps
