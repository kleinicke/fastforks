# fastforks: `diffusers.LCMScheduler` (huggingface/diffusers, Apache-2.0, see ../LICENSE-huggingface) re-implemented for
# the configuration Marigold normals LCM ships (scheduler/scheduler_config.json): scaled-linear betas 0.00085 -> 0.012
# over 1000 steps (no zero-SNR rescale), v-prediction, original_inference_steps 50, timestep_scaling 10,
# set_alpha_to_one false, no clipping / thresholding. Same float32 arithmetic and operation order as diffusers.
from __future__ import annotations

import numpy as np
import torch

TRAIN_STEPS = 1000
ORIGINAL_STEPS = 50  # the distillation schedule: timesteps 999, 979, ..., 19
TIMESTEP_SCALING = 10.0
SIGMA_DATA = 0.5


class LCM:
    """Timesteps and the LCM update for `steps` denoising steps (at most `ORIGINAL_STEPS`)."""

    def __init__(self, steps: int) -> None:
        betas = torch.linspace(0.00085**0.5, 0.012**0.5, TRAIN_STEPS, dtype=torch.float32) ** 2
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = self.alphas_cumprod[0]
        k = TRAIN_STEPS // ORIGINAL_STEPS
        origin = (np.asarray(list(range(1, ORIGINAL_STEPS + 1))) * k - 1)[::-1].copy()
        index = np.floor(np.linspace(0, len(origin), num=steps, endpoint=False)).astype(np.int64)
        self.timesteps = [int(t) for t in origin[index]]
        nxt = [*self.timesteps[1:], None]
        self._coef = {t: self._coefficients(t, p) for t, p in zip(self.timesteps, nxt, strict=True)}

    def _coefficients(self, t: int, prev: int | None) -> tuple[torch.Tensor, ...]:
        """`sqrt(a_t)`, `sqrt(1 - a_t)`, `c_out`, `c_skip`, `sqrt(a_prev)`, `sqrt(1 - a_prev)` (0-d float32 CPU)."""
        a_t = self.alphas_cumprod[t]
        p = t if prev is None else prev
        a_prev = self.alphas_cumprod[p] if p >= 0 else self.final_alpha_cumprod
        scaled = torch.tensor(t) * TIMESTEP_SCALING  # diffusers: the int64 timestep tensor x 10 (float32)
        c_skip = SIGMA_DATA**2 / (scaled**2 + SIGMA_DATA**2)
        c_out = scaled / (scaled**2 + SIGMA_DATA**2) ** 0.5
        return a_t.sqrt(), (1 - a_t).sqrt(), c_out, c_skip, a_prev.sqrt(), (1 - a_prev).sqrt()

    def step(self, v: torch.Tensor, t: int, sample: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
        """x_t -> the next latent from the v-prediction `v` (diffusers `LCMScheduler.step`).

        `noise` (fresh Gaussian noise) is added for every step but the last one, where it must be None.
        """
        sa, sb, c_out, c_skip, sp, sq = self._coef[t]
        denoised = c_out * (sa * sample - sb * v) + c_skip * sample
        return denoised if noise is None else sp * denoised + sq * noise
