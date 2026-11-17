"""Constant-velocity Kalman filter of boxes, batched over tracks (multi-object tracking, SAMURAI).

The state of a box is `(cx, cy, a, h, vcx, vcy, va, vh)`: centre, aspect `w / h`, height and their velocities per
frame; the measurement is `(cx, cy, a, h)`. The noise model is ByteTrack's (and SAMURAI's): position noise
`h / 20`, velocity noise `h / 160` per frame, fixed small aspect noise. Every function takes `[N, ...]` stacks and
works on all tracks at once, in float64 (the reference trackers are float64 numpy): numpy arrays, or torch tensors
on their device (SAMURAI: no host round trip per frame; no error check, a singular innovation gives non-finite rows).
"""

from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import torch

A = TypeVar("A", np.ndarray, torch.Tensor)  # numpy arrays or torch tensors (one kind per call)
__all__ = ["initiate", "predict", "to_xyah", "to_xyxy", "update"]

POSITION = 1.0 / 20  # fixed: reference position noise per unit of box height
VELOCITY = 1.0 / 160  # fixed: reference velocity noise per unit of box height
_F = np.eye(8)
_F[range(4), range(4, 8)] = 1.0  # constant velocity, one frame per step


def _xp(x: A) -> Any:  # noqa: ANN401 - numpy or torch, the array module of `x`
    return torch if isinstance(x, torch.Tensor) else np


_CONST = {"F": _F, "eye4": np.eye(4), "eye8": np.eye(8)}
_on_device: dict[tuple[str, torch.dtype, torch.device], torch.Tensor] = {}


def _like(x: A, name: str) -> Any:  # noqa: ANN401 - an array of `x`'s kind
    """The constant `name` as an array of `x`'s kind (dtype, device); copied to a device once (no sync per call)."""
    a = _CONST[name]
    if not isinstance(x, torch.Tensor):
        return a
    key = (name, x.dtype, x.device)
    if (c := _on_device.get(key)) is None:
        c = _on_device[key] = torch.as_tensor(a, dtype=x.dtype, device=x.device)
    return c


def to_xyah(xyxy: A) -> A:
    """`[N, 4]` boxes `(x0, y0, x1, y1)` -> measurements `(cx, cy, w / h, h)`."""
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    return _xp(xyxy).stack((xyxy[:, 0] + w / 2, xyxy[:, 1] + h / 2, w / h, h), 1)


def to_xyxy(state: A) -> A:
    """`[N, >=4]` states or measurements `(cx, cy, a, h, ...)` -> boxes `(x0, y0, x1, y1)`."""
    w = state[:, 2] * state[:, 3]
    x0 = state[:, 0] - w / 2
    y0 = state[:, 1] - state[:, 3] / 2
    return _xp(state).stack((x0, y0, x0 + w, y0 + state[:, 3]), 1)


def _diag(std: A) -> A:
    """`[N, k]` standard deviations -> `[N, k, k]` diagonal covariances."""
    return (std * std)[:, :, None] * _like(std, f"eye{std.shape[1]}")


def _std(h: A, pos: float, aspect: float) -> A:
    """`[N, 4]` standard deviations `(pos h, pos h, aspect, pos h)` for heights `h`."""
    xp = _xp(h)
    return xp.stack((pos * h, pos * h, xp.full_like(h, aspect), pos * h), 1)


def initiate(xyah: A) -> tuple[A, A]:
    """New tracks from `[N, 4]` measurements: `(mean [N, 8], covariance [N, 8, 8])`, zero velocity."""
    xp = _xp(xyah)
    mean = xp.concatenate((xyah, xp.zeros_like(xyah)), 1)
    h = xyah[:, 3]
    return mean, _diag(xp.concatenate((_std(h, 2 * POSITION, 1e-2), _std(h, 10 * VELOCITY, 1e-5)), 1))


def predict(mean: A, cov: A) -> tuple[A, A]:
    """One frame ahead: `x = F x`, `P = F P F^T + Q` with the noise of the current heights."""
    h = mean[:, 3]
    q = _diag(_xp(h).concatenate((_std(h, POSITION, 1e-2), _std(h, VELOCITY, 1e-5)), 1))
    f = _like(mean, "F")
    return mean @ f.T, f @ cov @ f.T + q


def update(mean: A, cov: A, xyah: A) -> tuple[A, A]:
    """Correct the states with `[N, 4]` measurements (one per track)."""
    r = _diag(_std(mean[:, 3], POSITION, 1e-1))
    s = cov[:, :4, :4] + r  # innovation covariance H P H^T + R (H picks the first four)
    pht = cov[:, :, :4]  # P H^T
    ht = pht.swapaxes(1, 2)
    sol: Any = torch.linalg.solve_ex(s, ht)[0] if isinstance(s, torch.Tensor) else np.linalg.solve(s, ht)
    gain = sol.swapaxes(1, 2)  # P H^T S^-1 (S symmetric)
    mean = mean + (gain @ (xyah - mean[:, :4])[:, :, None])[:, :, 0]
    return mean, cov - gain @ s @ gain.swapaxes(1, 2)
