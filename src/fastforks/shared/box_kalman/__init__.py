"""Constant-velocity Kalman filter of boxes, batched over tracks (multi-object tracking, SAMURAI).

The state of a box is `(cx, cy, a, h, vcx, vcy, va, vh)`: centre, aspect `w / h`, height and their velocities per
frame; the measurement is `(cx, cy, a, h)`. The noise model is ByteTrack's (and SAMURAI's): position noise
`h / 20`, velocity noise `h / 160` per frame, fixed small aspect noise. Every function takes `[N, ...]` stacks and
works on all tracks at once, in float64 numpy (the reference trackers are float64 numpy too).
"""

from __future__ import annotations

import numpy as np

__all__ = ["initiate", "predict", "to_xyah", "to_xyxy", "update"]

POSITION = 1.0 / 20  # fixed: reference position noise per unit of box height
VELOCITY = 1.0 / 160  # fixed: reference velocity noise per unit of box height
_F = np.eye(8)
_F[range(4), range(4, 8)] = 1.0  # constant velocity, one frame per step


def to_xyah(xyxy: np.ndarray) -> np.ndarray:
    """`[N, 4]` boxes `(x0, y0, x1, y1)` -> measurements `(cx, cy, w / h, h)`."""
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    return np.stack((xyxy[:, 0] + w / 2, xyxy[:, 1] + h / 2, w / h, h), 1)


def to_xyxy(state: np.ndarray) -> np.ndarray:
    """`[N, >=4]` states or measurements `(cx, cy, a, h, ...)` -> boxes `(x0, y0, x1, y1)`."""
    w = state[:, 2] * state[:, 3]
    x0 = state[:, 0] - w / 2
    y0 = state[:, 1] - state[:, 3] / 2
    return np.stack((x0, y0, x0 + w, y0 + state[:, 3]), 1)


def _diag(std: np.ndarray) -> np.ndarray:
    """`[N, k]` standard deviations -> `[N, k, k]` diagonal covariances."""
    out = np.zeros((*std.shape, std.shape[1]))
    idx = np.arange(std.shape[1])
    out[:, idx, idx] = std * std
    return out


def _std(h: np.ndarray, pos: float, aspect: float) -> np.ndarray:
    """`[N, 4]` standard deviations `(pos h, pos h, aspect, pos h)` for heights `h`."""
    return np.stack((pos * h, pos * h, np.full_like(h, aspect), pos * h), 1)


def initiate(xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """New tracks from `[N, 4]` measurements: `(mean [N, 8], covariance [N, 8, 8])`, zero velocity."""
    mean = np.concatenate((xyah, np.zeros_like(xyah)), 1)
    h = xyah[:, 3]
    return mean, _diag(np.concatenate((_std(h, 2 * POSITION, 1e-2), _std(h, 10 * VELOCITY, 1e-5)), 1))


def predict(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One frame ahead: `x = F x`, `P = F P F^T + Q` with the noise of the current heights."""
    h = mean[:, 3]
    q = _diag(np.concatenate((_std(h, POSITION, 1e-2), _std(h, VELOCITY, 1e-5)), 1))
    return mean @ _F.T, _F @ cov @ _F.T + q


def update(mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correct the states with `[N, 4]` measurements (one per track)."""
    r = _diag(_std(mean[:, 3], POSITION, 1e-1))
    s = cov[:, :4, :4] + r  # innovation covariance H P H^T + R (H picks the first four)
    pht = cov[:, :, :4]  # P H^T
    gain = np.linalg.solve(s, pht.transpose(0, 2, 1)).transpose(0, 2, 1)  # P H^T S^-1 (S symmetric)
    mean = mean + (gain @ (xyah - mean[:, :4])[:, :, None])[:, :, 0]
    return mean, cov - gain @ s @ gain.transpose(0, 2, 1)
