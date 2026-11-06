"""Checks of the camera matrices models take: pinhole intrinsics `[..., 3, 3]`, camera-to-world poses `[..., 4, 4]`.

`valid` builds one 0-d bool on the matrices' own device (numpy / list inputs: the CPU, so no GPU work); the caller
reads it with the one host sync it already had for the finiteness check. `problem` explains a failure (slow path).
"""

from __future__ import annotations

import torch

ROTATION_TOLERANCE = 1e-4
"""Largest `|R R^T - I|` of a pose rotation, as `StereoRig` (files round rotations to ~1e-6)."""
_ROW_TOLERANCE = 1e-6  # the constant last rows ([0, 0, 1], [0, 0, 0, 1]), float32 rounding


def _row_ok(row: torch.Tensor) -> torch.Tensor:
    """The rows `[..., n]` are `[0, ..., 0, 1]`."""
    return (row[..., :-1].abs() <= _ROW_TOLERANCE).all() & ((row[..., -1] - 1).abs() <= _ROW_TOLERANCE).all()


def _rotation_ok(t: torch.Tensor) -> torch.Tensor:
    r = t[..., :3, :3]
    eye = torch.eye(3, dtype=t.dtype, device=t.device)
    return ((r @ r.mT - eye).abs() <= ROTATION_TOLERANCE).all() & (torch.linalg.det(r) > 0).all()


def valid(t: torch.Tensor, name: str) -> torch.Tensor:
    """0-d bool: `t` is finite and holds valid camera matrices.

    Intrinsics (`name == "intrinsics"`): fx, fy > 0, last row `[0, 0, 1]`. Otherwise rigid camera-to-world poses:
    last row `[0, 0, 0, 1]`, orthonormal rotation with det +1.
    """
    ok = torch.isfinite(t).all()
    if name == "intrinsics":
        return ok & (t[..., 0, 0] > 0).all() & (t[..., 1, 1] > 0).all() & _row_ok(t[..., 2, :])
    return ok & _row_ok(t[..., 3, :]) & _rotation_ok(torch.nan_to_num(t))


def problem(t: torch.Tensor, name: str) -> str:
    """Why `valid(t, name)` is False, for the error message."""
    if not bool(torch.isfinite(t).all()):
        return "non-finite values"
    if name == "intrinsics":
        if not bool((t[..., 0, 0] > 0).all() & (t[..., 1, 1] > 0).all()):
            return "the focal lengths fx = K[0, 0] and fy = K[1, 1] must be > 0"
        return "the last row must be [0, 0, 1] (OpenCV pixel matrices)"
    if not bool(_row_ok(t[..., 3, :])):
        return "the last row must be [0, 0, 0, 1]"
    return (
        f"the rotation [:3, :3] must be orthonormal (to {ROTATION_TOLERANCE:g}) with det +1: rigid camera-to-world"
        " poses, OpenCV axes"
    )
