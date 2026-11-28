"""Metal kernels of the pose RANSAC on Apple MLX, run through `mx.fast.metal_kernel`.

`mode="mlx"` of `matching.estimate_pose` / `estimate_fundamental` / `estimate_absolute_pose`: a port of
`fastforks.kernels.ransac` (CUDA).

`ops` holds the kernels and needs the `mlx` extra; this module does not import it (`load()` only tags the sources).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Library:
    """An MLX kernel library: its sources compile inside MLX on first use.

    Attributes:
        tag: `name@<first 12 hex of the SHA-256 of the sources>:mlx`, as reported in `Backend.kernels`.
    """

    tag: str


def load() -> Library:
    """The library's tag; MLX compiles the Metal sources of `ops` when a kernel first runs."""
    from fastforks._core.mlx import source_tag  # noqa: PLC0415 - needs the mlx extra

    return Library(source_tag("ransac_mlx", Path(__file__).with_name("ops.py")))
