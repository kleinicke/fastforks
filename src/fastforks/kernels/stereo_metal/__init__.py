"""Metal kernels for stereo on Apple MPS (`stereo_kernels.metal`); `modules` wraps them as the portable blocks."""

from __future__ import annotations

from pathlib import Path

from fastforks._core.metal import metal_library

load = metal_library("stereo_metal", Path(__file__).with_name("stereo_kernels.metal"))
