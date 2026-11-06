"""RAFT-Stereo on Apple MPS with Metal kernels (`fastforks.kernels.stereo_metal`): the portable network with fused
ConvGRUs and a one-kernel correlation lookup over a pyramid stored in one buffer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastforks.kernels.stereo_metal.modules import FlatCorr, fuse_grus

from ..net.update import ConvGRU

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

    from ..net.raft_stereo import RAFTStereo


def patch(net: RAFTStereo, lib: MetalLibrary) -> None:
    """Switch `net` (a portable RAFT-Stereo, NCHW update loop) to the Metal kernels; in place."""
    fuse_grus(net.update_block, ConvGRU, lib)
    net.corr_block = type("MetalCorr", (FlatCorr,), {"lib": lib})
    net.args.corr_implementation = "custom"
