"""CoAtNet: the timm network with the variant's weights, run through `fastforks._optimize`."""

from __future__ import annotations

from fastforks.shared import libnet


class Kernels(libnet.PoolKernels):
    """CoAtNet warms each fixed crop and batch on its first eager request."""

    _graph_capture_warmup = False


RUNNERS = {
    "portable": libnet.Portable,
    "kernels": Kernels,
    "portable_mlx": libnet.Mlx,
    "kernels_mlx": libnet.Mlx,
}
