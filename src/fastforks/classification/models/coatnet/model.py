"""CoAtNet: the timm network with the variant's weights, run through `fastforks._optimize`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from fastforks.shared import libnet

if TYPE_CHECKING:
    import torch


_ARCH = "coatnet_0_rw_224.sw_in1k"


class Kernels(libnet.PoolKernels):
    """CoAtNet warms each fixed crop and batch on its first eager request."""

    _graph_capture_warmup = False

    def _probe_batchnorm(self, net: torch.nn.Module, /) -> bool:
        ctx = self.ctx
        if (
            sys.platform == "linux"
            and ctx.mode == "kernels"
            and ctx.device.type == "cuda"
            and ctx.dtype == "float16"
            and ctx.variant.ref == f"{_ARCH}@1"
            and ctx.params.arch == _ARCH
            and ctx.user_weights is None
            and ctx.user_state is None
        ):
            lib = libnet.timm()
            arch = lib.models.maxxvit.MaxxVit
            if (
                lib.__version__ == "1.0.30"
                and isinstance(net, arch)
                and getattr(net.forward, "__func__", None) is arch.forward
            ):
                return False  # this fixed architecture's hook probe folds zero pairs; FX still fuses them
        return True


RUNNERS = {
    "portable": libnet.Portable,
    "kernels": Kernels,
    "portable_mlx": libnet.Mlx,
    "kernels_mlx": libnet.Mlx,
}
