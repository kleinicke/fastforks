"""FastViT: the timm network with the variant's weights, run through `fastforks._optimize`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastforks._optimize.fold import reparam
from fastforks.shared import libnet

if TYPE_CHECKING:
    import torch

    from fastforks._core.runner import Context


class Mlx(libnet.Mlx):
    """MLX runner of the re-parameterized network (as `_optimize` runs it on the other devices).

    Each RepMixer block computes `x + scale * (mixer(x) - norm(x))`: two nearly equal branches whose difference loses
    its digits in float16 (t8 logits mean |diff| 0.14 against a 0.023 bound). Folded into one conv in float32 at
    build time, the replay computes the difference exactly; the multi-branch MobileOne convs fold the same way.
    """

    def network(self, ctx: Context) -> torch.nn.Module:
        """The timm network with its RepMixer / MobileOne branches folded."""
        net = super().network(ctx)
        reparam(net)
        return net


RUNNERS = {
    "portable": libnet.Portable,
    "kernels": libnet.Kernels,
    "portable_mlx": Mlx,
    "kernels_mlx": Mlx,
}
