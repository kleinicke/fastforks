"""ConvNeXt (V1): the timm network with the variant's weights.

`portable` runs it through `fastforks._optimize` (`shared.libnet`). `kernels` reads the weights out of the same timm
network and runs the blocks as kernel calls (`shared.convnext_trunk`): 10-17 % faster than the generic `_optimize`
ConvNeXt path at batch 1, 9-15 % at batch 8 (RTX 5060 Ti).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from fastforks._core.half import fp16_accumulate, graph_sized
from fastforks._core.runner import Runner
from fastforks.shared import libnet
from fastforks.shared.convnext_trunk import Trunk
from fastforks.shared.cuda_graphs import SmallShapeGraphs

if TYPE_CHECKING:
    from fastforks._core.runner import Context

# CUDA graphs while the call is launch bound or close to it: up to this many input pixels (batch x H x W)
_GRAPH_PIXELS = 1 << 20  # fixed: batch <= 20 at 224, <= 7 at 384; larger calls are GPU bound (graph = eager time)
_GRAPHS = 4  # fixed: graph cache policy (the crop is fixed, so a key is a batch size)


class Kernels(Runner):
    """fvm_conv / fvm_seg kernels on fp16 channels-last activations (`shared.convnext_trunk`), the head in float32.

    Calls of up to `_GRAPH_PIXELS` replay a CUDA graph per batch size, captured on its second call.
    """

    batch = None
    Params = libnet.Params

    def __init__(self, ctx: Context) -> None:
        """Build timm's network from the checkpoint and keep its weights as fp16 channels-last kernel operands."""
        super().__init__(ctx)
        net = cast("Any", libnet.build(ctx))
        self.trunk = Trunk(net, ctx.device)
        head = net.head
        self.head = tuple(
            t.detach().float().to(ctx.device) for t in (head.norm.weight, head.norm.bias, head.fc.weight, head.fc.bias)
        )
        self._graphs = SmallShapeGraphs(self._forward, pixels=_GRAPH_PIXELS, graphs=_GRAPHS)

    def _forward(self, image: torch.Tensor) -> torch.Tensor:
        """Normalised `[B, 3, H, W]` float32 -> logits `[B, classes]` float32."""
        x = self.trunk(image)
        nw, nb, fw, fb = self.head
        pooled = x.permute(0, 2, 3, 1).float().mean((1, 2))
        return F.linear(F.layer_norm(pooled, (pooled.shape[-1],), nw, nb, self.trunk.eps), fw, fb)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()
        self.trunk.release()

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, classes]` of a transformed batch (a CUDA graph replay for small repeated batches)."""
        # `graph_sized` for every call, eager or captured, so the two agree: past `_GRAPH_PIXELS` no GEMM has few
        # enough rows for it to change anything (21 x 49 rows at 224)
        with torch.autocast("cuda", enabled=False), fp16_accumulate(), graph_sized():
            return {"logits": self._graphs(image)}


RUNNERS = {"portable": libnet.Portable, "kernels": Kernels}
