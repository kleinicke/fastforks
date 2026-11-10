"""ConvNeXt (V1) as a feature backbone: the timm network with the variant's weights, without its classifier.

`last_hidden_state` is the stage-4 map (stride 32) as tokens, row-major and un-normalised: timm's
`forward_features`, the last map of `features_only=True`, HF `ConvNextModel.last_hidden_state`. `pooler_output` is
the head's LayerNorm of their mean: timm's `forward_head(pre_logits=True)`, HF `pooler_output`. There is no prefix
token. `portable` runs the network through `fastforks._optimize`; `kernels` runs the trunk as kernel calls
(`shared.convnext_trunk`, the classification ConvNeXt's).
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from fastforks._core.half import fp16_accumulate, graph_sized
from fastforks._optimize import optimize
from fastforks.features._runner import Forward, Runner
from fastforks.shared import libnet
from fastforks.shared.convnext_trunk import Trunk
from fastforks.shared.cuda_graphs import SmallShapeGraphs

if TYPE_CHECKING:
    from fastforks.features._runner import Context

# CUDA graphs while the call is launch bound or close to it: up to this many input pixels (batch x H x W)
_GRAPH_PIXELS = 1 << 20  # fixed: as the classification ConvNeXt (same trunk): larger calls are GPU bound
_GRAPHS = 4  # fixed: graph cache policy, not architecture


def _network(ctx: Context) -> Any:  # noqa: ANN401 - a timm ConvNeXt
    """The timm network with the variant's weights (CPU, float32), the 21841-class layer dropped."""
    net = cast("Any", libnet.build(ctx))
    net.reset_classifier(0)
    return net


class Portable(Runner):
    """Plain PyTorch through `fastforks._optimize` (functional blocks, folded LayerScale): float32, or fp16 weights."""

    dtypes = ("float32", "float16")
    batch = None
    Params = libnet.Params
    num_prefix_tokens = 0

    def __init__(self, ctx: Context) -> None:
        """Build and optimise the network."""
        super().__init__(ctx)
        net = _network(ctx)
        self.hidden_size, self.num_layers = net.num_features, len(net.stages)
        self.fast = optimize(net, mode="portable", device=ctx.device, fp16=ctx.dtype == "float16")
        self.acc = fp16_accumulate if self.fast._acc else contextlib.nullcontext  # noqa: SLF001 - cuBLAS GEMM flag
        if ctx.device.type == "cuda":  # the fp32 weights stay cached otherwise
            torch.cuda.empty_cache()

    def _run(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net: Any = self.fast.net
        x = net.forward_features(self.fast._input(net, image))  # noqa: SLF001 - the optimizer's input conversion
        return x.flatten(2).transpose(1, 2).float(), net.forward_head(x, pre_logits=True).float()

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch (`layers` is always empty: the card has no hidden_states)."""
        del layers
        with self.acc():
            last, pooled = self._run(image)
        return {"last_hidden_state": last, "pooler_output": pooled}

    def module(self) -> torch.nn.Module:
        """The network: normalised `[B, 3, h, w]` (sides multiples of 32) -> `last_hidden_state` `[B, N, C]`.

        Without the fp16-accumulate switch (a process-global cuBLAS flag that tracing cannot capture).
        """
        return Forward(self.fast.net, lambda _net, x: self._run(x)[0])


class Kernels(Runner):
    """fvm_conv / fvm_seg kernels on fp16 channels-last activations (`shared.convnext_trunk`), the pooling in float32.

    Calls of up to `_GRAPH_PIXELS` replay a CUDA graph per input shape, captured on its second call.
    """

    batch = None
    Params = libnet.Params
    num_prefix_tokens = 0

    def __init__(self, ctx: Context) -> None:
        """Build timm's network from the checkpoint and keep its weights as fp16 channels-last kernel operands."""
        super().__init__(ctx)
        net = _network(ctx)
        self.hidden_size, self.num_layers = net.num_features, len(net.stages)
        self.trunk = Trunk(net, ctx.device)
        self.norm = tuple(t.detach().float().to(ctx.device) for t in (net.head.norm.weight, net.head.norm.bias))
        self._graphs = SmallShapeGraphs(self._forward, pixels=_GRAPH_PIXELS, graphs=_GRAPHS)

    def _forward(self, image: torch.Tensor) -> torch.Tensor:
        """Normalised `[B, 3, H, W]` float32 -> `[B, 1 + N, C]` float32: the pooled embedding, then the tokens."""
        x = self.trunk(image)
        b, c = x.shape[:2]
        t = x.permute(0, 2, 3, 1).reshape(b, -1, c).float()
        pooled = F.layer_norm(t.mean(1), (c,), *self.norm, self.trunk.eps)
        return torch.cat([pooled[:, None], t], 1)

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()
        self.trunk.release()

    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Features of a resized, normalised batch (`layers` is always empty: the card has no hidden_states)."""
        del layers
        with torch.autocast("cuda", enabled=False), fp16_accumulate(), graph_sized():
            out = self._graphs(image)
        return {"last_hidden_state": out[:, 1:], "pooler_output": out[:, 0]}


RUNNERS = {"portable": Portable, "kernels": Kernels}
