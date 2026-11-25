"""The MLX runner of an nnU-Net family: every fold network of every part runs on MLX.

A plain 3D U-Net with 3x3x3 kernels (`mlx_net.supported`: TotalSegmentator's) is `mlx_net.MlxUNet`, written in
`mlx.core` operations, its parameters converted once into the conversion cache (`fastforks._core.mlx.converted`): no
PyTorch network is built. Any other network is an exported graph (`fastforks.shared.mlx_graph.MlxModule`).

The resampling stays PyTorch (CPU). With one fold network, no mirroring and argmax labels (TotalSegmentator), the
sliding window runs on MLX channels-last (`MlxPredictor._accumulate`): the windows are cut from the volume on the GPU,
their logits weighted, summed and decided there; otherwise `Predictor` calls `net(window)` window by window.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from fastforks.shared import mlx_wrap
from fastforks.volume._nnunet import Predictor, _depth_slices, _labels_dtype, gaussian, sliding_steps

if TYPE_CHECKING:
    from fastforks.volume._nnunet import Plan
    from fastforks.volume._runner import Context, Runner


class MlxPredictor(Predictor):
    """`Predictor` whose argmax over the heads runs on the GPU (the CPU argmax of 100+ heads is a second per part)."""

    def decide(self, logits: torch.Tensor, c0: int = 0, seg: torch.Tensor | None = None) -> torch.Tensor:
        """`Predictor.decide`; without regions the argmax runs on MLX."""
        if self.plan.regions is not None:
            return super().decide(logits, c0, seg)
        from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - MLX only

        dtype = _labels_dtype(self.plan.n_labels)
        idx = mx.argmax(to_mx(logits), axis=0).astype(mx.uint8 if dtype == torch.uint8 else mx.int16)
        mx.eval(idx)
        return torch.from_numpy(np.array(idx))

    def _accumulate(self, x: torch.Tensor, *, full: bool) -> torch.Tensor:
        """`Predictor._accumulate` with the windows, their weighted logit sums and the argmax on MLX, channels-last.

        Only for one `MlxUNet` without mirroring and argmax labels (the rest: `Predictor`'s loop). The rounding is
        `Predictor`'s: each window's logits (run dtype) times the float16 Gaussian, added to the float16 sums and
        rounded. The sums are `[patch depth, H, W, heads]`, so a window's logits are added without a transpose and the
        argmax reads the heads contiguously; one window per network call (two only raise the peak).
        """
        from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - MLX only
        from fastforks.volume._nnunet_runner import Portable  # noqa: PLC0415 - import cycle
        from fastforks.volume.mlx_net import MlxUNet  # noqa: PLC0415 - MLX only

        net = self.nets[0]
        # The portable runner's forward is a plain `net(x)`; kernels runners replace it.
        plain = self.forward is None or getattr(self.forward, "__func__", None) is Portable._forward_patch  # noqa: SLF001
        if full or len(self.nets) != 1 or self.mirror_axes or self.plan.regions is not None or not plain:
            return super()._accumulate(x, full=full)
        if not isinstance(net, MlxUNet):
            return super()._accumulate(x, full=full)
        p = self.plan
        k0, k1, k2 = p.patch_size
        self._gauss = gaussian(p.patch_size, x.device)
        g = to_mx(self._gauss)[..., None]
        steps = sliding_steps(x.shape[1:], p.patch_size, self.step)
        d, h, w = x.shape[1:]
        dtype = _labels_dtype(p.n_labels)
        xm = mx.contiguous(to_mx(x).transpose(1, 2, 3, 0).astype(net.dtype))  # [D, H, W, C]
        acc = mx.zeros((k0, h, w, p.n_outputs), dtype=mx.float16)
        labels = torch.empty((d, h, w), dtype=dtype)
        for i, a in enumerate(steps[0]):
            for b in steps[1]:
                for c in steps[2]:
                    ys, xs = slice(b, b + k1), slice(c, c + k2)
                    y = net.logits(xm[None, a : a + k0, ys, xs])[0] * g  # [*patch, heads]
                    for physical, local in _depth_slices(a, k0, k0):
                        acc[physical, ys, xs] = (acc[physical, ys, xs] + y[local]).astype(mx.float16)
                    mx.eval(acc)
                    del y
            end = steps[0][i + 1] if i + 1 < len(steps[0]) else d
            for physical, local in _depth_slices(a, end - a, k0):
                idx = mx.argmax(acc[physical], axis=-1).astype(mx.uint8 if dtype == torch.uint8 else mx.int16)
                if i + 1 < len(steps[0]):
                    acc[physical] = mx.zeros((1,), dtype=mx.float16)
                mx.eval(idx, acc)
                labels[a + local.start : a + local.stop] = torch.from_numpy(np.array(idx))
        return labels


def port(portable: type[Runner]) -> type[Runner]:
    """`portable` with its fold networks on MLX."""

    class Graph(portable):  # ty: ignore[unsupported-base]
        """The runner whose fold networks are `MlxUNet`s (or `MlxModule`s)."""

        def __init__(self, ctx: Context) -> None:
            """Build the runner with MLX networks, then replace each exported network by its MLX graph."""
            from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - MLX only (imports mlx)

            super().__init__(ctx)
            dtype = mlx_wrap.run_dtype(ctx)
            kernels = ctx.mode == "kernels"
            self.parts = [
                MlxPredictor(**{f.name: getattr(part, f.name) for f in dataclasses.fields(part) if f.init})
                for part in self.parts
            ]
            for part in self.parts:
                part.nets = [  # ty: ignore[invalid-assignment]
                    net
                    if not isinstance(net, torch.nn.Module)
                    else MlxModule(net, dtype, kernels=kernels, release=True)
                    for net in part.nets
                ]

        def _net(self, index: int, plan: Plan, channels: int) -> Any:  # noqa: ANN401 - an MlxUNet or a torch module
            """`MlxUNet` from the conversion cache where it runs the plan, else the PyTorch network (exported later)."""
            from fastforks._core.mlx import converted  # noqa: PLC0415 - MLX only
            from fastforks.volume import mlx_net  # noqa: PLC0415 - MLX only
            from fastforks.volume._nnunet import UNet  # noqa: PLC0415 - import cycle
            from fastforks.volume._nnunet_runner import CHECKPOINT_GLOBALS  # noqa: PLC0415 - import cycle

            if not mlx_net.supported(plan):
                return super()._net(index, plan, channels)

            def build() -> dict[str, Any]:
                sd = self.ctx.state_dict(index, key="network_weights", strip="", extra=CHECKPOINT_GLOBALS)
                return mlx_net.params(UNet.state_dict_of(sd, plan.dims), plan)

            arrays = converted(self.ctx, mlx_net.LAYOUT, build, weights=index)
            return mlx_net.MlxUNet(arrays, plan, mlx_wrap.run_dtype(self.ctx))

    Graph.__name__ = Graph.__qualname__ = f"Graph{portable.__name__}"
    # A window's transients (Winograd bands, the 128^3 x 32 maps) are fresh zero-filled allocations past the cache:
    # Dataset291 ran 420 ms per window at the base 128 MB, 350 ms at 2 GB (the cap at 16 GB), 255 ms unlimited.
    return cast("type[Runner]", mlx_wrap.port(Graph, attrs=(), cache_mb=2048))
