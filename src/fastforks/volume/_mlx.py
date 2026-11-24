"""The MLX runner of an nnU-Net family: every fold network of every part is an exported graph.

The sliding-window loop and the resampling stay PyTorch (CPU); `Predictor` calls `net(window)`, which for an
`MlxModule` runs the window on the GPU through `fastforks.shared.mlx_graph`. With one fold network, no mirroring and
argmax labels (TotalSegmentator), the Gaussian merging and the argmax run on MLX too (`MlxPredictor._accumulate`).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from fastforks.shared import mlx_wrap
from fastforks.volume._nnunet import Predictor, _depth_slices, _labels_dtype, gaussian, sliding_steps

if TYPE_CHECKING:
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
        """`Predictor._accumulate` with the window logits, their float16 sums and the argmax kept on MLX.

        Only for one network without mirroring and argmax labels (the rest: `Predictor`'s CPU loop). The rounding is
        `Predictor`'s: the float32 logits times the float16 Gaussian, added to the float16 sums in float32 and rounded.
        Before, each window's logits (118 heads for TotalSegmentator's total) came back to the CPU as float32 and were
        added there (totalsegmentator-total-3mm: about 330 ms of CPU adds and 150 ms of argmax upload per call).
        """
        from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - MLX only
        from fastforks.shared.mlx_graph import MlxModule  # noqa: PLC0415 - MLX only
        from fastforks.volume._nnunet_runner import Portable  # noqa: PLC0415 - import cycle

        net = self.nets[0]
        # The portable runner's forward is a plain `net(x)`; kernels runners replace it.
        plain = self.forward is None or getattr(self.forward, "__func__", None) is Portable._forward_patch  # noqa: SLF001
        if full or len(self.nets) != 1 or self.mirror_axes or self.plan.regions is not None or not plain:
            return super()._accumulate(x, full=full)
        if not isinstance(net, MlxModule):
            return super()._accumulate(x, full=full)
        p = self.plan
        k0, k1, k2 = p.patch_size
        self._gauss = gaussian(p.patch_size, x.device)
        g = to_mx(self._gauss)
        steps = sliding_steps(x.shape[1:], p.patch_size, self.step)
        d, h, w = x.shape[1:]
        dtype = _labels_dtype(p.n_labels)
        acc = mx.zeros((p.n_outputs, k0, h, w), dtype=mx.float16)
        labels = torch.empty((d, h, w), dtype=dtype)
        for i, a in enumerate(steps[0]):
            starts = [(a, b, c) for b in steps[1] for c in steps[2]]
            for j in range(0, len(starts), self.batch):
                chunk = starts[j : j + self.batch]
                batch = torch.stack([x[:, a0 : a0 + k0, b : b + k1, c : c + k2] for a0, b, c in chunk])
                y = net.raw(batch.to(dtype=self.dtype or batch.dtype)) * g  # [B, heads, *patch] float32
                for n, (a0, b, c) in enumerate(chunk):
                    ys, xs = slice(b, b + k1), slice(c, c + k2)
                    for physical, local in _depth_slices(a0, k0, k0):
                        acc[:, physical, ys, xs] = (acc[:, physical, ys, xs] + y[n, :, local]).astype(mx.float16)
                mx.eval(acc)
                del y
            end = steps[0][i + 1] if i + 1 < len(steps[0]) else d
            for physical, local in _depth_slices(a, end - a, k0):
                idx = mx.argmax(acc[:, physical], axis=0).astype(mx.uint8 if dtype == torch.uint8 else mx.int16)
                if i + 1 < len(steps[0]):
                    acc[:, physical] = mx.zeros((1,), dtype=mx.float16)
                mx.eval(idx, acc)
                labels[a + local.start : a + local.stop] = torch.from_numpy(np.array(idx))
        return labels


def port(portable: type[Runner]) -> type[Runner]:
    """`portable` with its fold networks on MLX."""

    class Graph(portable):  # ty: ignore[unsupported-base]
        """The runner whose fold networks are `MlxModule`s."""

        def __init__(self, ctx: Context) -> None:
            """Build the PyTorch runner, then replace each network by its MLX graph."""
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
                    MlxModule(net, dtype, kernels=kernels, release=True) for net in part.nets
                ]

    Graph.__name__ = Graph.__qualname__ = f"Graph{portable.__name__}"
    return cast("type[Runner]", mlx_wrap.port(Graph, attrs=()))
