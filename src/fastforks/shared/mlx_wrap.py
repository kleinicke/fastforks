"""Apple MLX ports of a family's PyTorch runner: its network runs through `fastforks.shared.mlx_graph`.

`port(Portable)` is the family's `portable_mlx` / `kernels_mlx` runner: the portable runner is built on the CPU in float32
(its pre- and post-processing stays PyTorch on the CPU), then each network attribute (`attrs`, default `net`) is replaced
by an `MlxModule`, which exports that module once per input signature, converts its parameters to the run dtype and
replays the graph with `mx.compile` on the GPU. Kernels mode (`ctx.mode == "kernels"`) swaps in our Metal kernels
(`fastforks.kernels.graph_mlx`) for depthwise convolution and global pooling.
"""

from __future__ import annotations

import contextvars
import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from fastforks.shared.mlx_graph import MlxModule

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.runner import Context, Runner

__all__ = ["port", "run_dtype"]

POST_ON_MLX: contextvars.ContextVar[bool] = contextvars.ContextVar("mlx_post", default=False)
"""Set while an MLX runner computes: segmentation's logits resize and argmax run on MLX (`fastforks.shared.mlx_seg`)."""

_RUN_DTYPE: contextvars.ContextVar[str | None] = contextvars.ContextVar("mlx_run_dtype", default=None)


def run_dtype(ctx: Context) -> str:
    """The dtype the network runs in: while an MLX runner builds its float32 PyTorch twin, the MLX run dtype.

    A runner that rescales weights for float16 (to keep activations finite) asks this instead of `ctx.dtype`.
    """
    return _RUN_DTYPE.get() or ctx.dtype


class _Method(torch.nn.Module):
    """A runner method (tensors in, tensors out) as a module that owns the runner's network: what MLX exports."""

    def __init__(self, fn: Callable[..., Any], net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net
        self.fn = fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.fn(*args, **kwargs)


def _resolve(obj: Any, path: str) -> Any:  # noqa: ANN401
    """`obj.a.b` for the dotted `path` `"a.b"`."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def port(  # noqa: C901 - one builder for the four wrapping styles
    portable: type[Runner],
    *,
    attrs: tuple[str, ...] = ("net",),
    whole: bool = False,
    methods: tuple[str, ...] = (),
    net_attr: str = "net",
) -> type[Runner]:
    """The MLX runner of `portable`: same call, the module(s) named by `attrs` run on MLX in `ctx.dtype`.

    `methods`: names of runner methods (pure tensor computation on `self.net`) that run on MLX as one graph each; the
    instance's method is replaced by the MLX call (use with `attrs=()` when the runner reads inside its network);
    `net_attr` names the module they compute with.

    `whole=True`: the runner's `module()` (the network with its pre- and post-processing, image in, image out) runs on
    MLX as one graph and `__call__` returns `{"image": ...}` from it; for a runner whose forward is not `net(x)`.
    """

    class Mlx(portable):  # ty: ignore[unsupported-base]
        __doc__ = f"Apple MLX: `{portable.__name__}` with its network run by the exported-graph translation."

        def __init__(self, ctx: Context) -> None:
            """Build the PyTorch runner in float32 on the CPU, then hand its network(s) to MLX."""
            token = _RUN_DTYPE.set(ctx.dtype)
            try:
                super().__init__(dataclasses.replace(ctx, dtype="float32"))
            finally:
                _RUN_DTYPE.reset(token)
            kernels = ctx.mode == "kernels"
            for name in methods:
                owner, _, leaf = name.rpartition(".")
                host = _resolve(self, owner) if owner else self
                if isinstance(host, torch.nn.Module):  # a method of a submodule: it owns the weights it computes with
                    mod = MlxModule(_Method(getattr(host, leaf), host), ctx.dtype, kernels=kernels)
                    object.__setattr__(host, leaf, mod)  # the instance entry beats the class's function
                else:
                    net = getattr(self, net_attr)
                    setattr(host, leaf, MlxModule(_Method(getattr(host, leaf), net), ctx.dtype, kernels=kernels))
            if whole:
                self._whole = MlxModule(self.module(), ctx.dtype, kernels=kernels)
                return
            for name in attrs:
                owner, _, leaf = name.rpartition(".")
                host = _resolve(self, owner) if owner else self
                mod = MlxModule(getattr(host, leaf), ctx.dtype, kernels=kernels)
                if isinstance(host, torch.nn.Module):  # the MLX callable is no child module
                    host._modules.pop(leaf, None)  # noqa: SLF001 - the module registry
                    object.__setattr__(host, leaf, mod)
                else:
                    setattr(host, leaf, mod)

        if hasattr(portable, "labels"):  # segmentation: the logits' resize and argmax run on MLX too

            def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                """The runner's call with the post-processing on MLX."""
                token = POST_ON_MLX.set(True)
                try:
                    return super().__call__(*args, **kwargs)
                finally:
                    POST_ON_MLX.reset(token)

            def labels(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                """The labels-only call with the post-processing on MLX."""
                token = POST_ON_MLX.set(True)
                try:
                    return super().labels(*args, **kwargs)
                finally:
                    POST_ON_MLX.reset(token)

        if whole:

            def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
                """The module's output on MLX."""
                return {"image": self._whole(image)}

    Mlx.__name__ = Mlx.__qualname__ = f"Mlx{portable.__name__}"
    return Mlx
