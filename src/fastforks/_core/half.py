"""fp16 weights for the portable runners of networks that otherwise run under `torch.autocast`.

Plain PyTorch only, so the converted network still exports (`torch.export`, ONNX). Two levels:

- `half_weights(net)`: Linear / Conv weights stored in fp16. Autocast casts exactly these weights to fp16 on every call
  (and caches the copy only inside one autocast region), so under autocast the outputs are bit-identical; the fp32
  copy and the per-call cast disappear. Everything else (norms, layer scales, embeddings) stays fp32.
- `pure_half(net)`: the whole network in fp16 without autocast (norms, softmax and sums run in the kernels' fp32
  accumulators, residual streams in fp16); inputs are cast to fp16 and `keep_fp32` modules (e.g. a 1-channel
  output conv) run in fp32. Faster and smaller still, but the outputs differ from autocast at the 1e-4 level.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from fastforks._core.patch import base_type, override

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

_WEIGHTED = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)


_ACC_LOCK = threading.Lock()
_acc = [0, False]  # threads inside `fp16_accumulate`, the flag before the first one entered


@contextlib.contextmanager
def fp16_accumulate() -> Generator[None]:
    """fp16-accumulate cuBLAS GEMMs (`allow_fp16_accumulation`, 2x tensor-core rate on GeForce) inside the block.

    The flag is process-wide: refcounted under a lock, so a thread leaving never resets it under another one still
    inside, and the last one out restores the value from before the first one entered.
    """
    with _ACC_LOCK:
        if _acc[0] == 0:
            _acc[1] = torch.backends.cuda.matmul.allow_fp16_accumulation
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        _acc[0] += 1
    try:
        yield
    finally:
        with _ACC_LOCK:
            _acc[0] -= 1
            if _acc[0] == 0:
                torch.backends.cuda.matmul.allow_fp16_accumulation = _acc[1]


_CUBLAS = contextvars.ContextVar("cublas_gemms", default=False)


@contextlib.contextmanager
def cublas_gemms() -> Generator[None]:
    """The kernels' token GEMMs (`_optimize.conv_ops.lin`) run as cuBLAS `F.linear` + a torch epilogue inside the block.

    Outside `fp16_accumulate` that is fp32 accumulation: the portable precision, at the rate of the GPU's
    fp32-accumulate tensor-core path (half the fp16-accumulate rate on GeForce, the same on data-centre and
    workstation parts). Per thread / context (unlike the cuBLAS flag), read at trace time, so a CUDA graph captured
    inside the block keeps it.
    """
    tok = _CUBLAS.set(True)
    try:
        yield
    finally:
        _CUBLAS.reset(tok)


def cublas_gemms_on() -> bool:
    """Whether the caller is inside `cublas_gemms`."""
    return _CUBLAS.get()


_GRAPH_SIZED = contextvars.ContextVar("graph_sized", default=False)


@contextlib.contextmanager
def graph_sized() -> Generator[None]:
    """The small token GEMMs inside the block run on cuBLAS (`_optimize.conv_ops.lin`, `LIN_GRAPH_ROWS`). Opt-in.

    `shared.libnet.features.Kernels` runs every call small enough for a CUDA graph inside it: the capture, the eager
    calls before it and the eager calls that return hidden states, so they all give the same output. Per thread /
    context, read at trace time. No other runner enters it.
    """
    tok = _GRAPH_SIZED.set(True)
    try:
        yield
    finally:
        _GRAPH_SIZED.reset(tok)


def graph_sized_on() -> bool:
    """Whether the caller is inside `graph_sized`."""
    return _GRAPH_SIZED.get()


def half_weights(net: nn.Module, *, channels_last: bool = True) -> None:
    """Store the Linear / Conv weights of `net` in fp16 (use under `torch.autocast`; see the module docstring)."""
    for m in net.modules():
        if isinstance(m, _WEIGHTED):
            m.half()
    if channels_last:
        net.to(memory_format=torch.channels_last)


def _cast(x: Any, dtype: torch.dtype) -> Any:  # noqa: ANN401 - any forward argument
    return x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x


def _half_forward(self: nn.Module, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - any forward signature
    """The module's own `forward` on fp16 inputs with autocast off."""
    with torch.autocast("cuda", enabled=False):
        return base_type(self).forward(
            self, *(_cast(a, torch.float16) for a in args), **{k: _cast(v, torch.float16) for k, v in kwargs.items()}
        )


def _fp32_forward(self: nn.Module, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - any forward signature
    """The module's own `forward` on fp32 inputs with autocast off."""
    with torch.autocast("cuda", enabled=False):
        return base_type(self).forward(
            self, *(_cast(a, torch.float32) for a in args), **{k: _cast(v, torch.float32) for k, v in kwargs.items()}
        )


def pure_half(net: nn.Module, *, fp32: Iterable[nn.Module] = (), channels_last: bool = True) -> None:
    """Convert `net` to fp16 without autocast; `net` casts its floating inputs to fp16, `fp32` modules stay fp32.

    Args:
        net: The network (its `forward` is overridden per instance, see `fastforks._core.patch`).
        fp32: Submodules of `net` that keep fp32 weights and compute (their input is cast up, output stays fp32).
        channels_last: Also store 4-D weights channels-last.
    """
    net.half()
    if channels_last:
        net.to(memory_format=torch.channels_last)
    for m in fp32:
        m.float()
        override(m, _fp32_forward)
    override(net, _half_forward)
