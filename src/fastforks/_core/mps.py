"""Apple MPS: what the portable runners need there beyond plain PyTorch.

MPS has no float16 / bfloat16 3-D transposed convolution ("ConvTranspose 3D with BF16 or FP16 types is not supported
on MPS", torch 2.14), and its transposed convolutions are slow in every dtype. `adapt` replaces those layers: a
stride-2 one (the cost-volume families' 3-D upsampling, most 2-D ones) as one convolution per parity of the output
position, a 2-D one whose kernel equals its stride (ViT necks and DPT heads) as a 1 x 1 convolution and
`pixel_shuffle`, each 1.3-4x faster on an M1 Pro; any other float16 / bfloat16 3-D one in float32 (input and weights
cast for the call, the output cast back).

`sync_at_exit` waits for the GPU before the interpreter shuts down: a CPU tensor copied to MPS (a NumPy image, say)
stays referenced by the command buffer, and a NumPy-owned one is freed from Metal's completion thread, which needs the
GIL; if that happens during interpreter shutdown, the process hangs or aborts ("Fatal Python error:
gilstate_tss_set", torch 2.14).

`quiet_nd_padding` silences MPS's note that it pads tensors of more than 3 dimensions as view ops (slower, not wrong).
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import itertools
import warnings
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.patch import base_type, override

_POLYPHASE = 2  # the stride `conv_transpose` splits into output parities

if TYPE_CHECKING:
    from collections.abc import Generator


def _float32_conv_transpose3d(
    self: torch.nn.ConvTranspose3d,
    input: torch.Tensor,  # noqa: A002 - torch's name
    output_size: list[int] | None = None,
) -> torch.Tensor:
    """`ConvTranspose3d.forward` in float32 whatever the parameter and input dtypes (MPS: no fp16 kernel)."""
    pad = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 3, self.dilation)  # ty: ignore[invalid-argument-type] - torch's own call
    bias = None if self.bias is None else self.bias.float()
    with torch.autocast(input.device.type, enabled=False):
        out = F.conv_transpose3d(
            input.float(),
            self.weight.float(),
            bias,
            self.stride,
            self.padding,  # ty: ignore[invalid-argument-type] - never a str for a transposed conv
            pad,
            self.groups,
            self.dilation,
        )
    return out.to(input.dtype)


def conv_transpose(  # noqa: PLR0913, PLR0917 - torch's arguments
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    output_padding: int = 0,
) -> torch.Tensor:
    """`F.conv_transpose2d` / `conv_transpose3d` (groups 1, dilation 1, the same stride and padding on every axis).

    Kernel equal to the stride, no padding (2-D): one 1 x 1 `conv2d` to every output phase, then `pixel_shuffle`.
    Stride 2: one convolution per parity of the output position. Otherwise torch's own op.
    """
    nd = x.dim() - 2
    size = weight.shape[2:]
    sizes = [(n - 1) * stride - 2 * padding + k + output_padding for n, k in zip(x.shape[2:], size, strict=True)]
    if nd == 2 and size[0] == size[1] == stride and padding == output_padding == 0:  # noqa: PLR2004 - 2-D
        out, ch = weight.shape[1], stride * stride
        w = weight.permute(1, 2, 3, 0).reshape(out * ch, -1, 1, 1)
        return F.pixel_shuffle(F.conv2d(x, w, None if bias is None else bias.repeat_interleave(ch)), stride)
    if stride == _POLYPHASE:
        return _polyphase(x, weight, bias, [padding] * nd, sizes)
    op = F.conv_transpose2d if nd == 2 else F.conv_transpose3d  # noqa: PLR2004 - 2-D
    return op(x, weight, bias, stride, padding, output_padding)


def _polyphase(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, p: list[int], sizes: list[int]
) -> torch.Tensor:
    """Stride-2 transposed convolution of `x` `[N, C, *n]` with `weight` `[C, O, *k]` to `sizes`, per output parity.

    Per axis, the outputs `o = 2 m + e` take the taps `k = k0 + 2 t` (`k0 = (e + p) % 2`) at the inputs
    `i = m + c - t` (`c = (e + p - k0) / 2`): a plain convolution with every second tap, flipped.
    """
    nd = x.dim() - 2
    n, size = x.shape[2:], weight.shape[2:]
    taps: list[list[tuple[list[int], int, int]]] = []  # per axis and parity: kernel taps, first / last input
    for d in range(nd):
        axis = []
        for e in (0, 1):
            k0 = (e + p[d]) % 2
            t, c = (size[d] - k0 + 1) // 2, (e + p[d] - k0) // 2
            axis.append(([k0 + 2 * (t - 1 - j) for j in range(t)], c - t + 1, (sizes[d] - e + 1) // 2 - 1 + c))
        taps.append(axis)
    lo = [max(0, -min(a[1] for a in axis)) for axis in taps]
    hi = [max(0, max(a[2] for a in axis) - n[d] + 1) for d, axis in enumerate(taps)]
    x = F.pad(x, [v for d in reversed(range(nd)) for v in (lo[d], hi[d])])
    conv = F.conv2d if nd == 2 else F.conv3d  # noqa: PLR2004 - 2-D
    out = x.new_empty(x.shape[0], weight.shape[1], *sizes)
    for parity in itertools.product((0, 1), repeat=nd):
        axes = [taps[d][e] for d, e in enumerate(parity)]
        phase = (..., *(slice(e, None, 2) for e in parity))
        if not all(k for k, _, _ in axes):  # a kernel shorter than the stride: no tap reaches this parity
            out[phase] = 0 if bias is None else bias.view(-1, *(1,) * nd)
            continue
        if any(last - first + 1 < len(k) for k, first, last in axes):
            continue  # no output of this parity (a size-1 output axis)
        window = x[(..., *(slice(f + lo[d], g + lo[d] + 1) for d, (_, f, g) in enumerate(axes)))]
        taps_ = (slice(k[-1], k[0] + 1, 2) for k, _, _ in axes)  # every second tap (descending: flipped below)
        w = weight[(slice(None), slice(None), *taps_)].flip(list(range(2, 2 + nd)))  # slices: no index upload
        out[phase] = conv(window, w.transpose(0, 1), bias)
    return out


def _conv_transpose(
    self: torch.nn.ConvTranspose2d | torch.nn.ConvTranspose3d,
    input: torch.Tensor,  # noqa: A002 - torch's name
    output_size: list[int] | None = None,
) -> torch.Tensor:
    """`forward` of a layer `adapt` picked, by `conv_transpose` (output padding up to 1 for stride 2)."""
    nd = self.weight.dim() - 2
    unbatched = input.dim() == nd + 1
    if unbatched:
        input = input.unsqueeze(0)  # noqa: A001 - same tensor, a batch of one
    dtype = input.dtype
    if torch.is_autocast_enabled(input.device.type):  # torch's op returns the autocast dtype
        dtype = torch.get_autocast_dtype(input.device.type)
    input = input.to(dtype)  # noqa: A001 - no-op copy when equal
    weight, bias = self.weight.to(dtype), _cast(self.bias, input)
    pad = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, nd, self.dilation)  # ty: ignore[invalid-argument-type] - torch's own call
    padding, stride = cast("tuple[int, ...]", self.padding), self.stride  # never a str here
    if len(set(pad)) > 1 and stride[0] == _POLYPHASE:  # per-axis output padding (an odd `output_size`)
        sizes = [
            (n - 1) * stride[0] - 2 * padding[0] + k + q
            for n, k, q in zip(input.shape[2:], self.kernel_size, pad, strict=True)
        ]
        out = _polyphase(input, weight, bias, list(padding), sizes)
    elif len(set(pad)) > 1:  # other strides: torch's op (the polyphase split is for stride 2 only)
        op = F.conv_transpose2d if nd == 2 else F.conv_transpose3d  # noqa: PLR2004 - 2-D
        out = op(input, weight, bias, stride, padding, pad)
    else:
        out = conv_transpose(input, weight, bias, stride[0], padding[0], pad[0])
    return out.squeeze(0) if unbatched else out


def _cast(bias: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | None:
    return None if bias is None else bias.to(like.dtype)


def quiet_nd_padding(device: torch.device) -> contextlib.AbstractContextManager[object]:
    """Silence MPS's warning that it pads more than 3 dimensions as view ops (slower, not wrong) inside the block.

    A no-op off MPS: a warnings filter is a graph break for `torch.compile(fullgraph=True)`.
    """
    return _quiet_nd_padding() if device.type == "mps" else contextlib.nullcontext()


@contextlib.contextmanager
def _quiet_nd_padding() -> Generator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "MPS: The constant padding of more than 3 dimensions", UserWarning)
        yield


def adapt(net: torch.nn.Module) -> None:
    """Replace the transposed convolutions of `net` (a runner's module) that MPS runs slowly or not at all; in place.

    A layer `conv_transpose` speeds up (groups 1, dilation 1, the same stride and padding on every axis, square
    kernel equal to the stride or stride 2) goes through it in the input dtype; any other
    `ConvTranspose3d` runs in float32 (a no-op for a float32 one; a later `.half()` stays covered).
    """
    for m in net.modules():
        kind = base_type(m)
        if kind not in (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d):
            continue
        m = cast("torch.nn.ConvTranspose2d | torch.nn.ConvTranspose3d", m)
        k, s, p = m.kernel_size, m.stride, m.padding
        plain = m.groups == 1 and set(m.dilation) == {1} and m.padding_mode == "zeros" and len({*s}) == len({*p}) == 1
        if plain and (s[0] == _POLYPHASE or (kind is torch.nn.ConvTranspose2d and set(k) == {s[0]} and p[0] == 0)):
            override(m, _conv_transpose)
        elif kind is torch.nn.ConvTranspose3d:  # whatever dtype the layer gets later (a no-op cast in float32)
            override(m, _float32_conv_transpose3d)


@functools.cache
def sync_at_exit() -> None:
    """Make the interpreter wait for queued MPS work before it shuts down (once per process)."""
    atexit.register(torch.mps.synchronize)
