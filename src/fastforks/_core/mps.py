"""Apple MPS: what the portable runners need there beyond plain PyTorch.

MPS has no float16 / bfloat16 3-D transposed convolution ("ConvTranspose 3D with BF16 or FP16 types is not supported
on MPS", torch 2.14), and its transposed convolutions are slow in every dtype. `adapt` replaces those layers: a
stride-2 one (the cost-volume families' 3-D upsampling, most 2-D ones) as one convolution per parity of the output
position, a 2-D one whose kernel equals its stride (ViT necks and DPT heads) as a 1 x 1 convolution and
`pixel_shuffle`, each 1.3-4x faster on an M1 Pro; any other float16 / bfloat16 3-D one in float32 (input and weights
cast for the call, the output cast back). Its 3-D convolutions are slow too (channels-last-3d far more so): `adapt`
runs a plain `Conv3d` as one 2-D convolution over the depth planes per depth tap (`conv3d`, 3-30x faster).

`sync_at_exit` waits for the GPU before the interpreter shuts down: a CPU tensor copied to MPS (a NumPy image, say)
stays referenced by the command buffer, and a NumPy-owned one is freed from Metal's completion thread, which needs the
GIL; if that happens during interpreter shutdown, the process hangs or aborts ("Fatal Python error:
gilstate_tss_set", torch 2.14).

`fence` waits for the GPU at a stage boundary. MPS reuses a freed intermediate only once the command buffer that used
it has completed, so a long queue of stages keeps every stage's temporaries alive, and the allocator backs allocations
of 10-512 MB with 1 GB heaps (counted whole in the process footprint): a synchronize between big stages lets the next
stage reuse the last one's memory (vitmatte-s at 1080p: 3.7 -> 2.6 GB, same speed).

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


def conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
) -> torch.Tensor:
    """`F.conv3d` (groups 1, dilation 1, zero padding) as one `conv2d` over the depth planes per depth tap.

    Output plane `d` sums tap `k` of the input plane `d * stride + k - padding`; the planes of `x` in
    channels-last-3d are channels-last 2-D images (`[N, C, D, H, W]` -> `[N * D, C, H, W]` is a view), and the output
    is channels-last-3d. On an M1 Pro, 192 -> 96 (1, 3, 3) / 96 -> 48 3x3x3 / 48 -> 8 3x3x3 on 18 x 256 x 376 fp32:
    torch NCDHW 481 / 413 / 83 ms, channels-last-3d 3303 / 2512 / 1245 ms, this 105 / 130 / 21 ms (fp16 3-6x).
    """
    n, c, d = x.shape[:3]
    kd, sd, pd = weight.shape[2], stride[0], padding[0]
    d_out = (d + 2 * pd - kd) // sd + 1
    x = x.contiguous(memory_format=torch.channels_last_3d)  # no-op when it already is
    out: torch.Tensor | None = None
    for k in sorted(range(kd), key=lambda k: k != pd):  # the centre tap first: it covers every output plane
        lo = max(0, -((k - pd) // sd))  # first output plane whose input plane d * sd + k - pd is >= 0
        hi = min(d_out - 1, (d - 1 + pd - k) // sd)
        if hi < lo:
            continue
        i0 = lo * sd + k - pd
        planes = x[:, :, i0 : i0 + (hi - lo) * sd + 1 : sd].permute(0, 2, 1, 3, 4).reshape(-1, c, *x.shape[3:])
        full = out is None and lo == 0 and hi == d_out - 1
        y = F.conv2d(planes, weight[:, :, k], bias if full else None, stride[1:], padding[1:])
        y = y.view(n, hi - lo + 1, *y.shape[1:]).permute(0, 2, 1, 3, 4)
        if full:
            out = y
            continue
        if out is None:
            out = y.new_zeros(n, y.shape[1], d_out, *y.shape[3:]).contiguous(memory_format=torch.channels_last_3d)
            if bias is not None:
                out += bias.view(-1, 1, 1, 1)
        out[:, :, lo : hi + 1] += y
    if out is None:  # no tap reaches any output plane (a padding wider than the kernel)
        out = F.conv3d(x, weight, bias, stride, padding)
    return out


def _conv3d(self: torch.nn.Conv3d, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - torch's name
    """`Conv3d.forward` of a layer `adapt` picked, by `conv3d` (in the input dtype)."""
    unbatched = input.dim() == 4  # noqa: PLR2004 - [C, D, H, W]
    x = input.unsqueeze(0) if unbatched else input
    out = conv3d(
        x,
        self.weight.to(x.dtype),
        _cast(self.bias, x),
        cast("tuple[int, int, int]", self.stride),
        cast("tuple[int, int, int]", self.padding),
    )
    return out.squeeze(0) if unbatched else out


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
    conv = F.conv2d if nd == 2 else functools.partial(conv3d, stride=(1, 1, 1), padding=(0, 0, 0))  # noqa: PLR2004 - 2-D
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
    """Replace the 3-D / transposed convolutions of `net` (a runner's module) MPS runs slowly or not at all; in place.

    A `Conv3d` with groups 1, dilation 1 and zero padding goes through `conv3d`.

    A layer `conv_transpose` speeds up (groups 1, dilation 1, the same stride and padding on every axis, square
    kernel equal to the stride or stride 2) goes through it in the input dtype; any other
    `ConvTranspose3d` runs in float32 (a no-op for a float32 one; a later `.half()` stays covered).
    """
    for m in net.modules():
        kind = base_type(m)
        if kind is torch.nn.Conv3d:
            m = cast("torch.nn.Conv3d", m)
            if (
                m.groups == 1
                and set(m.dilation) == {1}
                and m.padding_mode == "zeros"
                and not isinstance(m.padding, str)
            ):
                override(m, _conv3d)
            continue
        if kind not in (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d):
            continue
        m = cast("torch.nn.ConvTranspose2d | torch.nn.ConvTranspose3d", m)
        k, s, p = m.kernel_size, m.stride, m.padding
        plain = m.groups == 1 and set(m.dilation) == {1} and m.padding_mode == "zeros" and len({*s}) == len({*p}) == 1
        if plain and (s[0] == _POLYPHASE or (kind is torch.nn.ConvTranspose2d and set(k) == {s[0]} and p[0] == 0)):
            override(m, _conv_transpose)
        elif kind is torch.nn.ConvTranspose3d:  # whatever dtype the layer gets later (a no-op cast in float32)
            override(m, _float32_conv_transpose3d)


def fence(t: torch.Tensor) -> torch.Tensor:
    """Wait for the queued MPS work when `t` is on MPS (eager only), so freed intermediates become reusable; `t`."""
    if t.device.type == "mps" and not (torch.compiler.is_compiling() or torch.jit.is_tracing()):
        torch.mps.synchronize()
    return t


@functools.cache
def sync_at_exit() -> None:
    """Make the interpreter wait for queued MPS work before it shuts down (once per process)."""
    atexit.register(torch.mps.synchronize)
