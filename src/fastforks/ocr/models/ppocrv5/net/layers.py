"""Building blocks shared by the PP-OCRv5 networks, and the Paddle layer types they load from."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Generator


class LegacyBatchNorm(nn.BatchNorm2d):
    """Paddle's `nn.BatchNorm` (parameters named `batch_norm_N`, not `batch_norm2d_N`); same maths."""


class LearnableAffineBlock(nn.Module):
    """`scale * x + bias` with scalar parameters (PP-LCNetV3).

    `freeze_affine` stores the parameters as Python numbers (`consts`): a one-element CUDA tensor broadcasts through
    PyTorch's strided elementwise kernel, a number through its vectorised one, with the same result bits (both compute
    in float from the same values; 6 x 240 x 12 x 665 float16: 0.25 -> 0.19 ms).
    """

    consts: tuple[float, float] | None = None

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.consts is not None:
            scale, bias = self.consts
            return x if (scale, bias) == (1.0, 0.0) else x * scale + bias  # identity: folded into a conv
        return x * self.scale + self.bias


def freeze_affine(net: nn.Module) -> None:
    """Store every `LearnableAffineBlock`'s parameters, as they are now (device dtype), as Python numbers."""
    for m in net.modules():
        if isinstance(m, LearnableAffineBlock):
            m.consts = (m.scale.item(), m.bias.item())


class ConvBN(nn.Module):
    """Convolution without bias + BatchNorm2D (+ activation)."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        act: nn.Module | None = None,
        padding: int | tuple[int, int] | None = None,
        bn: type[nn.BatchNorm2d] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        kh, kw = (k, k) if isinstance(k, int) else k
        pad = ((kh - 1) // 2, (kw - 1) // 2) if padding is None else padding
        self.conv = nn.Conv2d(cin, cout, (kh, kw), stride, pad, groups=groups, bias=False)
        self.bn = bn(cout)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def up(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Paddle `F.upsample(..., mode="nearest")` by an integer factor."""
    return F.interpolate(x, scale_factor=factor, mode="nearest")


PADDLE_TYPES: tuple[tuple[type[nn.Module], str], ...] = (  # checked in order: subclasses first
    (LegacyBatchNorm, "batch_norm"),
    (nn.BatchNorm2d, "batch_norm2d"),
    (nn.ConvTranspose2d, "conv2d_transpose"),
    (nn.Conv2d, "conv2d"),
    (nn.Linear, "linear"),
    (nn.LayerNorm, "layer_norm"),
    (LearnableAffineBlock, "learnable_affine_block"),
)


@contextlib.contextmanager
def creation_order() -> Generator[list[nn.Module]]:
    """Collect the parameter layers in the order they are attached while a network is built.

    Paddle numbers its parameters per layer type in creation order; a layer attached right after it is created (as
    upstream's code and this port do) is attached in that order too, whatever container it ends up in.
    """
    seen: list[nn.Module] = []
    kinds = tuple(t for t, _ in PADDLE_TYPES)

    def hook(_parent: nn.Module, _name: str, sub: nn.Module | None) -> None:
        if isinstance(sub, kinds) and all(sub is not s for s in seen):
            seen.append(sub)

    handle = nn.modules.module.register_module_module_registration_hook(hook)
    try:
        yield seen
    finally:
        handle.remove()


def paddle_layers(net: nn.Module, created: list[nn.Module]) -> dict[str, list[tuple[str, nn.Module]]]:
    """Paddle layer type -> `(qualified name, module)` in Paddle's numbering order.

    Re-parameterised convolutions (`late = True`, PP-LCNetV3) were created by Paddle's export after the whole network
    and come last among the convolutions.
    """
    names = {id(m): n for n, m in net.named_modules()}
    out: dict[str, list[tuple[str, nn.Module]]] = {}
    for m in sorted(created, key=lambda m: bool(getattr(m, "late", False))):  # stable: creation order otherwise
        if id(m) not in names:
            continue
        kind = next(k for t, k in PADDLE_TYPES if isinstance(m, t))
        out.setdefault(kind, []).append((names[id(m)], m))
    return out
