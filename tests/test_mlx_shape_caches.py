"""Per-shape constant caches (EMA-VFI, PointDiT) keep no traced or fake tensor.

An MLX export traces the module with fake tensors (and first runs it once in a `FakeTensorMode` to find the tensors it
reads). A cache filled there handed that tensor to the next export of the same module, which lifted it as a constant
and failed on `.numpy()`. Weights-free: the cached methods run on small stand-ins.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

pytest.importorskip("mlx.core")

from fastforks.depth.models.pointdit.model import Kernels as PointDiTKernels
from fastforks.depth.models.pointdit.net import GRID, PointDiT
from fastforks.interpolation.models.ema_vfi.model import _constants
from fastforks.shared import mlx_graph

pytestmark = pytest.mark.mlx

C, H, W = 64, 4, 6


def _plain(cache: dict) -> bool:
    return all(type(t) is torch.Tensor for v in cache.values() for t in (v if isinstance(v, tuple) else (v,)))


class Dit(torch.nn.Module):
    """PointDiT's position and RoPE tables (its own methods) on a random 32 x 32 grid, no checkpoint."""

    position = PointDiT.position
    _position = PointDiT._position
    rope = PointDiT.rope
    _rope_table = staticmethod(PointDiT._rope_table)

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.register_buffer("pos", torch.randn(1, C, GRID, GRID))
        self.register_buffer("const", torch.randn(C))
        self.lin = torch.nn.Linear(C, C)
        self._pos: dict = {}
        self._rope: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rope(H, W, x.device)
        return self.lin(x + self.position(H, W)) * cos[:, 0, 0] + sin[:, 0, 0]


def test_pointdit_exports_share_no_fake_tensor():
    ref, net = Dit().eval(), Dit().eval()
    m = mlx_graph.MlxModule(net)
    for b in (2, 3):  # the same patch grid: the same cache keys
        x = torch.randn(b, H * W, C)
        with torch.no_grad():
            want = ref(x)
        torch.testing.assert_close(m(x), want, atol=1e-5, rtol=1e-5)
    assert _plain(net._pos)
    assert _plain(net._rope)
    assert ref._pos
    assert _plain(ref._pos)  # the eager run still caches (plain tensors)


def test_pointdit_fake_run_caches_nothing():
    net = Dit().eval()
    with FakeTensorMode(allow_non_fake_inputs=True) as fake, torch.no_grad():
        net(fake.from_tensor(torch.randn(1, H * W, C)))
    assert not net._pos
    assert not net._rope
    stub = SimpleNamespace(_cs={})
    with FakeTensorMode():
        PointDiTKernels._table(stub, H, W, torch.device("cpu"))  # ty: ignore[invalid-argument-type]
    assert not stub._cs
    PointDiTKernels._table(stub, H, W, torch.device("cpu"))  # ty: ignore[invalid-argument-type]
    assert _plain(stub._cs)


class Ema(torch.nn.Module):
    """EMA-VFI's per-shape attention mask and position windows of a shifted, padded block."""

    def __init__(self) -> None:
        super().__init__()
        self.blk = SimpleNamespace(window_size=(4, 4), shift_size=(2, 2))
        self.cache: dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cor, mask = _constants(self.blk, (x.shape[0], H, W), x, self.cache)  # ty: ignore[invalid-argument-type]
        return x[:, None] * cor.sum() + mask.sum()


def test_ema_vfi_exports_share_no_fake_tensor():
    ref, net = Ema(), Ema()
    x = torch.randn(2, 8)
    want = ref(x)
    for _ in range(2):  # a fresh export of the same shape (same cache key)
        m = mlx_graph.MlxModule(net)
        torch.testing.assert_close(m(x), want, atol=1e-4, rtol=1e-5)
    assert _plain(net.cache)
    assert ref.cache
    assert _plain(ref.cache)


def test_position_encodings_cache_no_fake_tensor():
    """SAM 2's and RF-DETR's sine position caches (keyed by the feature map's shape)."""
    from fastforks.promptable_segmentation.models.sam2.net.position_encoding import (  # noqa: PLC0415
        PositionEmbeddingSine as Sam2Sine,
    )
    from fastforks.shared.rfdetr.net.models.position_encoding import PositionEmbeddingSine  # noqa: PLC0415
    from fastforks.shared.rfdetr.net.utilities.tensors import NestedTensor  # noqa: PLC0415

    sam2, rfdetr = Sam2Sine(32, warmup_cache=False).eval(), PositionEmbeddingSine(16, normalize=True).eval()
    x = torch.randn(1, 8, H, W)

    def run(t: torch.Tensor) -> None:
        sam2(t)
        rfdetr(NestedTensor(t, torch.zeros(1, H, W, dtype=torch.bool), no_padding=True))

    with FakeTensorMode(allow_non_fake_inputs=True) as fake:
        run(fake.from_tensor(x))
    assert not sam2.cache
    assert not rfdetr._pos_cache
    run(x)  # eager: cached
    assert _plain(sam2.cache)
    assert rfdetr._pos_cache
    assert _plain(rfdetr._pos_cache)
