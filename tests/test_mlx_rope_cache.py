"""The RoPE position / frequency caches of DA3, VGGT, AnySplat, MegaFlow and Pi3 keep no traced tensor.

An MLX export traces the module with fake tensors. A cache filled while tracing handed that fake tensor to the next
export of the same module (another branch, method or batch size with the same patch grid), which lifted it as a
constant and failed on `.numpy()`.
"""

from __future__ import annotations

import importlib

import pytest
import torch

pytest.importorskip("mlx.core")

from fastforks._core.patch import override
from fastforks.shared import mlx_graph
from fastforks.shared.rope2d import rope_forward

pytestmark = pytest.mark.mlx

MODULES = {
    "da3": "fastforks.depth.models.depth_anything_3.net.dinov2.layers.rope",
    "vggt": "fastforks.multiview.models.vggt.net.layers.rope",
    "anysplat": "fastforks.multiview.models.anysplat.net.layers.rope",
    "megaflow": "fastforks.flow.models.megaflow.net.layers.rope",
    "pi3": "fastforks.multiview.models.pi3.net.layers.pos_embed",
}
H, W = 4, 6


class Net(torch.nn.Module):
    """Positions + RoPE as the ports use them (the overridden forward, or MegaFlow's direct table read)."""

    def __init__(self, family: str) -> None:
        super().__init__()
        mod = importlib.import_module(MODULES[family])
        self.family = family
        if family == "pi3":
            from fastforks.multiview.models.pi3.model import _rope_forward  # noqa: PLC0415

            self.rope = mod.RoPE2D(100.0)
            override(self.rope, _rope_forward)
        else:
            self.rope = mod.RotaryPositionEmbedding2D(100.0)
            if family != "megaflow":
                override(self.rope, rope_forward)
        self.pos = mod.PositionGetter()
        self.lin = torch.nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pos(x.shape[0], H, W, device=x.device)
        t = self.lin(x)
        if self.family == "megaflow":
            cc, sc = self.rope._compute_frequency_components(8, H * W + 1, x.device, x.dtype)
            return t * cc[pos[..., 0]][:, None].repeat(1, 1, 1, 2) + sc[pos[..., 1]][:, None].repeat(1, 1, 1, 2)
        return self.rope(t, pos)


def _net(family: str) -> Net:
    torch.manual_seed(0)
    return Net(family).eval()


@pytest.mark.parametrize("family", list(MODULES))
def test_exports_of_one_module_share_no_fake_tensor(family):
    ref, m = _net(family), mlx_graph.MlxModule(_net(family))
    for b in (2, 3):  # same patch grid and token count: the same cache keys
        x = torch.randn(b, 2, H * W, 16)
        with torch.no_grad():
            want = ref(x)
        torch.testing.assert_close(m(x), want, atol=1e-5, rtol=1e-5)
    # the eager run still caches (plain tensors)
    cached = next(iter(ref.pos.__dict__.values()))
    assert cached
    assert all(type(t) is torch.Tensor for t in cached.values())
