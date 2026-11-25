"""RealPLKSR: the card's variant, `load(path)` detection and the network against upstream's forward."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks._core.errors import WeightsError
from fastforks.restoration.models.realplksr import model
from fastforks.restoration.models.realplksr.net import realplksr

if TYPE_CHECKING:
    from pathlib import Path


def test_card_variant() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "realplksr"}
    assert sorted(cards) == ["realplksr-nomoswebphoto-x4"]


@pytest.mark.parametrize(("scale", "use_ea"), [(4, True), (2, False)])
def test_load_file_matches_upstream(tmp_path: Path, scale: int, use_ea: bool) -> None:  # noqa: FBT001
    torch.manual_seed(0)
    net = realplksr(dim=16, n_blocks=2, upscaling_factor=scale, kernel_size=7, split_ratio=0.25, use_ea=use_ea).eval()
    save_file(net.state_dict(), tmp_path / "w.safetensors")
    m = restoration.load(tmp_path / "w.safetensors", device="cpu", dtype="float32")
    assert m.variant.id == "realplksr-nomoswebphoto-x4"
    img = synthetic_image(17, 23, torch.device("cpu"))
    with torch.inference_mode():
        want = net(img.clone())
        raw = m._runner(img)["image"]
        exported = m.module()(img)
    torch.testing.assert_close(raw, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(exported, raw.clamp(0, 1), rtol=0, atol=1e-6)
    assert m(img).image.shape == (1, 3, 17 * scale, 23 * scale)


def test_from_checkpoint_rejects_dysample() -> None:
    sd = realplksr(dim=8, n_blocks=1, kernel_size=3).state_dict()
    assert model.from_checkpoint(sd) is not None
    with pytest.raises(WeightsError, match="DySample"):
        model.from_checkpoint({**sd, "to_img.init_pos": torch.zeros(1)})


@pytest.mark.mlx
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("shape", [(1, 16, 17, 40, 56), (2, 4, 7, 9, 13), (1, 6, 3, 5, 4)])
def test_mlx_large_kernel_matches_conv2d(monkeypatch: pytest.MonkeyPatch, dtype: str, shape: tuple[int, ...]) -> None:
    """The row GEMMs against `F.conv2d`: images smaller than the kernel, batches, and bands of rows."""
    b, c, k, h, w = shape
    mx = pytest.importorskip("mlx.core")
    import numpy as np  # noqa: PLC0415

    from fastforks.restoration.models.realplksr import mlx_ops  # noqa: PLC0415

    torch.manual_seed(c + k)
    x = torch.randn(b, c, h, w)
    weight = torch.randn(c, c, k, k) / (c**0.5 * k)
    want = F.conv2d(x, weight, padding=k // 2).permute(0, 2, 3, 1).numpy()
    dt = getattr(mx, dtype)
    tol = 2e-2 if dtype == "float16" else 1e-4
    args = (mx.array(x.permute(0, 2, 3, 1).numpy()).astype(dt), mx.array(weight.numpy()).astype(dt))
    for band in (mlx_ops.BAND_BYTES, 3 * (k + 2) * w * k * c * 4, 1):  # one band, bands of some rows, of one row
        monkeypatch.setattr(mlx_ops, "BAND_BYTES", band)
        got = mlx_ops.rows_conv(*args)
        assert got.dtype == dt
        np.testing.assert_allclose(np.array(got.astype(mx.float32)), want, atol=tol, rtol=tol)


@pytest.mark.mlx
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_mlx_mish_matches_torch(dtype: str) -> None:
    """The Mish kernel against `F.mish`, through the softplus threshold and both tails."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np  # noqa: PLC0415

    from fastforks.kernels.graph_mlx import ops  # noqa: PLC0415

    torch.manual_seed(0)
    x = torch.cat([torch.randn(1, 4, 5, 7).flatten() * 4, torch.tensor([-60.0, -20.0, 0.0, 19.9, 20.1, 60.0, 0.5])])
    x = x.reshape(1, 3, 7, 7).to(getattr(torch, dtype)).float()
    want = F.mish(x).numpy()
    got = ops.mish(mx.array(x.numpy()).astype(getattr(mx, dtype)))
    assert got.dtype == getattr(mx, dtype)
    tol = 2e-3 if dtype == "float16" else 1e-5
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), want, atol=tol, rtol=tol)
