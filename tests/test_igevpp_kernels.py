"""IGEV++ / GREAT-IGEV max (`igevpp` kernels): a batch of two == two single calls, and each GREAT kernel override
against the module's own forward at tiny shapes that hit the edge cases (disparity planes d >= W, outlook maps with
fewer pixels than k^4, a correlation volume wider than the image)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastforks import stereo
from fastforks._core.errors import WeightsError
from fastforks._core.patch import base_type
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cnn_kernels import CL
from fastforks.stereo.models.great_igev.kernels import inference as gx
from fastforks.stereo.models.igev_plusplus.kernels import inference as ix
from synth import pair

pytestmark = [pytest.mark.model, pytest.mark.gpu]

REFS = ["igev-plusplus-sceneflow@1", "great-igev-sceneflow@1"]
SIZE = (67, 99)  # small and odd: padding, and widths below the 1/4-resolution disparity range


@pytest.fixture(autouse=True)
def _plans_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """cuDNN plans on a geometry's first call: every call runs the same convolutions."""
    monkeypatch.setattr(cudnn_hacc, "BUILD_AFTER", 0)


def _load(ref: str) -> stereo.StereoModel:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        return stereo.load(ref, mode="kernels", device="cuda", options={"iters": 2, "cuda_graphs": False})
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")


@pytest.mark.parametrize("ref", REFS)
def test_kernels_batch_equals_singles(ref):
    model = _load(ref)
    inputs = [pair(*SIZE, seed=s) for s in (0, 1)]
    batch = model(*(np.stack(xs) for xs in zip(*inputs, strict=True))).disparity
    for i, (left, right) in enumerate(inputs):
        single = model(left, right).disparity
        assert batch[i].shape == single.shape
        torch.testing.assert_close(batch[i], single, rtol=0, atol=2e-3)


def _net(ref: str) -> torch.nn.Module:
    return _load(ref)._runner.net


def _modules(net: torch.nn.Module, name: str) -> list[torch.nn.Module]:
    return [m for m in net.modules() if base_type(m).__name__ == name]


def _rand(*shape: int, fmt: torch.memory_format = CL, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, generator=g, device="cuda").half()
    return x.contiguous(memory_format=fmt)


def _close(got: torch.Tensor, ref: torch.Tensor, tol: float) -> None:
    """max |got - ref| <= tol * max |ref| (fp16 activations, fp32 accumulation in both paths)."""
    err = (got.float() - ref.float()).abs().max().item()
    assert err <= tol * ref.float().abs().max().item(), err


@torch.no_grad()
def test_great_kernels_match_module_forwards():
    net = _net("great-igev-sceneflow@1")
    eav = net.excitive_attention_volume
    # cc_stem0 (split 1x3x3 conv summed per disparity plane): W = 11 < D = 48, so most planes have d >= W
    c, h, w = eav.cc_stem[0].conv.weight.shape[1] // 2, 9, 11
    left, right, ctx = _rand(1, c, h, w, seed=1), _rand(1, c, h, w, seed=2), _rand(1, 8, h, w, seed=3)
    _close(gx._cost_volume_kernels(eav, left, right, ctx), base_type(eav).cost_volume(eav, left, right, ctx), 2e-2)
    # outlook attention + fold for every (k, C), at 3 x 4 pixels (fewer than k^4 logits per map)
    seen = set()
    for m in _modules(net, "OutlookAttentionLayer"):
        key = (m.kernel_size, m.norm.normalized_shape[0])
        if key in seen:
            continue
        seen.add(key)
        x = _rand(1, key[1], 3, 4, seed=4)
        _close(gx._outlook_forward(m, x), base_type(m).forward(m, x), 2e-2)
    assert seen == gx._OUTLOOK
    # volume attention + feed-forward (in place on a channels-last-3d volume) for every C
    seen = set()
    for m in _modules(net, "VolumeTransformerBlock"):
        cv = m.volume_attn.norm.normalized_shape[0]
        if cv in seen:
            continue
        seen.add(cv)
        vol, feats = _rand(1, cv, 5, 3, 4, fmt=torch.channels_last_3d, seed=5), _rand(1, cv, 3, 4, seed=6)
        ref = base_type(m).forward(m, vol.clone(), feats)
        _close(gx._volume_block_forward(m, vol.clone(), feats), ref, 2e-2)
    assert seen == gx._VOL
    # feed-forward (LayerNorm kernel, dw357) and the final softmax + regression
    for m in _modules(net, "FeedForwardLayer")[:2]:
        x = _rand(1, m.norm.normalized_shape[0], 5, 7, seed=7)
        _close(gx._ffn_forward(m, x), base_type(m).forward(m, x), 2e-2)
    t = _rand(1, 48, 5, 7, 8, fmt=torch.contiguous_format, seed=8)  # [B, D, H, W, C] tokens
    g = (_rand(1, 5, 7, 8, fmt=torch.contiguous_format, seed=9).float() * 8**-0.5).contiguous()
    got = torch.empty((1, 1, 5, 7), device="cuda")
    gx.igevpp_kernels.load(t.device).launch("vol_regress_c8", *gx.linear(35), t, g, got, 1, 48, 35, 1e-6)
    want = gx.disparity_regression(gx.volume_softmax(t, g, 1e-6), 48)
    _close(got, want, 1e-3)


@torch.no_grad()
def test_igevpp_gwc_volume_wider_than_image():
    _load("igev-plusplus-sceneflow@1")  # loads the kernels
    left, right = _rand(2, 96, 5, 11, seed=1), _rand(2, 96, 5, 11, seed=2)
    _close(ix.gwc_volume(left, right, 48), ix.build_gwc_volume_fast(left, right, 48, 8, channels_last=True), 1e-3)
