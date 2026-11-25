"""Volume segmentation (`fastforks.volume`): the nnU-Net v2 engine on a tiny random network, resampling, label maps."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fastforks import volume
from fastforks._core.errors import InputError, OptionsError, WeightsError
from fastforks.volume import _nnunet, _resample
from fastforks.volume._api import _volume
from fastforks.volume.models.nninteractive import model as nni
from fastforks.volume.models.totalsegmentator import model as ts

if TYPE_CHECKING:
    from pathlib import Path

_PATCH = (16, 16, 16)
_DNA = "dynamic_network_architectures.architectures.unet"


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cuda_ct_normalization_preserves_rounding(dtype: torch.dtype) -> None:
    from fastforks.kernels import vol_norm  # noqa: PLC0415 - CUDA kernel only

    x = torch.linspace(-1500, 2100, 9 * 15 * 21, device="cuda").reshape(9, 15, 21)[:, ::2, 1::2].transpose(0, 2)
    expected = x.clamp(-741.25, 1121.125).sub_(41.725).div_(93.37).to(dtype)
    actual = torch.empty_like(x, dtype=dtype)
    vol_norm.normalize_ct_(actual, x, (-741.25, 1121.125), mean=41.725, std=93.37)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
def test_cuda_label_merge_maps_and_preserves_background() -> None:
    from fastforks.kernels import vol_norm  # noqa: PLC0415 - CUDA kernel only

    source = torch.arange(8 * 9 * 10, device="cuda").remainder_(4).to(torch.uint8).reshape(8, 9, 10).transpose(0, 2)
    lut = torch.tensor([0, 11, 0, 29], device="cuda", dtype=torch.uint8)
    expected = lut[source.long()]
    first = vol_norm.merge_labels(source, lut)
    torch.testing.assert_close(first, expected, atol=0, rtol=0)
    previous = torch.full_like(source, 6)
    merged = vol_norm.merge_labels(source, lut, previous)
    assert merged is previous
    torch.testing.assert_close(merged, torch.where(expected != 0, expected, 6), atol=0, rtol=0)


def test_encoder_example_inputs_are_3d_volumes() -> None:
    """The export input has the encoder's channels and all three spatial dimensions."""
    from types import SimpleNamespace  # noqa: PLC0415

    encoder = SimpleNamespace(
        channels=2,
        device=torch.device("cpu"),
        _device=torch.device("cpu"),
        max_size=None,
        _runner=SimpleNamespace(multiple=8),
    )
    (x,) = volume.VolumeEncoder.example_inputs(encoder, 12, 20, batch=3)
    assert x.shape == (3, 2, 16, 24, 24)
    assert x.dtype == torch.float32
    assert torch.count_nonzero(x[:, :, 12:]) == 0
    assert torch.count_nonzero(x[:, :, :, 20:]) == 0
    assert torch.count_nonzero(x[:, :, :, :, 20:]) == 0
    encoder.max_size = 32
    encoder._runner.arch = "vit"
    (x,) = volume.VolumeEncoder.example_inputs(encoder, 12, 20)
    assert x.shape == (1, 2, 32, 32, 32)
    encoder.variant = SimpleNamespace(ref="test-encoder")
    with pytest.raises(InputError, match="up to 32"):
        volume.VolumeEncoder.example_inputs(encoder, 33, 20)


def test_interactive_example_is_a_prepared_patch() -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    model = SimpleNamespace(
        device=torch.device("cpu"), _device=torch.device("cpu"), _runner=SimpleNamespace(_dtype=torch.float16)
    )
    (x,) = volume.VolumeInteractive.example_inputs(model, 192, 192)
    assert x.shape == (1, 8, 192, 192, 192)
    assert x.dtype == torch.float16
    assert x[0, 4, 96, 96, 96] == 1
    assert x[0, 1].count_nonzero() == 0
    with pytest.raises(InputError, match="192x192x192"):
        volume.VolumeInteractive.example_inputs(model, 128, 128)


def _plans(scheme: str = "CTNormalization") -> dict:
    arch = {
        "network_class_name": f"{_DNA}.PlainConvUNet",
        "arch_kwargs": {
            "n_stages": 2,
            "features_per_stage": [4, 8],
            "kernel_sizes": [[3, 3, 3], [3, 3, 3]],
            "strides": [[1, 1, 1], [2, 2, 2]],
            "n_conv_per_stage": [1, 1],
            "n_conv_per_stage_decoder": [1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
    }
    conf = {"patch_size": list(_PATCH), "spacing": [2.0, 1.5, 1.5], "normalization_schemes": [scheme],
            "use_mask_for_norm": [False], "architecture": arch}  # fmt: skip
    props = {"0": {"mean": 50.0, "std": 200.0, "percentile_00_5": -900.0, "percentile_99_5": 1200.0}}
    return {"configurations": {"3d_fullres": conf}, "foreground_intensity_properties_per_channel": props}


_DATASET = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "liver": 1, "spleen": 2}}


def _net(seed: int = 0) -> _nnunet.UNet:
    torch.manual_seed(seed)
    net = _nnunet.UNet(_nnunet.read_plan(_plans(), _DATASET), 1).eval()
    with torch.no_grad():  # non-trivial norm affine, so the norm path matters
        for m in net.modules():
            if isinstance(m, torch.nn.InstanceNorm3d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(0, 0.3)
    return net


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny nnU-Net v2 export: `<root>/Dataset001_Tiny/nnUNetTrainer__nnUNetPlans__3d_fullres/`."""
    d = tmp_path_factory.mktemp("nnunet") / "Dataset001_Tiny" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    for k in (0, 1):
        (d / f"fold_{k}").mkdir(parents=True)
        ckpt = {"network_weights": _net(k).state_dict(), "inference_allowed_mirroring_axes": (0, 1, 2)}
        torch.save(ckpt, d / f"fold_{k}" / "checkpoint_final.pth")
    (d / "plans.json").write_text(json.dumps(_plans()), encoding="utf-8")
    (d / "dataset.json").write_text(json.dumps(_DATASET), encoding="utf-8")
    return d


def test_local_folder_compile_uses_patch_network(folder: Path) -> None:
    """The normal volume API dispatches each fold through a compiled network with unchanged labels."""
    from torch._dynamo.testing import CompileCounter  # noqa: PLC0415 - only compile tests need Dynamo

    model = volume.load(folder, mode="portable", device="cpu", mirror=False)
    ct = torch.arange(20 * 16 * 16, dtype=torch.float32).view(20, 16, 16)
    spacing = (2.0, 1.5, 1.5)
    expected = model(ct, spacing=spacing).labels
    counter = CompileCounter()
    model.compile(backend=counter, fullgraph=True)
    assert torch.equal(model(ct, spacing=spacing).labels, expected)
    assert counter.frame_count >= 1


def test_sliding_steps() -> None:
    """nnU-Net's `compute_steps_for_sliding_window`: evenly spread starts, the last window flush with the end."""
    assert _nnunet.sliding_steps((200, 128, 129), (128, 128, 128), 0.5) == [[0, 36, 72], [0], [0, 1]]
    assert _nnunet.sliding_steps((300,), (128,), 0.8) == [[0, 86, 172]]


@pytest.mark.parametrize("depth", [2, 3, 5])
def test_conv_slabs_match_one_convolution(depth: int) -> None:
    """`_conv_slabs_`: the same-width convolution written over its input, slab by slab (any last-slab size)."""
    torch.manual_seed(0)
    conv = torch.nn.Conv3d(4, 4, 3, padding=1)
    y = torch.randn(2, 4, 11, 6, 7)
    expected = F.conv3d(y, conv.weight, conv.bias, 1, 1)
    with torch.no_grad():
        out = _nnunet._conv_slabs_(conv, y, conv.bias, depth)
    assert out is y
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    assert _nnunet.slab_depth(conv, y) == 0  # CPU: one call


def test_norm_act_matches_instance_norm() -> None:
    """The per-channel scale / shift block = `leaky_relu(InstanceNorm3d(conv(x)))`, in both memory formats."""
    block = _net().encoder.stages[0][0].convs[0]
    x = torch.randn(2, 1, 9, 10, 11) * 3 + 2
    with torch.no_grad():
        want = F.leaky_relu(block.norm(block.conv(x)), block.slope)
        got = block(x)
        cl = block(x.contiguous(memory_format=torch.channels_last_3d))
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(cl, want, rtol=1e-4, atol=1e-5)


def test_sliding_window_slabs_equal_full_logits() -> None:
    """Labels of the slab sweep (no full logits held) = argmax of the full fold-averaged logits; one window = a net."""
    plan = _nnunet.read_plan(_plans(), _DATASET)
    pred = _nnunet.Predictor(plan, [_net(), _net(1)], step=0.5, batch=3, amp=None)
    x = torch.randn(1, 37, 21, 16)
    with torch.no_grad():
        full = pred._accumulate(x, full=True)
        labels = pred._accumulate(x, full=False)
        one = pred._accumulate(x[:, :16, :16], full=True)
        direct = (pred.nets[0](x[None, :, :16, :16])[0] + pred.nets[1](x[None, :, :16, :16])[0]) / 2
    assert torch.equal(labels, full.argmax(0).to(labels.dtype))
    inner = (slice(None), *(slice(3, -3),) * 3)  # weights cancel (float16 logits); not near the corners, whose
    # float16 weights are near the smallest subnormal (nnU-Net's rounding, kept)
    torch.testing.assert_close(one.float()[inner], direct[inner], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("heads", [19, 260])
def test_resampled_labels_keep_first_tie_and_compact_dtype(heads: int) -> None:
    plan = _nnunet.read_plan(_plans(), _DATASET)
    plan = replace(plan, labels={i: str(i) for i in range(1, heads)})
    predictor = _nnunet.Predictor(plan, [])
    logits = torch.zeros(heads, 2, 3, 4)
    logits[3, 0] = logits[12, 0] = 5
    logits[-1, 1] = 7
    shape = (3, 6, 8)
    source, target = (1.5, 1.5, 1.5), (1.0, 1.0, 1.0)
    fn, kw = plan.resample_logits
    expected = _resample.resample(logits, shape, source, target, fn, kw).argmax(0)
    got = predictor._decide_resampled(logits.half(), shape, source, target)
    assert torch.equal(got.long(), expected)
    assert got.dtype == (torch.uint8 if heads < 256 else torch.int16)


def test_nnunet_folder_with_safetensors(folder: Path, tmp_path: Path) -> None:
    from safetensors.torch import save_file  # noqa: PLC0415 - only the safetensors fixture needs the writer

    path = tmp_path / "Dataset002_Tiny" / folder.name
    fold = path / "fold_0"
    fold.mkdir(parents=True)
    for name in ("plans.json", "dataset.json"):
        (path / name).write_bytes((folder / name).read_bytes())
    state = dict(_net().state_dict()) | {"__nnunet_checkpoint_state__": torch.zeros(1, dtype=torch.uint8)}
    save_file(state, fold / "checkpoint_final.safetensors", metadata={"inference_allowed_mirroring_axes": "[0,1,2]"})
    model = volume.load(path, mode="portable", device="cpu", checkpoint="checkpoint_final.safetensors")
    reference = volume.load(folder, mode="portable", device="cpu", folds=(0,))
    ct = torch.arange(16**3, dtype=torch.float32).view(16, 16, 16)
    opts = {"spacing": (2.0, 1.5, 1.5)}
    assert model._runner.parts[0].mirror_axes == (0, 1, 2)
    assert torch.equal(model(ct, **opts).labels, reference(ct, **opts).labels)


@pytest.mark.gpu
@pytest.mark.parametrize("patch", [(1, 37, 103), (16, 16, 16), (31, 48, 19), (192, 192, 192)])
def test_cuda_gaussian_matches_cpu_half_weights(patch: tuple[int, int, int]) -> None:
    """Compare complete weights, including zeros rounded away and replaced by the smallest positive weight."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    expected = _nnunet.gaussian(patch, torch.device("cpu"))
    actual = _nnunet.gaussian(patch, torch.device("cuda"), cuda_kernels=True)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.gpu
@pytest.mark.parametrize("channels", [19, 118])
def test_fused_window_accumulation_preserves_rounding(channels: int) -> None:
    from fastforks.kernels import vol_norm  # noqa: PLC0415 - CUDA kernel only

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(17)
    mem = {"dtype": torch.float16, "device": "cuda"}
    logits = torch.randn(8, 9, 13, channels, **mem).permute(3, 0, 1, 2)
    out = torch.randn(9, 12, 15, channels, **mem).permute(3, 0, 1, 2)
    reference = out.clone()
    weight = torch.rand(8, 9, 13, **mem) / 4
    window = (slice(None), slice(1, 9), slice(2, 11), slice(1, 14))
    reference[window].add_(logits * weight)
    vol_norm.accumulate_(out[window], logits, weight)
    assert torch.equal(out, reference)


@pytest.mark.gpu
def test_fused_volume_labels_match_trilinear_across_slabs() -> None:
    from fastforks.kernels import vol_norm  # noqa: PLC0415 - CUDA kernel only

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(18)
    scores = torch.randn(11, 13, 17, 118, dtype=torch.float16, device="cuda").permute(3, 0, 1, 2)
    scores[117].copy_(scores[0])
    shape = (23, 29, 27)
    expected = F.interpolate(scores[None].float(), size=shape, mode="trilinear", align_corners=False)[0].argmax(0)
    actual = torch.empty(shape, dtype=torch.uint8, device="cuda")
    vol_norm.resize_labels_(actual[:14], scores[:, :7], None, first_src_z=0, first_out_z=0, src_depth=11, dst_depth=23)
    previous = scores[:, 6].permute(1, 2, 0).contiguous()
    vol_norm.resize_labels_(
        actual[14:], scores[:, 7:], previous, first_src_z=7, first_out_z=14, src_depth=11, dst_depth=23
    )
    assert torch.equal(actual.long(), expected)


@pytest.mark.gpu
def test_tiled_volume_head_matches_full_projection(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two windows per batch, overlapping depth spans, and a partial final projection tile."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = volume.load(folder, mode="kernels", device="cuda", folds=(0,), mirror=False)
    ct = np.random.default_rng(17).normal(0, 400, (22, 21, 24)).astype(np.float32)
    spacing = (2.0, 1.5, 1.5)
    expected = model(ct, spacing=spacing).labels
    monkeypatch.setattr(_nnunet, "_HEAD_TILE_BYTES", 3 * 3 * 16 * 16 * 2)
    assert torch.equal(model(ct, spacing=spacing).labels, expected)


@pytest.mark.gpu
def test_tiled_score_ring_preserves_resampled_labels(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete and normalize wrapped tiles, including interpolation across a reused score plane."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = volume.load(folder, mode="kernels", device="cuda", folds=(0,), mirror=False)
    part = model._runner.parts[0]
    x = torch.rand(1, 37, 16, 16, dtype=torch.float16, device="cuda")
    keep = (slice(1, 36), slice(0, 16), slice(0, 16))
    with torch.inference_mode():
        expected = part._stream_resampled(x, keep, (53, 23, 19))
        monkeypatch.setattr(_nnunet, "_HEAD_TILE_BYTES", 3 * 3 * 16 * 16 * 2)
        assert torch.equal(part._stream_resampled(x, keep, (53, 23, 19)), expected)


@pytest.mark.gpu
def test_tiled_decoder_preserves_consumed_skip_halo(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiled convolution reads the original skip at each boundary and handles its final partial tile."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = volume.load(folder, mode="kernels", device="cuda", folds=(0,), mirror=False)
    ct = np.random.default_rng(29).normal(0, 400, (19, 16, 16)).astype(np.float32)
    spacing = (2.0, 1.5, 1.5)
    expected = model(ct, spacing=spacing).labels
    monkeypatch.setattr(_nnunet, "_DECODER_TILE_BYTES", 3 * 2 * 4 * 16 * 16 * 2)
    assert torch.equal(model(ct, spacing=spacing).labels, expected)


def test_zoom_matches_scipy() -> None:
    """TotalSegmentator's `change_spacing` zooms (scipy `zoom`, order 1 truncated to int32 / order 0)."""
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(0)
    img = rng.normal(0, 300, (11, 14, 9)).astype(np.float32)
    for shape in [(6, 20, 9), (22, 7, 13)]:
        zoom = [m / n for m, n in zip(shape, img.shape, strict=True)]
        want = ndimage.zoom(img, zoom, order=1).astype(np.int32)
        got = ts.zoom_linear(torch.from_numpy(img), shape).numpy()
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 1  # float64 vs scipy's spline coefficients: rare off-by-one truncation
        assert (got != want).mean() < 0.01
        lab = rng.integers(0, 5, img.shape).astype(np.uint8)
        assert np.array_equal(ts.zoom_nearest(torch.from_numpy(lab), shape).numpy(), ndimage.zoom(lab, zoom, order=0))


def test_nnunet_stream_requires_cuda(folder: Path) -> None:
    with pytest.raises(OptionsError, match="stream=True requires CUDA"):
        volume.load(folder, mode="portable", device="cpu", stream=True)


@pytest.mark.gpu
def test_nnunet_streamed_folder_keeps_folds_on_host(folder: Path) -> None:
    resident = volume.load(folder, mode="portable", device="cuda", mirror=False)
    streamed = volume.load(folder, mode="portable", device="cuda", mirror=False, stream=True)
    assert all(next(net.parameters()).device.type == "cpu" for net in streamed.networks())
    ct = np.random.default_rng(2).normal(0, 400, (18, 18, 18)).astype(np.float32)
    expected = resident(ct, spacing=(2.0, 1.5, 1.5)).labels
    for _ in range(2):
        assert torch.equal(streamed(ct, spacing=(2.0, 1.5, 1.5)).labels, expected)
    assert all(next(net.parameters()).device.type == "cpu" for net in streamed.networks())


def test_load_nnunet_folder_labels(folder: Path) -> None:
    """`load(folder)`: fold ensemble + mirroring on the input grid, nnU-Net's labels and names, input checks."""
    model = volume.load(folder, device="cpu")
    assert model.class_names == {1: "liver", 2: "spleen"}
    assert model.channels == 1
    assert len(model.networks()) == 2
    rng = np.random.default_rng(1)
    ct = rng.normal(0, 400, (20, 30, 25)).astype(np.float32)
    ct[:3] = 0  # outside the nonzero box: background
    out = model(ct, spacing=(2.5, 1.2, 1.5))
    assert out.labels.shape == ct.shape
    assert out.labels.dtype == torch.uint8
    assert int(out.labels[:3].max()) == 0
    assert set(out.labels.unique().tolist()) <= {0, 1, 2}
    # the same pipeline by hand: the Predictor with both folds and all mirror axes
    plan = _nnunet.read_plan(_plans(), _DATASET)
    nets = [n.float() for n in model.networks()]
    pred = _nnunet.Predictor(plan, nets, step=0.5, mirror_axes=(0, 1, 2), batch=2, amp=None)
    with torch.no_grad():
        want = pred(torch.from_numpy(ct)[None], (2.5, 1.2, 1.5))
    assert (out.labels != want).float().mean() < 1e-3
    with pytest.raises(InputError, match="spacing"):
        model(ct, spacing=(1.0, 0.0, 1.0))
    with pytest.raises(InputError, match="channel"):
        model(np.stack([ct, ct]), spacing=(1.0, 1.0, 1.0))


def test_call_meta_and_save(folder: Path, tmp_path: Path) -> None:
    """`meta=` gives the spacing and the grid `save` writes NIfTI on; one of `spacing` / `meta` is required."""
    nib = pytest.importorskip("nibabel")
    model = volume.load(folder, device="cpu", mirror=False, folds=[0])
    ct = np.random.default_rng(3).normal(0, 400, (12, 14, 10)).astype(np.float32)
    affine = np.diag([1.5, 1.2, 2.5, 1.0])
    nib.save(nib.Nifti1Image(ct.transpose(2, 1, 0), affine), tmp_path / "ct.nii.gz")
    data, meta = volume.read_nifti(tmp_path / "ct.nii.gz")
    out = model(data, meta=meta)
    assert torch.equal(out.labels, model(data, spacing=meta.spacing).labels)
    out.save(tmp_path / "seg.nii.gz")
    back, _ = volume.read_nifti(tmp_path / "seg.nii.gz")
    assert np.array_equal(back, out.labels.numpy())
    out.save(tmp_path / "seg.npy")
    assert np.array_equal(np.load(tmp_path / "seg.npy"), out.labels.numpy())
    with pytest.raises(InputError, match="meta="):
        model(data, spacing=meta.spacing).save(tmp_path / "x.nii.gz")
    with pytest.raises(InputError, match="one of the two"):
        model(data)
    with pytest.raises(InputError, match="one of the two"):
        model(data, spacing=meta.spacing, meta=meta)
    with pytest.raises(InputError, match="grid of meta"):
        model(data[1:], meta=meta)


def test_label_map_merge(folder: Path) -> None:
    """Parts merge into the card's class list (later parts win); a class missing from the list is an error."""
    model = volume.load(folder, device="cpu", mirror=False, folds=[0])
    runner = model._runner
    runner._luts = [torch.tensor([0, 5, 0], dtype=torch.int16), torch.tensor([0, 0, 7], dtype=torch.int16)]
    runner.parts = [runner.parts[0], runner.parts[0]]
    ct = np.random.default_rng(2).normal(0, 400, (18, 18, 18)).astype(np.float32)
    with torch.no_grad():
        raw = runner.parts[0](torch.from_numpy(ct)[None], (2.0, 1.5, 1.5)).long()
    labels = runner(torch.from_numpy(ct)[None], (2.0, 1.5, 1.5))["labels"].long()
    want = torch.where(raw == 2, 7, torch.where(raw == 1, 5, 0))
    assert torch.equal(labels, want)


def test_regions_and_ignore_label() -> None:
    """Region-based training: one sigmoid head per region, `logit > 0` painted in class order; the ignore label is
    no output."""
    ds = dict(_DATASET, labels={"background": 0, "whole": [1, 2], "core": 2, "ignore": 3}, regions_class_order=[1, 2])
    plan = _nnunet.read_plan(_plans(), ds)
    assert plan.regions == (1, 2)
    assert plan.n_outputs == 2
    assert dict(plan.labels) == {1: "whole", 2: "core"}
    logits = torch.tensor([[1.0, 1.0, -1.0, -2.0], [-1.0, 3.0, 2.0, -1.0]])
    assert _nnunet.Predictor(plan, []).decide(logits).tolist() == [1, 2, 2, 0]
    assert _nnunet.read_plan(_plans(), dict(_DATASET, labels={**_DATASET["labels"], "ignore": 3})).n_outputs == 3
    with pytest.raises(WeightsError, match="regions_class_order"):
        _nnunet.read_plan(_plans(), dict(_DATASET, labels={"background": 0, "liver": [1, 2]}))


def test_resample_matches_scipy() -> None:
    """nnU-Net's default resampling = skimage `resize` (scipy `zoom`, grid mode, clipped), separate z for anisotropy."""
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(3)
    x = rng.normal(0, 300, (2, 7, 12, 10)).astype(np.float32)

    def resize(a: np.ndarray, shape: tuple[int, ...], order: int) -> np.ndarray:
        zoom = [m / n for m, n in zip(shape, a.shape, strict=True)]
        out = ndimage.zoom(a.astype(np.float64), zoom, order=order, mode="nearest", grid_mode=True)
        return np.clip(out, a.min(), a.max())

    for order in (1, 3):
        shape = (9, 17, 8)
        got = _resample.resample(torch.from_numpy(x), shape, (1.0, 1.0, 1.0), (0.8, 0.7, 1.25), kwargs={"order": order})
        want = np.stack([resize(c, shape, order) for c in x])
        np.testing.assert_allclose(got.numpy(), want, rtol=0, atol=1e-3 * 300)
    # anisotropic (z 4x coarser): in-plane per slice, then nearest along z
    shape = (14, 9, 15)
    got = _resample.resample(torch.from_numpy(x), shape, (4.0, 1.0, 1.0), (2.0, 1.33, 0.67), kwargs={"order": 3})
    plane = np.stack([np.stack([resize(s, shape[1:], 3) for s in c]) for c in x])
    idx = np.clip(np.floor((np.arange(14) + 0.5) * 7 / 14 - 0.5 + 0.5), 0, 6).astype(int)
    np.testing.assert_allclose(got.numpy(), plane[:, idx], rtol=0, atol=1e-3 * 300)
    mask = rng.random((9, 10, 11)) > 0.3
    mask[3:6, 3:6, 3:6] = True
    mask[4, 4, 4] = False  # a hole
    assert np.array_equal(_resample.fill_holes(torch.from_numpy(mask)).numpy(), ndimage.binary_fill_holes(mask))


def _resenc_plans() -> dict:
    plans = _plans()
    arch = plans["configurations"]["3d_fullres"]["architecture"]
    arch["network_class_name"] = f"{_DNA}.ResidualEncoderUNet"
    a = arch["arch_kwargs"]
    a["n_blocks_per_stage"] = a.pop("n_conv_per_stage")
    a["block"] = "dynamic_network_architectures.building_blocks.residual.BasicBlockD"
    return plans


def test_residual_encoder_unet() -> None:
    """`ResidualEncoderUNet`: upstream's keys (stem, blocks, avg-pool + 1x1 skip) and `BasicBlockD`'s sums."""
    plan = _nnunet.read_plan(_resenc_plans(), _DATASET)
    assert plan.network == "resenc"
    torch.manual_seed(0)
    net = _nnunet.UNet(plan, 1).eval()
    keys = set(net.state_dict())
    assert {"encoder.stem.convs.0.conv.weight", "encoder.stages.1.blocks.0.conv1.conv.weight",
            "encoder.stages.1.blocks.0.conv2.norm.weight", "encoder.stages.1.blocks.0.skip.1.conv.weight",
            "decoder.seg_layers.0.weight"} <= keys  # fmt: skip
    assert "encoder.stages.0.blocks.0.skip.0.conv.weight" not in keys  # stride 1, same width: identity
    block = net.encoder.stages[1].blocks[0]
    assert isinstance(block, _nnunet._BasicBlock)
    x = torch.randn(1, 4, 8, 8, 8)
    with torch.no_grad():
        c1 = F.leaky_relu(block.conv1.norm(block.conv1.conv(x)), 0.01)
        proj = block.skip[1]
        assert isinstance(proj, _nnunet._ConvNormAct)
        res = proj.norm(proj.conv(F.avg_pool3d(x, 2, 2)))
        want = F.leaky_relu(block.conv2.norm(block.conv2.conv(c1)) + res, 0.01)
        torch.testing.assert_close(block(x), want, rtol=1e-4, atol=1e-5)
        assert net(torch.randn(1, 1, *_PATCH)).shape == (1, 3, *_PATCH)


def _plans_2d() -> dict:
    plans = _plans()
    conf = plans["configurations"]["3d_fullres"]
    a = conf["architecture"]["arch_kwargs"]
    a["kernel_sizes"], a["strides"] = [[3, 3], [3, 3]], [[1, 1], [2, 2]]
    a["norm_op"] = "torch.nn.modules.instancenorm.InstanceNorm2d"
    conf2 = dict(conf, patch_size=[16, 16], spacing=[1.5, 1.5])
    return dict(plans, configurations={"2d": conf2})


def test_2d_configuration(tmp_path: Path) -> None:
    """2D configurations: 2D weights lifted to depth-1 3D convs, one window per slice, slices independent."""
    plan = _nnunet.read_plan(_plans_2d(), _DATASET, "2d")
    assert plan.patch_size == (1, 16, 16)
    assert plan.spacing == (None, 1.5, 1.5)
    torch.manual_seed(1)
    net = _nnunet.UNet(plan, 1).eval()
    sd2 = {k: v.squeeze(2) if v.dim() == 5 else v for k, v in net.state_dict().items()}
    assert torch.equal(_nnunet.UNet.state_dict_of(sd2, 2)["encoder.stages.0.0.convs.0.conv.weight"],
                       net.state_dict()["encoder.stages.0.0.convs.0.conv.weight"])  # fmt: skip
    pred = _nnunet.Predictor(plan, [net], mirror_axes=(1, 2), amp=None)
    x = torch.randn(1, 5, 20, 18) * 300
    with torch.no_grad():
        full = pred(x, (3.0, 1.5, 1.5))
        one = pred(x[:, 2:3], (3.0, 1.5, 1.5))
    assert full.shape == (5, 20, 18)
    assert torch.equal(full[2:3], one)
    # a bare checkpoint with the plans in its init_args
    init = {"plans": _plans_2d(), "dataset_json": _DATASET, "configuration": "2d", "fold": 0}
    torch.save({"network_weights": sd2, "init_args": init, "inference_allowed_mirroring_axes": (0, 1)},
               tmp_path / "checkpoint_final.pth")  # fmt: skip
    model = volume.load(tmp_path / "checkpoint_final.pth", device="cpu")
    assert model._runner.parts[0].mirror_axes == (1, 2)
    assert torch.equal(model(x[0].numpy(), spacing=(3.0, 1.5, 1.5)).labels, full)


def test_nifti_round_trip(tmp_path: Path) -> None:
    """`read_nifti` gives the RAS-canonical `(z, y, x)` array; `write_nifti` puts labels back on the file's grid."""
    nib = pytest.importorskip("nibabel")
    data = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    affine = np.diag([-1.5, 2.0, 3.0, 1.0])  # x flipped (LAS): canonical flips it back
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "ct.nii.gz")
    vol, meta = volume.read_nifti(tmp_path / "ct.nii.gz")
    assert vol.shape == (6, 5, 4)
    assert meta.spacing == (3.0, 2.0, 1.5)
    assert np.array_equal(vol, data[::-1].transpose(2, 1, 0).astype(np.float32))
    volume.write_nifti(tmp_path / "seg.nii.gz", (vol % 7).astype(np.uint8), meta)
    seg = nib.load(tmp_path / "seg.nii.gz")
    assert np.allclose(seg.affine, affine)
    assert np.array_equal(np.asarray(seg.dataobj), data % 7)
    with pytest.raises(InputError, match="shape"):
        volume.write_nifti(tmp_path / "bad.nii.gz", vol[:2], meta)


def test_read_dicom_matches_nifti_convention(tmp_path: Path) -> None:
    """A synthetic LPS DICOM series (files in shuffled name order) reads as the RAS-canonical HU array of read_nifti."""
    pytest.importorskip("pydicom")
    pytest.importorskip("nibabel")
    from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: PLC0415
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: PLC0415

    rng = np.random.default_rng(0)
    hu = rng.integers(-1000, 1000, (4, 5, 6)).astype(np.int16)  # [slice, row, col] in HU
    series = tmp_path / "series"
    series.mkdir()
    (series / "README.txt").write_text("not dicom", encoding="utf-8")
    order = [2, 0, 3, 1]  # file name order differs from position order
    for k in range(4):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Rows, ds.Columns, ds.SamplesPerPixel = 5, 6, 1
        ds.PhotometricInterpretation, ds.BitsAllocated, ds.BitsStored, ds.HighBit = "MONOCHROME2", 16, 16, 15
        ds.PixelRepresentation, ds.PixelSpacing = 0, [0.5, 0.75]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # standard axial LPS
        ds.ImagePositionPatient = [10.0, 20.0, 100.0 + 2.5 * k]
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
        ds.PixelData = (hu[k].astype(np.int32) + 1024).astype(np.uint16).tobytes()
        ds.save_as(series / f"f{order[k]}.dcm", enforce_file_format=True)
    vol, meta = volume.read_dicom(series)
    assert meta.spacing == (2.5, 0.5, 0.75)
    assert meta.shape == (4, 5, 6)
    # LPS -> RAS: rows (y) and columns (x) are flipped, the slice axis (z, superior) keeps its order
    assert np.array_equal(vol, hu[:, ::-1, ::-1].astype(np.float32))
    assert np.allclose(meta.affine @ [0, 0, 0, 1], [-10.0, -20.0, 100.0, 1.0])
    with pytest.raises(InputError, match="no DICOM"):
        volume.read_dicom(tmp_path / "series" / "README.txt")


def test_unsupported_networks() -> None:
    """Other architectures, norms and labels fail at load with a clear message."""
    plans = _plans()
    plans["configurations"]["3d_fullres"]["architecture"]["network_class_name"] = f"{_DNA}.MyUNet"
    with pytest.raises(WeightsError, match="only"):
        _nnunet.read_plan(plans, _DATASET)
    plans = _plans()
    plans["configurations"]["3d_cascade_fullres"] = {"inherits_from": "3d_fullres", "previous_stage": "3d_lowres"}
    with pytest.raises(WeightsError, match="cascade"):
        _nnunet.read_plan(plans, _DATASET, "3d_cascade_fullres")
    plans = _plans()
    plans["configurations"]["3d_fullres"]["resampling_fn_data"] = "my_resampling"
    with pytest.raises(WeightsError, match="resampling"):
        _nnunet.read_plan(plans, _DATASET)
    with pytest.raises(WeightsError, match="normalisation"):
        _nnunet.read_plan(_plans("MyNormalization"), _DATASET)
    with pytest.raises(WeightsError, match="configuration"):
        _nnunet.read_plan(_plans(), _DATASET, "2d")


def test_nninteractive_cover_boxes() -> None:
    """The box prompt's cover (summed-area counts) covers every object voxel; box sums match brute force."""
    rng = np.random.default_rng(0)
    for _ in range(6):
        shape = tuple(int(n) for n in rng.integers(20, 70, 3))
        m = torch.zeros(shape, dtype=torch.bool)
        for _ in range(int(rng.integers(1, 4))):
            c, r = rng.integers(0, shape), rng.integers(2, 25, 3)
            m[tuple(slice(max(0, a - b), a + b) for a, b in zip(c.tolist(), r.tolist(), strict=True))] = True
        lo = torch.from_numpy(rng.integers(0, np.array(shape) // 2, (5, 3)))
        hi = lo + torch.from_numpy(rng.integers(1, np.array(shape) // 2, (5, 3)))
        sums = nni._box_sums(nni._sat(m), lo, hi)
        want = [
            int(m[a[0] : b[0], a[1] : b[1], a[2] : b[2]].sum()) for a, b in zip(lo.tolist(), hi.tolist(), strict=True)
        ]
        assert sums.tolist() == want
        covered = torch.zeros_like(m)
        for box in nni.cover_boxes(m, (24, 24, 24), np.random.RandomState(0)):
            covered[tuple(slice(max(0, a), b) for a, b in box)] = True
        assert bool(covered[m].all())


@pytest.mark.parametrize("bad", ["ct.nii.gz", [[1.0]]])
def test_volume_path_input_names_read_nifti(bad: object) -> None:
    with pytest.raises(InputError, match="numpy array or torch tensor") as e:
        _volume(bad, 1, "x")
    assert ("read_nifti" in str(e.value)) == isinstance(bad, str)


def test_bad_spacing_nan_and_dicom_dir_give_clear_errors(folder: Path, tmp_path: Path) -> None:
    """Spacing in metres / 10x too large and NaN input are InputErrors; DICOM paths point at read_dicom; short reprs."""
    pytest.importorskip("nibabel")
    model = volume.load(folder, device="cpu", mirror=False, folds=[0])
    ct = np.random.default_rng(3).normal(0, 400, (12, 14, 10)).astype(np.float32)
    with pytest.raises(InputError, match="spacing"):
        model(ct, spacing=(0.0015, 0.0012, 0.0025))
    with pytest.raises(InputError, match="spacing"):
        model(ct, spacing=(1e5, 1e5, 1e5))
    bad = ct.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(InputError, match="NaN"):
        model(bad, spacing=(1.0, 1.0, 1.0))
    with pytest.raises(InputError, match="read_dicom"):
        volume.read_nifti(tmp_path)
    with pytest.raises(InputError, match="read_dicom"):
        volume.read_nifti(tmp_path / "a.dcm")
    _, meta = volume.read_nifti(_write_nifti(tmp_path, ct))
    assert len(repr(meta)) < 300
    assert len(repr(model(ct, spacing=(1.0, 1.0, 1.0)))) < 500


def _write_nifti(tmp_path: Path, ct: np.ndarray) -> Path:
    import nibabel as nib  # noqa: PLC0415

    nib.save(nib.Nifti1Image(ct.transpose(2, 1, 0), np.eye(4)), tmp_path / "w.nii.gz")
    return tmp_path / "w.nii.gz"


def test_session_add_point_keyword_and_its_old_name(monkeypatch) -> None:
    from fastforks.volume._api import VolumeSession  # noqa: PLC0415 - next to its test

    s = object.__new__(VolumeSession)
    s.shape = (4, 5, 6)
    calls = []
    monkeypatch.setattr(s, "_run", lambda *a: calls.append(a), raising=False)
    s.add_point((1, 2, 3))
    s.add_point(point=(1, 2, 3), positive=False)
    with pytest.warns(FutureWarning, match=r"add_point\(zyx=\) is renamed to add_point\(point=\)"):
        s.add_point(zyx=(1, 2, 3))
    p = (1.0, 2.0, 3.0)
    assert calls == [("point", p, True), ("point", p, False), ("point", p, True)]
    with pytest.raises(InputError, match="not also its old name"):
        s.add_point((1, 2, 3), zyx=(1, 2, 3))
    with pytest.raises(InputError, match="needs point="):
        s.add_point()
