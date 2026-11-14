"""Volume segmentation (`fastforks.volume`): the nnU-Net v2 engine on a tiny random network, resampling, label maps."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fastforks import volume
from fastforks._core.errors import InputError, WeightsError
from fastforks.volume import _nnunet
from fastforks.volume.models.totalsegmentator import model as ts

if TYPE_CHECKING:
    from pathlib import Path

_PATCH = (16, 16, 16)


def _plans(scheme: str = "CTNormalization") -> dict:
    arch = {
        "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
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


def _net(seed: int = 0) -> _nnunet.PlainConvUNet:
    torch.manual_seed(seed)
    net = _nnunet.PlainConvUNet(_nnunet.read_plan(_plans(), _DATASET), 1).eval()
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
    (d / "plans.json").write_text(json.dumps(_plans()))
    (d / "dataset.json").write_text(json.dumps(_DATASET))
    return d


def test_sliding_steps() -> None:
    """nnU-Net's `compute_steps_for_sliding_window`: evenly spread starts, the last window flush with the end."""
    assert _nnunet.sliding_steps((200, 128, 129), (128, 128, 128), 0.5) == [[0, 36, 72], [0], [0, 1]]
    assert _nnunet.sliding_steps((300,), (128,), 0.8) == [[0, 86, 172]]


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
    """Labels from the slab sweep (no full logits held) = argmax of the full averaged logits; one window = the net."""
    plan = _nnunet.read_plan(_plans(), _DATASET)
    pred = _nnunet.Predictor(plan, [_net()], step=0.5, batch=3, amp=None)
    x = torch.randn(1, 37, 21, 16)
    with torch.no_grad():
        full = pred._accumulate(x, full=True)
        labels = pred._accumulate(x, full=False)
        one = pred._accumulate(x[:, :16, :16], full=True)
        net = _nnunet.PlainConvUNet(plan, 1).eval()
        net.load_state_dict(pred.nets[0].state_dict())
        direct = net(x[None, :, :16, :16])[0]
    assert torch.equal(labels, full.argmax(0).to(labels.dtype))
    torch.testing.assert_close(one, direct, rtol=2e-3, atol=2e-3)  # weights cancel; float16 window logits


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


def test_load_nnunet_labels(folder: Path) -> None:
    """`load_nnunet`: fold ensemble + mirroring on the input grid, nnU-Net's labels and names, input checks."""
    model = volume.load_nnunet(folder, device="cpu")
    assert model.class_names == {1: "liver", 2: "spleen"}
    assert model.channels == 1
    assert len(model.networks()) == 2
    rng = np.random.default_rng(1)
    ct = rng.normal(0, 400, (20, 30, 25)).astype(np.float32)
    ct[:3] = 0  # outside the nonzero box: background
    out = model(ct, (2.5, 1.2, 1.5))
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
        model(ct, (1.0, 0.0, 1.0))
    with pytest.raises(InputError, match="channel"):
        model(np.stack([ct, ct]), (1.0, 1.0, 1.0))


def test_label_map_merge(folder: Path) -> None:
    """Parts merge into the card's class list (later parts win); a class missing from the list is an error."""
    model = volume.load_nnunet(folder, device="cpu", mirror=False, folds=[0])
    runner = model._runner
    runner._luts = [torch.tensor([0, 5, 0], dtype=torch.int16), torch.tensor([0, 0, 7], dtype=torch.int16)]
    runner.parts = [runner.parts[0], runner.parts[0]]
    ct = np.random.default_rng(2).normal(0, 400, (18, 18, 18)).astype(np.float32)
    with torch.no_grad():
        raw = runner.parts[0](torch.from_numpy(ct)[None], (2.0, 1.5, 1.5)).long()
    labels = runner(torch.from_numpy(ct)[None], (2.0, 1.5, 1.5))["labels"].long()
    want = torch.where(raw == 2, 7, torch.where(raw == 1, 5, 0))
    assert torch.equal(labels, want)
    with pytest.raises(WeightsError, match="region-based"):
        _nnunet.read_plan(_plans(), dict(_DATASET, labels={"background": 0, "liver": [1, 2]}))


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


def test_unsupported_networks() -> None:
    """Other architectures, norms and labels fail at load with a clear message."""
    plans = _plans()
    plans["configurations"]["3d_fullres"]["architecture"]["network_class_name"] = "ResidualEncoderUNet"
    with pytest.raises(WeightsError, match="only"):
        _nnunet.read_plan(plans, _DATASET)
    with pytest.raises(WeightsError, match="normalisation"):
        _nnunet.read_plan(_plans("MyNormalization"), _DATASET)
    with pytest.raises(WeightsError, match="configuration"):
        _nnunet.read_plan(_plans(), _DATASET, "2d")
