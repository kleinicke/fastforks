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
from fastforks.volume import _nnunet, _resample
from fastforks.volume.models.totalsegmentator import model as ts

if TYPE_CHECKING:
    from pathlib import Path

_PATCH = (16, 16, 16)
_DNA = "dynamic_network_architectures.architectures.unet"


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
