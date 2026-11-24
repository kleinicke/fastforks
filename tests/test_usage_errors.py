"""Usage errors through every task's public API: the CPU usage probe of 2026-11-15 as tests.

The load-time checks (unknown option, bad `mode` / `dtype`, model-id typo, `mode="kernels"` on the CPU, missing
weights in offline mode) run with the empty caches of `conftest._isolated_cache`: they must fail before any weights are
looked up, or they would raise a `WeightsDownloadError` instead. The input checks (wrong channels, 1-D, complex /
int64 dtype, `None`, a missing file, the task-specific ones) need a loaded model: the smallest model of the task whose
weights are already on this machine, on the CPU, skipped when none is. `test_image.py` / `test_input_fuzz.py` test the
checks themselves (mostly on stereo); this file tests that each task's `load` and `__call__` route through them.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from fastforks import InputError, ModelNotFoundError, OptionsError, WeightsDownloadError
from fastforks._core import files, registry
from fastforks._core.errors import KernelUnavailableError, UnknownOptionError, WeightsError

# task -> (smallest model by weight size, a typo of it): the load-time checks need no weights
LOAD = {
    "calibration": ("geocalib-pinhole", "geocalibpinhole"),
    "classification": ("starnet_s1.in1k", "starnet_s1in1k"),
    "depth": ("depth-anything-v2-s", "depthanything-v2-s"),
    "detection": ("rtmdet-nano-person", "rtmdetnano-person"),
    "edges": ("teed", "tede"),
    "features": ("pe-core-t16-384", "pecore-t16-384"),
    "flow": ("raft-small", "raftsmall"),
    "interpolation": ("rife-4.25-lite", "rife4.25-lite"),
    "matching": ("aliked-lightglue", "alikedlightglue"),
    "matting": ("u2netp", "u2-netp"),
    "multiview": ("mvsplat-re10k", "mvsplatre10k"),
    "ocr": ("ppocrv5-mobile", "ppocrv5mobile"),
    "pose": ("yunet-s", "yunets"),
    "promptable_segmentation": ("mobile-sam", "mobilesam"),
    "restoration": ("realesr-general-x4v3", "realesrgeneral-x4v3"),
    "retrieval": ("boq-dinov2", "boqdinov2"),
    "segmentation": ("segnext-mscan-t-ade20k", "segnextmscan-t-ade20k"),
    "stereo": ("hitnet-eth3d", "hitneteth3d"),
    "tracking": ("locotrack-b", "locotrackb"),
    "volume": ("nnfoundation-cnn", "nnfoundationcnn"),
}
TASKS = sorted(LOAD)
# task -> models to load for the input checks, smallest first; the first whose weights are on this machine is used
# (plain call signatures: no prompt-depth, interactive or text-prompted variants)
LOADED = {
    "calibration": ("geocalib-pinhole",),
    "classification": ("starnet_s1.in1k", "fastvit_t8.apple_dist_in1k", "efficientnet_b0.tv_in1k"),
    "depth": ("depth-anything-v2-s", "video-depth-anything-s"),
    "detection": ("rtmdet-nano-person", "dfine-n-coco", "yolox-tiny"),
    "edges": ("teed", "pidinet"),
    "features": ("pe-core-t16-384", "dinov3-vits16", "dinov2-vits14", "dinov3-vitb16"),
    "flow": ("raft-small", "gmflow-scale1-mixdata", "raft-large"),
    "interpolation": ("rife-4.25-lite", "rife-4.26"),
    "matching": ("aliked-lightglue", "dedode-v2-b"),
    "matting": ("u2netp", "u2net"),
    "multiview": ("mvsplat-re10k",),
    "ocr": ("ppocrv5-mobile",),
    "pose": ("yunet-s", "yunet-n", "rtmo-t"),
    "promptable_segmentation": ("mobile-sam",),
    "restoration": ("realesr-general-x4v3", "span-x2"),
    "retrieval": ("boq-dinov2",),
    "segmentation": ("segnext-mscan-t-ade20k", "segnext-mscan-s-ade20k"),
    "stereo": ("hitnet-eth3d", "hitnet-middlebury-d400"),
    "tracking": ("locotrack-b", "tapir"),
    "volume": ("nnfoundation-cnn", "nnfoundation-vit"),
}
PAIRS = ("flow", "interpolation", "matching", "stereo")  # two image inputs
SAME_SIZE = ("flow", "interpolation", "stereo")  # matching takes two images of any sizes
WITH_OPTIONS = [t for t in TASKS if registry.resolve(t, LOAD[t][0])[0].options]

RNG = np.random.default_rng(0)
IMG = RNG.integers(0, 256, (64, 96, 3), dtype=np.uint8)
NARROW = IMG[:, :-8]


def api(task: str) -> Any:
    return importlib.import_module(f"fastforks.{task}")


# --- load: checked before any weights are looked up ------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_unknown_option_lists_the_valid_ones(task):
    ref = LOAD[task][0]
    card, _ = registry.resolve(task, ref)
    valid = ", ".join(card.options) or "none"
    with pytest.raises(UnknownOptionError, match=rf"'bogus_option' for {re.escape(ref)}@1; valid options: {valid} "):
        api(task).load(ref, device="cpu", bogus_option=1)


@pytest.mark.parametrize("task", WITH_OPTIONS)
def test_option_typo_suggests_the_option(task):
    ref = LOAD[task][0]
    name = next(iter(registry.resolve(task, ref)[0].options))
    with pytest.raises(UnknownOptionError, match=rf"'{name[:-1]}' \(did you mean '{name}'\?\)"):
        api(task).load(ref, device="cpu", **{name[:-1]: 1})


@pytest.mark.parametrize("task", TASKS)
def test_bad_mode(task):
    with pytest.raises(OptionsError, match="mode must be 'portable', 'kernels' or 'auto', got 'turbo'"):
        api(task).load(LOAD[task][0], device="cpu", mode="turbo")


@pytest.mark.parametrize("dtype", ["fp16", torch.int8], ids=["fp16", "torch.int8"])
@pytest.mark.parametrize("task", TASKS)
def test_bad_dtype(task, dtype):
    with pytest.raises(OptionsError, match=rf"dtype must be 'float32', 'float16', 'bfloat16' .* got {dtype!r}"):
        api(task).load(LOAD[task][0], device="cpu", dtype=dtype)


@pytest.mark.parametrize("task", TASKS)
def test_model_typo_suggests_the_id(task):
    ref, typo = LOAD[task]
    match = rf"unknown {task} model '{re.escape(typo)}'; did you mean (\S+, )*{re.escape(ref)}[,?]"
    with pytest.raises(ModelNotFoundError, match=match):
        api(task).load(typo, device="cpu")


@pytest.mark.parametrize("task", TASKS)
def test_kernels_on_cpu_names_auto(task):
    ref = LOAD[task][0]
    match = rf"^{re.escape(ref)}@1 \(mode='kernels'\): .*needs a CUDA device, got cpu; use mode='auto'"
    with pytest.raises(KernelUnavailableError, match=match):
        api(task).load(ref, device="cpu", mode="kernels")


@pytest.mark.parametrize("task", TASKS)
def test_offline_missing_weights_name_the_file_and_the_command(task, monkeypatch, tmp_path):
    ref = LOAD[task][0]
    monkeypatch.setenv(files.OFFLINE_ENV, "1")
    for var in (registry.WEIGHTS_PATH_ENV, files.READ_CACHES_ENV):  # nothing on this machine is found
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf_hub"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch_home"))

    def no_fetch(*_a, **_k):
        msg = "fetch attempted in offline mode"
        raise AssertionError(msg)

    monkeypatch.setattr(files, "_fetch", no_fetch)
    file = re.escape(registry.resolve(task, ref)[1].weights[0].name)
    match = rf"{re.escape(ref)}@1: {file} not found .*offline mode .*`fastforks download {re.escape(ref)}@1`"
    with pytest.raises(WeightsDownloadError, match=match):
        api(task).load(ref, device="cpu")


# --- call: a loaded model --------------------------------------------------------------------------------------------

_LAST: dict[str, tuple[str, Any]] = {}  # task -> (task, model): the last model loaded, for the next tests of its task


@pytest.fixture(scope="module")
def loaded(request, tmp_path_factory):
    """`(task, model)` of the task `request.param` (parametrized indirectly): the first model of `LOADED[task]` whose
    weights are on this machine, portable on the CPU; kept for the next tests of the task (the tests run per task).

    Loaded with the caller's caches (module scope: before `_isolated_cache` empties them for each test); this
    machine's own cache is only read, the weight stamps go to a temporary cache.
    """
    task = request.param
    if task in _LAST:
        return _LAST[task]
    _LAST.clear()  # one model in memory at a time
    with pytest.MonkeyPatch.context() as mp:
        read = [os.environ.get(files.READ_CACHES_ENV, ""), str(files.cache_root())]
        mp.setenv(files.READ_CACHES_ENV, os.pathsep.join(p for p in read if p))
        mp.setenv("FASTFORKS_CACHE", str(tmp_path_factory.mktemp("cache")))
        mp.setenv(registry.NONCOMMERCIAL_ENV, "1")
        for ref in LOADED[task]:
            card, variant = registry.resolve(task, ref)
            try:
                registry.weights(card, variant, allow_download=False)
            except WeightsError:
                continue
            _LAST[task] = task, api(task).load(ref, device="cpu", mode="portable")
            return _LAST[task]
    refs = LOADED[task]
    pytest.skip(f"{task}: no weights on this machine for {', '.join(refs)} (`fastforks download {refs[0]}` loads them)")


def call(task: str, model: Any, image: object) -> object:
    """`model` on `image` in its (first) image slot, the other inputs valid."""
    if task in PAIRS:
        return model(image, IMG)
    if task == "promptable_segmentation":
        return model(image, input_points=[[10, 10]], input_labels=[1])
    if task == "multiview":
        return model([image, IMG])
    if task == "tracking":
        return model([image, IMG, IMG], [[0, 10, 10]])
    return model(image)


MISSING = Path(__file__).parent / "data" / "no-such-image.png"
BAD = {
    "wrong channels": (np.zeros((64, 96, 2), np.uint8), r"1, 3 or 4 channels, got 2"),
    "1-D array": (np.zeros(64, np.uint8), r"HW, HWC or BHWC, got shape \(64,\)"),
    "complex": (IMG.astype(np.complex64), "unsupported image dtype torch.complex64"),
    "int64": (IMG.astype(np.int64), "unsupported image dtype torch.int64"),
    "None": (None, "unsupported image type NoneType"),
    "missing file": (MISSING, r"cannot read image .*no-such-image\.png"),
}
# a volume is an array of intensities (int64 is valid); a path is not read: the message names the reader
BAD_VOLUME = {
    "wrong channels": (np.zeros((3, 8, 8, 8), np.float32), r"takes 1 channel\(s\), got 3"),
    "1-D array": (np.zeros(8, np.float32), r"real \[D, H, W\] or \[C, D, H, W\] array, got torch.float32 \(8,\)"),
    "complex": (np.zeros((8, 8, 8), np.complex64), r"real \[D, H, W\] .* got torch.complex64"),
    "None": (None, "numpy array or torch tensor, got NoneType"),
    "missing file": (str(MISSING), r"got str; read a file with `vol, meta = fastforks\.volume\.read_nifti\(path\)`"),
}
BAD_CASES = [(t, b) for t in TASKS for b in sorted(BAD_VOLUME if t == "volume" else BAD)]


@pytest.mark.model
@pytest.mark.parametrize(("loaded", "bad"), BAD_CASES, indirect=["loaded"])
def test_bad_input(loaded, bad):
    task, m = loaded
    assert not MISSING.exists()
    x, match = (BAD_VOLUME if task == "volume" else BAD)[bad]
    with pytest.raises(InputError, match=match):
        m(x) if task == "volume" else call(task, m, x)


@pytest.mark.model
@pytest.mark.parametrize("loaded", SAME_SIZE, indirect=True)
def test_size_mismatch(loaded):
    _, m = loaded
    with pytest.raises(InputError, match=r"differ in shape: \(1, 3, 64, 96\) vs \(1, 3, 64, 88\)"):
        m(IMG, NARROW)


@pytest.mark.model
@pytest.mark.parametrize("loaded", PAIRS, indirect=True)
def test_list_length_mismatch(loaded):
    task, m = loaded
    first, second = ("left", "right") if task == "stereo" else ("image1", "image2")
    with pytest.raises(InputError, match=rf"non-empty lists of the same length \(got 1 {first}, 2 {second}\)"):
        m([IMG], [IMG, IMG])
    with pytest.raises(InputError, match=rf"\(got 1 {first}, one {second}\)"):
        m([IMG], IMG)


K = np.array([[50.0, 0, 48], [0, 50, 32], [0, 0, 1]], np.float32)  # one matrix: shared by every view
POSES = np.stack([np.eye(4, dtype=np.float32)] * 2)
NEAR_FAR = (0.5, 100.0)
CAMERAS: dict[str, tuple[list[np.ndarray], dict[str, Any], str]] = {  # case -> (views, arguments, message)
    "one view": ([IMG], {"intrinsics": K, "cam_to_world": POSES[:1], "depth_range": NEAR_FAR}, "takes at least 2 "),
    "no views": ([], {}, "a scene needs at least one image"),
    "no cameras": ([IMG, IMG], {}, r"needs intrinsics \(\[N, 3, 3\]\)"),
    "no intrinsics": ([IMG, IMG], {"cam_to_world": POSES, "depth_range": NEAR_FAR}, r"needs intrinsics \(\[N, 3, 3\]"),
    "no cam_to_world": ([IMG, IMG], {"intrinsics": K, "depth_range": NEAR_FAR}, r"needs cam_to_world \(\[N, 4, 4\]"),
    "no depth_range": ([IMG, IMG], {"intrinsics": K, "cam_to_world": POSES}, r"needs depth_range=\(near, far\)"),
    "pose count": (
        [IMG, IMG],
        {"intrinsics": K, "cam_to_world": POSES[:1], "depth_range": NEAR_FAR},
        "cam_to_world: expected 2 matrices",
    ),
    "view sizes": (
        [IMG, NARROW],
        {"intrinsics": K, "cam_to_world": POSES, "depth_range": NEAR_FAR},
        "views of a scene must be single images of one size",
    ),
}


@pytest.mark.model
@pytest.mark.parametrize("loaded", ["multiview"], indirect=True)
@pytest.mark.parametrize("case", sorted(CAMERAS))
def test_multiview_missing_cameras_and_views(loaded, case):
    _, m = loaded
    views, kwargs, match = CAMERAS[case]
    with pytest.raises(InputError, match=match):
        m(views, **kwargs)
