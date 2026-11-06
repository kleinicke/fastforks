"""`load(weights=PATH)`: the user's own weights, unverified but loaded strictly."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import logging
import os
import re
import shutil
import zipfile
from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import CheckpointMismatchError, WeightsError, features, stereo
from fastforks._core import checkpoint, registry, runner
from fastforks._core import files as files_mod
from fastforks._core.checkpoint import load_state_dict, load_strict
from fastforks._core.model import user_weights
from fastforks.promptable_segmentation.models.sam3.net import sam3 as sam3_net
from test_golden import API, SMALLEST, TASK

if TYPE_CHECKING:
    from pathlib import Path

REF = "fast-acvnet-sceneflow@1"  # small (13 MB), a plain {"model": state_dict} checkpoint


def _variant(ref: str):
    return registry.resolve(registry.task_of(ref) or "stereo", ref)[1]


def _local(ref: str) -> list[Path]:
    card, variant = registry.resolve(TASK.get(ref) or registry.task_of(ref) or "stereo", ref)
    try:
        return registry.weights(card, variant, allow_download=False)
    except WeightsError as e:
        pytest.skip(f"weights of {ref} not available locally: {e}")


def test_one_file_variant_takes_the_file_or_its_directory(tmp_path):
    v = _variant("s2m2-s@1")
    mine = tmp_path / "anything.pth"
    mine.write_bytes(b"")
    assert user_weights(v, mine) == (mine,)  # any name
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(WeightsError, match="or pass the file itself"):
        user_weights(v, empty)
    named = tmp_path / v.weights[0].name
    named.parent.mkdir(parents=True, exist_ok=True)
    named.write_bytes(b"")
    assert user_weights(v, str(tmp_path)) == (named,)


def test_several_files_need_a_directory(tmp_path):
    v = _variant("marigold-depth-v1-1@1")
    names = [w.name for w in v.weights]
    assert len(names) > 1
    f = tmp_path / "unet.safetensors"
    f.write_bytes(b"")
    with pytest.raises(WeightsError, match=r"is a file, but .* has 3 weight files"):
        user_weights(v, f)
    for n in names[:-1]:
        (tmp_path / n).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / n).write_bytes(b"")
    with pytest.raises(WeightsError, match=f"needs {names[-1]} in this directory"):
        user_weights(v, tmp_path)
    (tmp_path / names[-1]).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / names[-1]).write_bytes(b"")
    assert user_weights(v, tmp_path) == tuple(tmp_path / n for n in names)


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(WeightsError, match="no such file or directory"):
        stereo.load(REF, device="cpu", weights=tmp_path / "nope.ckpt")


@pytest.mark.model
def test_renamed_copy_loads_unverified(tmp_path, caplog):
    src = _local(REF)[0]
    mine = tmp_path / "my_finetune.bin"
    shutil.copy(src, mine)
    with caplog.at_level(logging.INFO, logger="fastforks"):
        model = stereo.load(REF, device="cpu", weights=mine)
    ref = stereo.load(REF, device="cpu")
    unverified = [r for r in caplog.records if "unverified weights" in r.getMessage()]
    assert len(unverified) == 1
    assert unverified[0].levelno == logging.WARNING
    assert str(mine) in unverified[0].getMessage()
    assert model.weights == (mine,)
    assert str(mine) in repr(model)
    assert ref.weights is None
    for a, b in zip(model.module().state_dict().values(), ref.module().state_dict().values(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.model
def test_changed_weights_are_used(tmp_path):
    sd = load_state_dict(_local(REF)[0], key="model")
    name = next(k for k, v in sd.items() if v.is_floating_point())
    sd[name] = sd[name] + 1
    torch.save({"model": sd}, tmp_path / "w.ckpt")
    model = stereo.load(REF, device="cpu", weights=tmp_path / "w.ckpt")
    ref = stereo.load(REF, device="cpu")
    mine, theirs = model.module().state_dict(), ref.module().state_dict()
    assert [k for k in mine if not torch.equal(mine[k], theirs[k])]  # the +1 arrived


@pytest.mark.model
@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("add", r"1 unexpected keys \(extra\.weight\)"),
        ("drop", r"1 missing keys \({name}\)"),
        ("shape", r"1 shape mismatches \({name}: checkpoint \[\d+, "),
    ],
)
def test_mismatched_weights_list_the_keys(tmp_path, change, error):
    sd = load_state_dict(_local(REF)[0], key="model")
    name = next(k for k, v in sd.items() if v.is_floating_point() and v.ndim > 1)
    if change == "add":
        sd["extra.weight"] = torch.zeros(1)
    elif change == "drop":
        del sd[name]
    else:
        sd[name] = torch.cat([sd[name], sd[name][:1]])
    torch.save({"model": sd}, tmp_path / "w.ckpt")
    with pytest.raises(
        CheckpointMismatchError,
        match=rf"weights=\S+ for {REF} does not fit the network: " + error.format(name=name.replace(".", r"\.")),
    ):
        stereo.load(REF, device="cpu", weights=tmp_path / "w.ckpt")


@pytest.mark.slow
@pytest.mark.parametrize("content", ["empty", "zip", "truncated"])
def test_unreadable_file_names_the_path_and_the_variant(tmp_path, content):
    """Another format, an empty or a truncated file is a `WeightsError` naming it, not a raw torch / pickle error."""
    path = tmp_path / "w.ckpt"
    if content == "zip":
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("notes.txt", "not a checkpoint")
    elif content == "truncated":
        path.write_bytes(_local(REF)[0].read_bytes()[:4096])
    else:
        path.write_bytes(b"")
    with pytest.raises(WeightsError, match=rf"weights={re.escape(str(path))} for {REF}") as info:
        stereo.load(REF, device="cpu", weights=path)
    assert not isinstance(info.value, CheckpointMismatchError)


@pytest.mark.model
def test_errors_after_the_weights_are_loaded_are_not_weights_errors(monkeypatch):
    """A CUDA or kernels-kernel error while building from user weights surfaces as it is, not as a `WeightsError`."""
    module = importlib.import_module("fastforks.stereo.models.fast_acvnet.model")

    class Failing(module.Fast_ACVNet):
        def eval(self):  # right after the weights are loaded
            msg = "CUDA error: an illegal memory access"
            raise RuntimeError(msg)

    monkeypatch.setattr(module, "Fast_ACVNet", Failing)
    with pytest.raises(RuntimeError, match="illegal memory access") as info:
        stereo.load(REF, device="cpu", weights=_local(REF)[0])
    assert not isinstance(info.value, WeightsError)


@pytest.mark.model
def test_other_precision_is_cast_to_the_original(tmp_path):
    """A float16 copy of float32 weights loads in float32, both copied (fast-acvnet) and assigned (DINOv2)."""
    sd = load_state_dict(_local(REF)[0], key="model")
    torch.save({"model": {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}}, tmp_path / "h.ckpt")
    mine = stereo.load(REF, device="cpu", weights=tmp_path / "h.ckpt").module().state_dict()
    theirs = stereo.load(REF, device="cpu").module().state_dict()
    for k, v in theirs.items():
        assert mine[k].dtype == v.dtype, k
        assert torch.equal(mine[k], v.half().to(v.dtype) if v.is_floating_point() else v), k
    ref = "dinov2-vits14"
    sd = load_state_dict(_local(ref)[0], strip="")
    torch.save({k: v.half() for k, v in sd.items()}, tmp_path / "dino.pth")
    del sd
    mine = features.load(ref, device="cpu", weights=tmp_path / "dino.pth").module().state_dict()
    theirs = features.load(ref, device="cpu").module().state_dict()
    for k, v in theirs.items():
        assert mine[k].dtype == v.dtype, k
        assert torch.equal(mine[k], v.half().float()), k


def test_load_strict_casts_floats_and_rejects_integers():
    net = torch.nn.Linear(2, 3)
    half = {k: v.half() for k, v in torch.nn.Linear(2, 3).state_dict().items()}
    for assign, dtype, want in [
        (False, None, torch.float32),
        (True, None, torch.float32),
        (True, torch.half, torch.half),
    ]:
        with torch.device("meta") if assign else contextlib.nullcontext():
            net = torch.nn.Linear(2, 3)
        load_strict(net, half, "test checkpoint", assign=assign, dtype=dtype)
        assert {v.dtype for v in net.state_dict().values()} == {want}
        assert torch.equal(net.weight.half(), half["weight"])
    with pytest.raises(
        CheckpointMismatchError, match=r"1 dtype mismatches \(bias: checkpoint int64 vs network float32\)"
    ):
        load_strict(torch.nn.Linear(2, 3), {**half, "bias": torch.zeros(3, dtype=torch.long)}, "test checkpoint")


ASSIGNED = [r for r in SMALLEST if r.startswith(("da3", "depth-pro", "dinov2", "dinov3", "sam3", "unik3d"))]


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", ASSIGNED)
def test_recorded_checkpoints_load_without_a_cast(ref, monkeypatch):
    """The families that assign their checkpoint get it in the dtype it always had (the cast is for users' files)."""
    if ref.split("@")[0] in _BIG and not os.environ.get("FASTFORKS_HEAVY_CPU_TESTS"):
        pytest.skip(f"{ref}: a {_BIG[ref.split('@')[0]]} load on the CPU; run alone with FASTFORKS_HEAVY_CPU_TESTS=1")
    _local(ref)
    cast = []
    real = checkpoint._cast

    def recording(tensor, dtype):
        if tensor.is_floating_point() and tensor.dtype != dtype:
            cast.append(f"{tensor.dtype} -> {dtype}")
        return real(tensor, dtype)

    monkeypatch.setattr(checkpoint, "_cast", recording)
    API[TASK[ref]].load(ref, device="cpu")
    gc.collect()
    assert cast == []


def _perturbed(change: str):
    """`load_state_dict` with one change to every state dict it returns."""
    real = load_state_dict

    def load(*args, **kwargs):
        sd = real(*args, **kwargs)
        floats = [k for k, v in sd.items() if isinstance(v, torch.Tensor) and v.is_floating_point() and v.ndim >= 1]
        big = max(floats, key=lambda k: sd[k].numel())  # surely used by the network
        if change == "add":
            sd["zz_unexpected.weight"] = torch.zeros(3)
        elif change == "drop":
            del sd[big]
        else:
            sd[big] = torch.cat([sd[big], sd[big][:1]])
        return sd

    return load


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("change", ["add", "drop", "shape"])
@pytest.mark.parametrize("ref", [r for r in SMALLEST if not r.startswith("hitnet")])  # hitnet: graph files
def test_every_family_loads_user_weights_strictly(ref, change, tmp_path, monkeypatch):
    """A state dict with an unexpected, a missing or a reshaped tensor is a `WeightsError`, never a silent load."""
    if ref.split("@")[0] in _BIG and not os.environ.get("FASTFORKS_HEAVY_CPU_TESTS"):
        pytest.skip(f"{ref}: a {_BIG[ref.split('@')[0]]} load (~50 s) per case; run with FASTFORKS_HEAVY_CPU_TESTS=1")
    files = _local(ref)
    variant = _variant(ref)
    if len(files) == 1:
        path = files[0]
    else:
        path = tmp_path
        for spec, f in zip(variant.weights, files, strict=True):
            (tmp_path / spec.name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / spec.name).symlink_to(f)
    monkeypatch.setattr(runner, "load_state_dict", _perturbed(change))
    monkeypatch.setattr(registry, "weights", _no_download)  # e.g. gated SAM 3: a local copy only
    monkeypatch.setattr(files_mod, "download", _no_download)
    keys = {
        "add": r"1 unexpected keys \(\S*zz_unexpected\.weight\)",
        "drop": r"1 missing keys \(\S+\)",
        "shape": r"1 shape mismatches \(\S+: checkpoint \[",
    }[change]
    with pytest.raises(CheckpointMismatchError, match=rf"for {re.escape(ref)} does not fit the network: {keys}"):
        API[TASK[ref]].load(ref, device="cpu", weights=path)


def _no_download(*args, **kwargs):
    msg = "weights=PATH must not look up or download the recorded weights"
    raise AssertionError(msg)


# SAM 2.1's image-mode modules and the tensors SAM 3's image detector skips (the filters before the strict loads)
SAM2_PARTS = ("image_encoder.", "sam_prompt_encoder.", "sam_mask_decoder.", "no_mem_embed")
SAM3_SKIPPED = ("tracker_model.", "tracker_neck.", *sam3_net.UNUSED)

# family -> (model module, the load it used before it became strict): upstream's `my_load_state_dict` (s2m2),
# `strict=False` (aanet), a filter that kept known prefixes and dropped any other key (SAM 2.1, SAM 3)
LENIENT = {
    "s2m2": ("fastforks.stereo.models.s2m2.model", lambda net, state, *_, **__: net.my_load_state_dict(state)),
    "aanet": (
        "fastforks.stereo.models.aanet.model",
        lambda net, state, *_, **__: net.load_state_dict(state, strict=False),
    ),
    "sam2.1": (
        "fastforks.promptable_segmentation.models.sam2.model",
        lambda net, state, *_, **__: net.load_state_dict({k: v for k, v in state.items() if k.startswith(SAM2_PARTS)}),
    ),
    "sam3": (
        "fastforks.promptable_segmentation.models.sam3.model",
        lambda net, state, *_, **__: net.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith(SAM3_SKIPPED)}, assign=True
        ),
    ),
}
LENIENT_REFS = [
    m for task in ("stereo", "promptable_segmentation") for m in registry.cards(task) if m.split("-")[0] in LENIENT
]

_BIG = {"s2m2-xl": "1.6 GB", "sam3": "3.4 GB"}  # keep the suite well under its 20 GB cgroup


@pytest.mark.slow
@pytest.mark.model
@pytest.mark.parametrize("ref", LENIENT_REFS)
def test_strict_load_keeps_every_tensor_of_the_lenient_one(ref, monkeypatch):
    """The recorded weights load strictly (minus the declared unused tensors) into exactly the old network state."""
    if ref in _BIG and not os.environ.get("FASTFORKS_HEAVY_CPU_TESTS"):
        pytest.skip(f"{ref}: two {_BIG[ref]} loads on the CPU; run alone with FASTFORKS_HEAVY_CPU_TESTS=1")
    _local(ref)
    name, lenient_load = LENIENT[ref.split("-")[0]]
    module = importlib.import_module(name)
    torch.manual_seed(0)
    strict = _loaded(ref, module, module.load_strict, monkeypatch)
    torch.manual_seed(1)  # a tensor the lenient load left at its random init would differ
    lenient = _loaded(ref, module, lenient_load, monkeypatch)
    assert strict
    assert strict.keys() == lenient.keys()
    assert [k for k in strict if strict[k] != lenient[k]] == []


def _loaded(ref: str, module, load, monkeypatch) -> dict[str, str]:
    """Digests of the network right after `load` (as the family's `load_strict`); one network in memory at a time."""
    seen: dict[str, str] = {}

    def digest_after(net, state, *args, **kwargs):
        load(net, state, *args, **kwargs)
        seen.update(_digests(net))

    with monkeypatch.context() as m:
        m.setattr(module, "load_strict", digest_after)
        API[registry.task_of(ref) or "stereo"].load(ref, device="cpu")
    gc.collect()
    return seen


def _digests(module: torch.nn.Module) -> dict[str, str]:
    """SHA-256, dtype and shape of every tensor of `module`."""
    return {
        k: hashlib.sha256(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()
        + f"{v.dtype}{list(v.shape)}"
        for k, v in module.state_dict().items()
    }


def test_a_family_keyerror_on_user_weights_is_a_checkpoint_mismatch(tmp_path):
    from fastforks._core.model import _mismatch  # noqa: PLC0415

    err = _mismatch(KeyError("model.weight"), (tmp_path / "w.ckpt",), "x@1")
    assert err is not None
    assert "the checkpoint has no 'model.weight'" in str(err)
