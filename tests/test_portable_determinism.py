"""Portable on CUDA is deterministic: repeated calls and fresh processes give bitwise equal outputs.

`Model._session` runs CUDA calls of runners with `deterministic_cudnn` with `torch.backends.cudnn.deterministic =
True`; without it cuDNN picked transposed-conv algorithms with atomics and the GRU / cost-volume families amplified the
last-bit differences (IGEV++ fp32 at Middlebury-H: up to 16 px max, 0.0017 px mean between two calls). CoEx and
Fast-ACVNet keep cuDNN's default (deterministic costs them ~5 %); `torch.backends.cudnn.deterministic = True` set by the
user makes them deterministic too.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from fastforks._core import model as core_model
from synth import pair
from test_golden import API

# one per family with `deterministic_cudnn` (each varied between calls in fp32 or its default dtype before)
REFS = [
    "igev-plusplus-sceneflow@1",
    "igev-stereo-sceneflow@1",
    "great-igev-sceneflow@1",
    "selective-igev-sceneflow@1",
    "mocha-v1-sceneflow@1",
    "monster-mix-all@1",
    "monster-plusplus-mix-all@1",
    "rt-monster-plusplus-mix-all@1",
    "cfnet-sceneflow@1",
    "las2-s@1",
    "foundationstereo-vit-s@1",
    "fast-foundationstereo@1",
    "c-fast-foundationstereo@1",
]
PROCESS_REFS = ["igev-plusplus-sceneflow@1", "great-igev-sceneflow@1", "monster-plusplus-mix-all@1"]
OPT_IN_REFS = ["fast-acvnet-plus-generalization@1", "coex-sceneflow@1"]  # keep cuDNN's default: ~5 % slower
SIZE = (480, 640)

_CHILD = """
import hashlib, sys
import torch
ref, dtype, h, w, tests = sys.argv[1:6]
sys.path.insert(0, tests)
from synth import pair
from fastforks import stereo
m = stereo.load(ref, device="cuda", mode="portable", dtype=None if dtype == "default" else dtype)
d = m(*pair(int(h), int(w), 0)).disparity
print(hashlib.sha256(d.float().cpu().numpy().tobytes()).hexdigest())
"""


def _load(ref: str, dtype: str) -> Any:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return API["stereo"].load(ref, device="cuda", mode="portable", dtype=None if dtype == "default" else dtype)


def _call(model: Any) -> torch.Tensor:
    return model(*pair(*SIZE, 0)).disparity.clone()


def _sha(x: torch.Tensor) -> str:
    return hashlib.sha256(x.float().cpu().numpy().tobytes()).hexdigest()


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("dtype", ["default", "float32"])
@pytest.mark.parametrize("ref", REFS)
def test_portable_calls_are_bitwise_equal(ref, dtype):
    """Calls 1-4 on the same input are `torch.equal`, and the global cuDNN flag is restored after each call."""
    model = _load(ref, dtype)
    before = torch.backends.cudnn.deterministic
    outs = [_call(model) for _ in range(4)]
    assert torch.backends.cudnn.deterministic == before
    for i in range(1, 4):
        diff = (outs[i] - outs[0]).abs().max()
        assert torch.equal(outs[i], outs[0]), f"{ref} {dtype}: call {i + 1} != call 1: {diff}"


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("ref", OPT_IN_REFS)
def test_portable_opt_in_with_the_torch_flag(ref, monkeypatch):
    """CoEx / Fast-ACVNet: `torch.backends.cudnn.deterministic = True` (the user's switch) makes calls 1-4 equal."""
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    model = _load(ref, "default")
    outs = [_call(model) for _ in range(4)]
    assert torch.backends.cudnn.deterministic
    for i in range(1, 4):
        assert torch.equal(outs[i], outs[0]), f"{ref}: call {i + 1} != call 1: {(outs[i] - outs[0]).abs().max()}"


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize("dtype", ["default", "float32"])
@pytest.mark.parametrize("ref", PROCESS_REFS)
def test_portable_fresh_process_gives_the_same_output(ref, dtype):
    """A fresh process (cold cuDNN heuristics cache) returns this process's output bit for bit."""
    want = _sha(_call(_load(ref, dtype)))
    torch.cuda.synchronize()
    tests = str(Path(__file__).parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    args = [sys.executable, "-c", _CHILD, ref, dtype, str(SIZE[0]), str(SIZE[1]), tests]
    got = subprocess.run(args, env=env, capture_output=True, text=True, timeout=300, check=True)
    assert got.stdout.strip().splitlines()[-1] == want


@pytest.mark.gpu
@pytest.mark.model
def test_portable_without_the_flag_varies(monkeypatch):
    """Negative check: with the session's flag disabled, portable IGEV++ fp32 differs between calls."""
    monkeypatch.setattr(core_model, "_deterministic_cudnn", contextlib.nullcontext)
    model = _load("igev-plusplus-sceneflow@1", "float32")
    outs = [_call(model) for _ in range(4)]
    assert any(not torch.equal(o, outs[0]) for o in outs[1:])
