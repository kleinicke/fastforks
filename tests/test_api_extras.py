"""`load(dtype=)`, revision ids and `Result.numpy()` (CPU only, fake runner)."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
import torch

from fastforks import ModelNotFoundError, OptionsError, stereo
from fastforks._core import registry
from fastforks._core.instances import Instances
from fastforks.stereo._runner import Runner

pytestmark = pytest.mark.usefixtures("card_min_size")  # tiny images through fake runners


@dataclasses.dataclass(frozen=True)
class Params:
    feature_channels: int
    num_transformer: int
    dim_expansion: int
    use_positivity: bool
    refine_iter: int


class Dummy(Runner):
    Params = Params
    dtypes = ("float32", "float16")

    def __call__(self, left, right):  # noqa: ARG002 - test stub
        return {"disparity": left[:, 0], "confidence": left[:, 1], "occlusion": left[:, 2]}


@pytest.fixture(autouse=True)
def fake_runner(monkeypatch):
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": Dummy}))


def test_dtype_float32():
    assert stereo.load("s2m2-s", device="cpu", dtype="float32").backend.dtype == "float32"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_reduced_dtype_needs_cuda(dtype):
    with pytest.raises(OptionsError, match=rf"cannot compute in {dtype}.*\['float32'\]"):
        stereo.load("s2m2-s", device="cpu", dtype=dtype)


def test_bad_dtype_and_conflicts():
    with pytest.raises(OptionsError, match="dtype must be"):
        stereo.load("s2m2-s", device="cpu", dtype="int8")  # runtime check
    with pytest.raises(OptionsError, match="kernels runs reduced-precision"):
        stereo.load("s2m2-s", device="cpu", mode="kernels", dtype="float32")


def test_revision_is_the_id_suffix():
    assert stereo.load("s2m2-s@1", device="cpu").variant.ref == "s2m2-s@1"
    with pytest.raises(ModelNotFoundError, match="no revision '99'"):
        stereo.load("s2m2-s@99", device="cpu")


def test_numpy_of_stereo_result():
    m = stereo.load("s2m2-s", device="cpu")
    img = np.full((4, 5, 3), 200, np.uint8)
    out = m(img, img).numpy()
    assert set(out) == {"disparity", "confidence", "occlusion"}
    assert out["disparity"].shape == (4, 5)
    assert all(isinstance(v, np.ndarray) for v in out.values())


def test_numpy_skips_none_and_converts_bfloat16():
    from fastforks.features import FeaturesResult  # noqa: PLC0415

    r = FeaturesResult(
        last_hidden_state=torch.ones(1, 2, 3, dtype=torch.bfloat16),
        pooler_output=None,
        hidden_states=(None, torch.ones(1, 2, 3)),
        layers=(1,),
        grid=(1, 2),
        num_prefix_tokens=0,
        num_register_tokens=0,
        model="x@1",
        backend=stereo.Backend("portable", "cpu", "float32", ()),
    )
    out = r.numpy()
    assert set(out) == {"last_hidden_state", "hidden_states"}
    assert out["last_hidden_state"].dtype == np.float32
    assert out["hidden_states"][0] is None
    assert out["hidden_states"][1].shape == (1, 2, 3)


def test_numpy_of_instances():
    r = Instances(
        boxes=torch.zeros(2, 4),
        scores=torch.ones(2),
        labels=None,
        masks=torch.ones(2, 3, 3, dtype=torch.bool),
        class_names=(),
        model="x@1",
        backend=stereo.Backend("portable", "cpu", "float32", ()),
    )
    out = r.numpy()
    assert set(out) == {"boxes", "scores", "masks"}
    assert out["masks"].dtype == np.bool_
