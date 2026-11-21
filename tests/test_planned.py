"""`[planned]`: features known not to work in this release raise one clear error (load, kernels, compile, export)."""

from __future__ import annotations

import copy
import dataclasses
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastforks import BackendUnavailableError, CardError
from fastforks._core import registry
from fastforks._core.cards import parse_card, planned_reason, require_available
from fastforks._core.export import check_exportable, export_onnx, export_tensorrt
from fastforks._core.model import Model, _no_auto_kernels

S2M2 = Path(__file__).parents[1] / "src/fastforks/stereo/models/s2m2/card.toml"


def _parse(d: dict) -> object:
    return parse_card(d, family="s2m2", task="stereo", where="card.toml")


@pytest.fixture
def data() -> dict:
    return tomllib.loads(S2M2.read_text(encoding="utf-8"))


def test_card_and_variant_planned_merge(data: dict) -> None:
    d = copy.deepcopy(data)
    d["planned"] = {"onnx": "family reason"}
    d["variants"][0]["planned"] = {"compile": "variant reason"}
    card = _parse(d)
    first, *rest = card.variants
    assert dict(first.planned) == {"onnx": "family reason", "compile": "variant reason"}
    assert all(dict(v.planned) == {"onnx": "family reason"} for v in rest)
    assert planned_reason(first, "tensorrt") == "family reason"  # tensorrt exports ONNX first
    assert planned_reason(first, "kernels") is None


@pytest.mark.parametrize("bad", [{"trt": "x"}, {"onnx": ""}, {"onnx": 3}])
def test_bad_planned_is_a_card_error(data: dict, bad: dict) -> None:
    d = copy.deepcopy(data)
    d["planned"] = bad
    with pytest.raises(CardError):
        _parse(d)


def _variant(**planned: str) -> object:
    v = _parse(tomllib.loads(S2M2.read_text(encoding="utf-8"))).variants[0]
    return dataclasses.replace(v, planned=planned)


@pytest.mark.parametrize("feature", ["load", "kernels", "compile", "onnx", "tensorrt"])
def test_message_names_release_and_reason(feature: str) -> None:
    v = _variant(**{feature: "the reason."})
    with pytest.raises(BackendUnavailableError, match=r"not available in 0\.2\.0, planned for 0\.2\.1: the reason"):
        require_available(v, feature)
    require_available(_variant(), feature)  # nothing planned: no error


def test_hooks_raise_before_any_work() -> None:
    v = _variant(onnx="o", compile="c")
    fake = SimpleNamespace(variant=v, backend=SimpleNamespace(mode="portable"))
    with pytest.raises(BackendUnavailableError, match="export_onnx is not available"):
        check_exportable(fake, 64, 64, 1)
    with pytest.raises(BackendUnavailableError, match=r"export_tensorrt is not available.*: o"):
        check_exportable(fake, 64, 64, 1, "tensorrt")
    with pytest.raises(BackendUnavailableError, match=r"compile\(\) is not available"):
        Model.compile(fake)
    for export in (export_onnx, export_tensorrt):  # the gate comes before "give an export size"
        with pytest.raises(BackendUnavailableError, match=r"is not available.*: o"):
            export(fake, "x.onnx")


def test_auto_skips_planned_kernels() -> None:
    card = _parse(tomllib.loads(S2M2.read_text(encoding="utf-8")))
    reason = _no_auto_kernels(card, _variant(kernels="k"), torch.device("cuda"), fp32=False, want=None)
    assert reason is not None
    assert "planned for 0.2.1" in reason


def test_shipped_planned_entries_are_valid() -> None:
    planned = [v for t in registry.TASKS for _, v in registry.cards(t).values() if v.planned]
    assert planned, "the shipped cards gate the known-broken features (docs/plans/KNOWN_BROKEN_0.2.1.md)"
    for v in planned:
        assert {*v.planned} <= {"load", "kernels", "compile", "onnx", "tensorrt"}
