"""Test-user findings TE-7, TG-2 (multi-view), TF-2/3/4/8/9, TH-1/3/4/7 (CPU, no weights)."""

from __future__ import annotations

import csv
import io
import json
import logging
import pathlib
import re
import types

import pytest
import torch

import fastforks
from fastforks import LicenceError
from fastforks._core import cli, registry
from fastforks._core import model as model_mod
from fastforks._core.model import Backend, Model, ReadOnlyDict

ROOT = pathlib.Path(fastforks.__file__).parent


@pytest.mark.slow
def test_pil_resize_is_one_clean_graph_break():  # TE-7
    from fastforks.detection._resize import pil_bilinear  # noqa: PLC0415

    def fn(x: torch.Tensor) -> torch.Tensor:
        return pil_bilinear(x, (24, 40)) / 255

    explain = torch._dynamo.explain(fn)(torch.randint(0, 256, (1, 3, 50, 70)).float())
    assert explain.graph_break_count <= 1, explain.break_reasons
    x = torch.randint(0, 256, (1, 3, 50, 70)).float()
    torch.testing.assert_close(torch.compile(fn)(x), fn(x))  # inductor may fold / 255 into * (1 / 255)


@pytest.mark.parametrize("path", ["vggt/model.py", "mapanything/model.py", "vggt_omega/model.py"])
def test_multiview_portable_residual_is_out_of_place_under_grad(path):  # TG-2
    src = (ROOT / "multiview/models" / path).read_text()
    assert re.search(r"torch\.addcmul\(x, o, [\w.]+\) if torch\.is_grad_enabled\(\) else x\.addcmul_\(", src)


def test_env_has_a_cpu_line(capsys):  # TF-2
    assert cli.main(["env"]) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("cpu "))
    assert f"{torch.get_num_threads()} torch threads" in line
    assert "torch" in line
    assert "build" in line


def test_env_cpu_line_does_not_claim_cpu_with_an_apple_gpu(capsys, monkeypatch):  # Mac verify 2026-11-05
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert cli.main(["env"]) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("cpu "))
    assert "run portable on the CPU" not in line


def test_fork_warning_once(caplog, monkeypatch):  # TF-3
    ns = types.SimpleNamespace(
        _pid=-1,
        _fork_warned=False,
        device=torch.device("cpu"),
        card=types.SimpleNamespace(task="depth"),
        variant=types.SimpleNamespace(ref="m@1"),
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    with caplog.at_level(logging.WARNING, logger="fastforks.depth"):
        Model._warn_if_forked(ns)
        Model._warn_if_forked(ns)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "forked" in msgs[0]
    assert "spawn" in msgs[0]
    ns2 = types.SimpleNamespace(**{**vars(ns), "_pid": __import__("os").getpid(), "_fork_warned": False})
    caplog.clear()
    Model._warn_if_forked(ns2)
    assert not caplog.records


def test_fork_flag_set_in_forked_child():  # TF-3
    import os  # noqa: PLC0415

    if not hasattr(os, "fork"):
        pytest.skip("no fork")
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: only the flag, then exit without running pytest teardown
        os.write(w, b"1" if model_mod._FORKED else b"0")
        os._exit(0)
    os.waitpid(pid, 0)
    assert os.read(r, 1) == b"1"
    assert model_mod._FORKED is False


def test_results_and_backend_load_with_weights_only():  # TF-4
    from fastforks.depth import DepthResult  # noqa: PLC0415
    from fastforks.detection import DetectionResult  # noqa: PLC0415

    b = Backend("portable", "cpu", "float32", library="1", weights=("ab",))
    det = DetectionResult(
        torch.zeros(1, 4),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        None,
        ("a",),
        "m@1",
        b,
        ReadOnlyDict({"x": 1}),
    )
    buf = io.BytesIO()
    torch.save(det, buf)
    buf.seek(0)
    got = torch.load(buf)  # weights_only=True by default
    assert isinstance(got, DetectionResult)
    assert got.backend.weights == ("ab",)
    assert got.options["x"] == 1
    assert DepthResult.__mro__  # registered through NumpyOutputs.__init_subclass__


def test_backend_provenance_not_in_equality_or_repr():  # TH-7
    a = Backend("portable", "cpu", "float32", library="1.0", weights=("aa",))
    b = Backend("portable", "cpu", "float32", library="2.0", weights=("bb",))
    assert a == b
    assert "1.0" not in repr(a)
    assert "aa" not in repr(a)


def test_onnx_extra_has_onnxruntime():  # TF-8
    text = (ROOT.parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^onnx = \[.*"onnxruntime', text, re.MULTILINE)


def test_classification_example_inputs_default_size():  # TF-9
    from fastforks.classification._api import ClassificationModel  # noqa: PLC0415

    m = object.__new__(ClassificationModel)
    m.card, m.variant = registry.resolve_any("resnet50.tv2_in1k")
    m.device = torch.device("cpu")
    (x,) = m.example_inputs()
    assert x.shape[-2:] == m.export_size()
    with pytest.raises(fastforks.InputError, match="both height and width"):
        m.example_inputs(224)


def test_list_format_json_and_csv(capsys):  # TH-1
    assert cli.main(["list", "hitnet*", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows
    assert {"id", "commercial", "weights_sha256", "weights_urls", "code_licence"} <= set(rows[0])
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for r in rows for h in r["weights_sha256"])
    assert cli.main(["list", "hitnet*", "--format", "csv"]) == 0
    got = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["id"] for r in got] == [r["id"] for r in rows]


def _nc_ref() -> str:
    return registry.list_models(registry.TASKS, commercial="no")[0]


def test_commercial_only_refuses_and_filters(monkeypatch, capsys):  # TH-4
    ref = _nc_ref()
    card, v = registry.resolve_any(ref)
    registry.require_commercial(card, v)  # off: no error
    monkeypatch.setenv(registry.COMMERCIAL_ONLY_ENV, "1")
    with pytest.raises(LicenceError, match=registry.COMMERCIAL_ONLY_ENV):
        registry.require_commercial(card, v)
    assert ref not in registry.list_models(registry.TASKS)
    assert all(
        registry.resolve_any(m)[0].status(registry.resolve_any(m)[1]).commercial == "yes"
        for m in registry.list_models(registry.TASKS)
    )
    assert cli.main(["list"]) == 0
    assert "showing only models cleared" in capsys.readouterr().err
    with pytest.raises(LicenceError):  # load of the task's API goes through load_model
        model_mod.load_model(card.task, ref, Model)


def test_hf_weight_urls_are_pinned_to_a_commit():  # TH-3
    bad = []
    for ref in registry.list_models(registry.TASKS):
        v = registry.resolve_any(ref)[1]
        for w in v.weights:
            m = re.match(r"https://huggingface\.co/(?!fastforks/)[^/]+/[^/]+/resolve/([^/]+)/", w.url or "")
            if m and not re.fullmatch(r"[0-9a-f]{40}", m[1]):
                bad.append((ref, w.url))
    assert not bad, bad
