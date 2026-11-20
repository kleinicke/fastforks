"""The committed speed floor keeps its mode order: kernels <= portable <= the faster original, per ref and input.

Exceptions need an entry with the measured reason in docs/data/speed_allowlist.json; an entry that no longer matches a
violation fails too, so the list shrinks with each fix. Re-measuring against the floor needs a GPU:
`python tools/speed_check.py run --out rows.jsonl --check` (see CONTRIBUTING.md).
"""

import json

import pytest

import speed_check


def test_floor_mode_order_or_allowlisted():
    allow = json.loads(speed_check.ALLOW.read_text())
    found = dict(speed_check.static_violations())
    assert not sorted(set(found) - set(allow)), "slower than a mode it must not lose to; fix it or allowlist it"
    assert not sorted(set(allow) - set(found)), "stale allowlist entries: remove them"


def test_allowlist_entries_give_a_reason():
    for key, reason in json.loads(speed_check.ALLOW.read_text()).items():
        assert isinstance(reason, str), key
        assert len(reason) > 20, key


def test_tensorrt_rows_fail_on_all_nan_outputs():
    """A TensorRT engine that builds but outputs only NaN (mvsanywhere: a fused-graph TensorRT bug) is a failed row."""
    import torch  # noqa: PLC0415

    from fastforks._core.export import all_nan  # noqa: PLC0415 - speed_check's check

    nan = torch.full((2, 3), float("nan"))
    assert all_nan(nan) == "0"
    assert all_nan((torch.ones(2), nan)) == "1"
    assert all_nan({"depth": nan}) == "depth"
    assert all_nan((torch.ones(2), torch.tensor([float("nan"), 1.0]), torch.zeros(0))) is None
    assert all_nan(torch.zeros(3, dtype=torch.long)) is None


def test_only_takes_model_ids_and_rejects_unknown_names():
    """`--only` with a name that is no floor ref used to select nothing, so `run` ended without a row or an error."""
    ref = speed_check.floor_rows()[0]["ref"]
    assert speed_check._floor_refs([ref]) == {ref}
    assert speed_check._floor_refs(["mvsplat-re10k@1"]) == {"mvsplat-re10k"}
    with pytest.raises(SystemExit, match="no-such-model"):
        speed_check._floor_refs(["no-such-model"])


def test_full_measurement_prepares_input_without_core_capture(monkeypatch, tmp_path):
    """Interactive sessions have no safely replayable runner call, but still get a full speed measurement."""
    import torch  # noqa: PLC0415

    calls, rows = [], []

    def full(*, prepare=False):
        calls.append("prepare" if prepare else "call")

    monkeypatch.setattr(speed_check, "measure", lambda fn: (fn(), {"ms": 1.0})[1])
    monkeypatch.setattr(speed_check, "_append", lambda _out, row: rows.append(row.copy()))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    spec = {
        "task": "volume",
        "side": "lib",
        "mode": "compile",
        "inputs": [{"label": "200x200", "scene": ""}],
        "no_profile": True,
    }
    speed_check.speed_step(None, spec, {}, [("200x200", full, None)], tmp_path / "rows.jsonl")
    assert calls == ["prepare", "call"]
    assert rows[0]["ms"] == 1.0
    assert "core_ms" not in rows[0]
    assert "fails" not in rows[0]


def test_full_timing_can_skip_runner_replay(monkeypatch, tmp_path):
    """The full-call timing is recorded and its inputs released without a runner replay."""
    import torch  # noqa: PLC0415

    calls, rows = [], []

    def full(*, prepare=False):
        calls.append("prepare" if prepare else "call")

    def core(*, release=False):
        assert release
        calls.append("release")

    monkeypatch.setattr(speed_check, "measure", lambda fn: (fn(), {"ms": 1.0, "peak_alloc_mb": 2})[1])
    monkeypatch.setattr(speed_check, "_append", lambda _out, row: rows.append(row.copy()))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    spec = {
        "task": "multiview",
        "side": "lib",
        "mode": "portable",
        "inputs": [{"label": "480x640", "scene": ""}],
        "no_profile": True,
        "no_core": True,
    }
    speed_check.speed_step(None, spec, {}, [("480x640", full, core)], tmp_path / "rows.jsonl")
    assert calls == ["prepare", "call", "release"]
    assert (rows[0]["ms"], rows[0]["peak_alloc_mb"]) == (1.0, 2)
    assert rows[0]["core_skipped"]
    assert "core_ms" not in rows[0]


def test_resume_distinguishes_full_timing_and_runner_timing(monkeypatch):
    """A timing receipt without the optional runner result can only resume the same timing scope."""
    monkeypatch.setattr(speed_check, "files_hash", lambda _: "hash")
    job = {"ref": "model", "mode": "portable", "inputs": [{"label": "32x32"}]}
    row = {"side": "lib", "ref": "model", "mode": "portable", "input": "32x32", "code_hash": "hash"}
    fast = job | {"no_core": True}
    skipped = row | {"core_skipped": True}
    assert speed_check._done([skipped], "lib", fast)
    assert not speed_check._done([skipped], "lib", job)
    assert speed_check._done([row], "lib", fast)
    assert speed_check._done([row], "lib", job)
