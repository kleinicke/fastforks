"""The committed speed floor keeps its mode order: kernels <= portable <= the faster original, per ref and input.

Exceptions need an entry with the measured reason in docs/data/speed_allowlist.json; an entry that no longer matches a
violation fails too, so the list shrinks with each fix. Re-measuring against the floor needs a GPU:
`python tools/speed_check.py run --out rows.jsonl --check` (see CONTRIBUTING.md).
"""

import json
from argparse import Namespace
from subprocess import CompletedProcess

import pytest

import merge_check
import speed_check


def test_floor_mode_order_or_allowlisted():
    allow = json.loads(speed_check.ALLOW.read_text(encoding="utf-8"))
    found = dict(speed_check.static_violations())
    assert not sorted(set(found) - set(allow)), "slower than a mode it must not lose to; fix it or allowlist it"
    assert not sorted(set(allow) - set(found)), "stale allowlist entries: remove them"


def test_allowlist_entries_give_a_reason():
    for key, reason in json.loads(speed_check.ALLOW.read_text(encoding="utf-8")).items():
        assert isinstance(reason, str), key
        assert len(reason) > 20, key


def test_speed_floor_recognizes_task_organized_scene(tmp_path):
    scene = "Middlebury/MiddEval3/trainingH/Motorcycle/im0.png"
    flat = tmp_path / scene
    grouped = tmp_path / "stereo" / scene
    assert speed_check._scene_path(tmp_path, scene) == flat
    grouped.parent.mkdir(parents=True)
    grouped.touch()
    assert speed_check._scene_path(tmp_path, scene) == grouped
    flat.parent.mkdir(parents=True)
    flat.touch()
    assert speed_check._scene_path(tmp_path, scene) == flat


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


def test_supplemental_tracking_row_is_not_scheduled():
    """The fixed FP32 TensorRT comparison uses different arguments from the tracking speed worker."""
    jobs = speed_check.jobs(["cotracker3-online"])
    assert {j["mode"] for j in jobs} == {"portable", "kernels"}
    assert all(j["inputs"] for j in jobs)
    assert all(len(inp["label"].split("x")) == 2 for j in jobs for inp in j["inputs"])


def test_edges_compile_and_tensorrt_use_the_task_standard_input():
    jobs = speed_check.jobs(["hed"], modes=("compile", "tensorrt"))
    assert {j["mode"] for j in jobs} == {"compile", "tensorrt"}
    assert all([inp["label"] for inp in j["inputs"]] == ["1080x1920"] for j in jobs)


def test_timing_export_preserves_clip_scope(tmp_path, capsys):
    """Keep the job key and measured clip shape available to downstream report generation."""
    frames = Namespace(shape=(8, 3, 256, 256))
    queries = Namespace(shape=(16, 3))
    row = {
        "side": "lib",
        "ref": "model",
        "task": "tracking",
        "mode": "tensorrt",
        "input": "256x256",
        "ms": 20,
        **speed_check._tracking_scope("tracking", (frames, queries), exported=True),
    }
    rows = tmp_path / "timings.jsonl"
    rows.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert speed_check.export(rows, "test", "test") == 0
    published = json.loads(capsys.readouterr().out)
    assert published["input"] == "256x256"
    assert published["comparison_input"] == "8x256x256"
    assert (published["frames"], published["queries"]) == (8, 16)
    assert published["backend_part"] == "8 frames with 16 queries"


def test_merge_runs_goldens_for_modes_without_speed_rows(monkeypatch, tmp_path):
    floor = tmp_path / "speed.jsonl"
    floor.write_text(
        json.dumps({"ref": "measured", "mode": "portable", "task": "depth", "input": "480x640"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(speed_check, "FLOOR", floor)
    sel = {"families": {"demo": {"task": "depth", "refs": ["measured@1", "fresh@1", "fresh@2"]}}}
    jobs = merge_check.jobs(sel, [], ["portable", "kernels"])
    assert {(j["ref"], j["mode"], bool(j["inputs"])) for j in jobs} == {
        ("measured", "portable", True),
        ("measured@1", "kernels", False),
        ("fresh@1", "portable", False),
        ("fresh@1", "kernels", False),
        ("fresh@2", "portable", False),
        ("fresh@2", "kernels", False),
    }


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


def test_interrupted_merge_worker_invalidates_previous_pass(monkeypatch, tmp_path):
    """A worker killed after writing a timing needs a fresh golden on the next run."""
    ref, mode = "demo", "portable"
    out = tmp_path / "rows.jsonl"
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"families": {"demo": {"task": "depth", "refs": [ref + "@1"]}}}), encoding="utf-8")
    floor = tmp_path / "speed.jsonl"
    floor.write_text(
        json.dumps({"ref": ref, "mode": mode, "task": "depth", "input": "32x32", "ms": 1.0}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(merge_check, "speed_check", lambda: speed_check)
    monkeypatch.setattr(speed_check, "FLOOR", floor)
    monkeypatch.setattr(speed_check, "files_hash", lambda _: "hash")
    monkeypatch.setattr(speed_check, "_commit", lambda: "revision")
    base = {"side": "lib", "ref": ref, "mode": mode, "task": "depth", "code_hash": "hash"}
    timing = base | {"input": "32x32", "ms": 1.0, "peak_alloc_mb": 10, "peak_reserved_mb": 20}
    golden = base | {"input": "golden", "golden": "pass", "golden_mean": 0.0, "golden_p99": 0.0}
    speed_check._append(out, timing)
    speed_check._append(out, golden)

    def interrupted(cmd, **_kwargs):
        speed_check._append(out, timing)
        return CompletedProcess(cmd, returncode=124)

    monkeypatch.setattr(merge_check.subprocess, "run", interrupted)
    args = Namespace(
        selection=selection,
        out=out,
        family=[],
        modes=[mode],
        golden_only=False,
        no_core=True,
        profile=False,
        force=True,
        prefix="",
        batch_size=1,
    )
    assert merge_check.run(args) == 1
    rows = speed_check.read_rows(out)
    assert rows[-1]["error"] == "worker rc=124"
    (job,) = merge_check.jobs(json.loads(selection.read_text(encoding="utf-8")), [], [mode])
    assert merge_check._done(speed_check, rows[:2], job)
    assert not merge_check._done(speed_check, rows, job)
    assert merge_check._done(speed_check, [*rows, golden], job)


def test_grouped_goldens_retry_only_unfinished_jobs(monkeypatch, tmp_path):
    """A failed group keeps its passing checks and isolates the unfinished checks."""
    out = tmp_path / "rows.jsonl"
    selection = tmp_path / "selection.json"
    refs = ["a@1", "b@1", "c@1"]
    selection.write_text(json.dumps({"families": {"demo": {"task": "depth", "refs": refs}}}), encoding="utf-8")
    monkeypatch.setattr(merge_check, "speed_check", lambda: speed_check)
    monkeypatch.setattr(speed_check, "files_hash", lambda _: "hash")
    monkeypatch.setattr(speed_check, "_commit", lambda: "revision")
    monkeypatch.setattr(
        merge_check, "_golden_sources_by_ref", lambda jobs, _commit: {j["ref"]: ("scope",) for j in jobs}
    )
    attempts = []

    def record(spec, verdict):
        row = {k: spec[k] for k in ("side", "ref", "mode", "task", "golden_source")}
        speed_check._append(
            out,
            row | {"input": "golden", "golden": verdict, "code_hash": "hash", "golden_mean": 0.0, "golden_p99": 0.0},
        )

    def run(cmd, **_kwargs):
        specs = json.loads(cmd[-1])
        if isinstance(specs, list):
            attempts.append(tuple(s["ref"] for s in specs))
            record(specs[0], "pass")
            record(specs[1], "fail")
            return CompletedProcess(cmd, returncode=1)
        attempts.append((specs["ref"],))
        record(specs, "pass")
        return CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(merge_check.subprocess, "run", run)
    args = Namespace(
        selection=selection,
        out=out,
        family=[],
        modes=["portable"],
        golden_only=True,
        no_core=True,
        profile=False,
        force=False,
        prefix="",
        batch_size=3,
    )
    assert merge_check.run(args) == 0
    assert attempts == [tuple(refs), ("b@1",), ("c@1",)]
    assert merge_check.run(args) == 0
    assert len(attempts) == 3


def test_grouped_worker_stops_before_a_failed_golden_affects_other_jobs(monkeypatch, tmp_path):
    import torch  # noqa: PLC0415

    out = tmp_path / "rows.jsonl"
    attempts = []

    def worker(spec):
        attempts.append((spec["ref"], spec["warm_worker"]))
        verdict = "pass" if len(attempts) == 1 else "fail"
        speed_check._append(out, {"input": "golden", "golden": verdict})
        return 0

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(merge_check, "worker", worker)
    specs = [{"ref": ref, "out": str(out)} for ref in ("first", "second", "third")]
    assert merge_check.worker_batch(specs) == 1
    assert attempts == [("first", False), ("second", True)]
