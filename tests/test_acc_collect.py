"""The accuracy collector: a GPU-runnable port fills its original's columns (tools/acc_collect.py)."""

import json

import acc_collect


def _row(ref: str, tier: str) -> dict:
    return {"ref": ref, "tier": tier, "metrics": {"bad": 1.0}}


def test_port_fills_the_original_columns(tmp_path):
    rows = [_row("crestereo-eth3d", t) for t in ("upstream", "upstream16", "portable")]
    rows += [_row("crestereo-eth3d-pt", t) for t in ("upstream", "upstream16")]
    picked = acc_collect.choose(rows)
    cols = {(r["ref"], r["tier"]): r["column"] for r in rows}
    assert cols == {
        ("crestereo-eth3d", "upstream-cpu"): None,
        ("crestereo-eth3d", "upstream16-cpu"): None,
        ("crestereo-eth3d", "portable"): "portable",
        ("crestereo-eth3d", "upstream"): "original fp32",
        ("crestereo-eth3d", "upstream16"): "original fp16",
    }
    (tmp_path / "speed").mkdir()
    rec = {"ref": "crestereo-eth3d-pt", "task": "stereo", "mode": "upstream", "input": "375x1242", "ms": 1.0}
    lines = [json.dumps(rec | {"tier": t}) for t in ("upstream", "upstream16")]
    (tmp_path / "speed/foreign.jsonl").write_text("\n".join(lines) + "\n")
    sp = acc_collect.speed_rows(tmp_path, {"stereo": picked})
    assert [(r["ref"], r["tier"], r["mode"]) for r in sp] == [
        ("crestereo-eth3d", "upstream", "upstream-default"),
        ("crestereo-eth3d", "upstream16", "upstream"),
    ]
