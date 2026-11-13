"""The accuracy collector: a GPU-runnable port fills its original's columns (tools/acc_collect.py)."""

import json

import pytest

import acc_collect

FP16 = ("upstream16", "upstreamac")


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


@pytest.mark.parametrize("own_fails", ["middH", None])
def test_fp16_original_is_chosen_per_dataset(own_fails):
    """Autocast replaces the authors' fp16 option only on the datasets where that option fails."""
    rows = [{"ref": "m", "tier": t, "dataset": d, "metrics": {"bad": 1.0}} for t in FP16 for d in ("eth3d", "middH")]
    for r in rows:
        if (r["tier"], r["dataset"]) == ("upstream16", own_fails):
            r["error"] = "call: CUDA error: invalid argument"
    picked = acc_collect.choose(rows)
    fp16 = {r["dataset"]: r["tier"] for r in rows if r["mode"] == "upstream-fp16"}
    assert fp16 == {"eth3d": "upstream16", "middH": "upstreamac" if own_fails else "upstream16"}
    assert picked == {"m": "upstreamac" if own_fails else "upstream16"}


def test_text_prompt_rows_are_their_own_dataset(tmp_path):
    """SAM 3 (class-name prompts, mAP) and SAM (point prompts, mIoU) share the images, not the benchmark."""
    rows = [
        {"task": "promptable_segmentation", "dataset": "coco-val2017-every10", "metrics": {"miou_best": 1.0}},
        {"task": "promptable_segmentation", "dataset": "coco-val2017-every10", "metrics": {"box_map": 1.0}},
        {"task": "detection", "dataset": "coco-val2017-every10", "metrics": {"box_map": 1.0}},
    ]
    (tmp_path / "coco_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = [r["dataset"] for r in acc_collect._coco(tmp_path)]
    assert got == ["coco-val2017-every10", "coco-val2017-every10-text", "coco-val2017-every10"]


def test_overlapping_chunks_of_two_sizes_are_refused():
    """A resumed run with another --chunk must not count the images of both chunk sizes."""
    rec = {"model": "m", "tier": "portable", "subset": 10, "seed": 0, "images": 5, "top1_correct": 1,
           "top5_correct": 2, "ms_per_image": 1.0, "alloc_mb": 1, "res_mb": 1}  # fmt: skip
    ok = [rec | {"start": 0, "stop": 5}, rec | {"start": 5, "stop": 10}]
    assert acc_collect._chunks(ok, "classification")[0]["n"] == 10
    with pytest.raises(SystemExit, match="overlapping"):
        acc_collect._chunks([*ok, rec | {"start": 0, "stop": 10, "images": 10}], "classification")
