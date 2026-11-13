"""MOT17 train: fastforks `tracking.ByteTracker` vs the official ByteTrack on the same detections, TrackEval metrics.

    FASTFORKS_KERNEL_PATH=build/kernels python tools/eval_mot.py detect --model dfine-x-coco --out dets/
    BYTETRACK_SRC=... python tools/eval_mot.py track --dets dets/ --out mot/   # MOT txt per tracker + compare.json
    TRACKEVAL_SRC=... python tools/eval_mot.py score --out mot/                # HOTA / DetA / AssA / MOTA / IDF1

Data: `$FASTFORKS_DATA/mot17/det/train/MOT17-XX/img1` (frames of MOTChallenge `MOT17Det.zip`) and a TrackEval
ground-truth folder `--gt` (`MOT17-train/MOT17-XX-FRCNN/{gt/gt.txt,seqinfo.ini}` of `MOT17Labels.zip`,
`seqmaps/MOT17-train.txt`).
`detect` keeps the COCO `person` class (label 0) above score 0.05 and writes `[frame, x0, y0, x1, y1, score]` rows per
sequence. `track` runs both trackers on the same rows with the MOT17 defaults (track_thresh 0.6, buffer 30, match 0.9),
counts frames whose track ids differ and the largest box difference, times one `update` per frame (median), and
writes the MOT txt after the reference output filter (area > 100, not w / h > 1.6). The reference (`$BYTETRACK_SRC`,
an ifzhang/ByteTrack checkout; it needs `lap` and `cython_bbox`, or shims of them) is imported as
`yolox.tracker.byte_tracker`. `score` evaluates `<out>/{ref,ours}/data` with TrackEval (`$TRACKEVAL_SRC`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

FRAMES = Path(os.environ.get("FASTFORKS_DATA", "data")) / "mot17/det/train"
MIN_AREA, MAX_ASPECT = 100, 1.6  # the reference output filter
PARAMS = {"track_thresh": 0.6, "track_buffer": 30, "match_thresh": 0.9}


def detect(model: str, out: Path) -> None:
    """Person detections of every train frame -> `<out>/<seq>.npy`."""
    from fastforks import detection  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    m = detection.load(model, options={"score_threshold": 0.05, "max_detections": 300})
    for seq in sorted(p.name for p in FRAMES.iterdir()):
        rows = []
        for k, f in enumerate(sorted((FRAMES / seq / "img1").glob("*.jpg")), 1):
            d = m(str(f))
            d = d[d.labels == 0].cpu()
            box, score = d.boxes.double().numpy(), d.scores.double().numpy()
            rows.append(np.concatenate([np.full((len(score), 1), k), box, score[:, None]], 1))
        np.save(out / f"{seq}.npy", np.concatenate(rows))
        print(seq, len(rows), flush=True)


def _reference(params: dict) -> object | None:
    """The official BYTETracker (fresh ids), or None without `$BYTETRACK_SRC`."""
    if "BYTETRACK_SRC" not in os.environ:
        return None
    np.float = float  # type: ignore[attr-defined]  # noqa: NPY001  # the reference uses the removed alias
    sys.path.insert(0, os.environ["BYTETRACK_SRC"])
    from yolox.tracker.basetrack import BaseTrack  # noqa: PLC0415  # ty: ignore[unresolved-import]
    from yolox.tracker.byte_tracker import BYTETracker  # noqa: PLC0415  # ty: ignore[unresolved-import]

    BaseTrack._count = 0  # noqa: SLF001  # fresh ids per sequence
    return BYTETracker(types.SimpleNamespace(mot20=False, **params), frame_rate=30)


def track(dets: Path, out: Path) -> None:  # noqa: C901, PLR0912
    """Both trackers on the same detections -> MOT txt + `<out>/compare.json`."""
    from fastforks.tracking import ByteTracker  # noqa: PLC0415

    res = {}
    for f in sorted(dets.glob("MOT17-*.npy")):
        seq, d = f.stem, np.load(f)
        n = len(list((FRAMES / seq / "img1").glob("*.jpg")))
        trackers = {"ours": ByteTracker(**PARAMS), "ref": _reference(PARAMS)}
        lines: dict[str, list[str]] = {k: [] for k, v in trackers.items() if v is not None}
        times: dict[str, list[float]] = {k: [] for k in lines}
        mismatch, maxdiff = 0, 0.0
        for k in range(1, n + 1):
            det = d[d[:, 0] == k][:, 1:6]
            got = {}
            for name in lines:
                t0 = time.perf_counter()
                if name == "ours":
                    o = trackers[name].update(det.copy())
                    got[name] = sorted(zip(o.ids.tolist(), *o.boxes.double().numpy().T.tolist(), strict=True))
                else:
                    r = trackers[name].update(det.copy(), (1, 1), (1, 1))  # type: ignore[attr-defined]
                    got[name] = sorted((t.track_id, *t.tlbr) for t in r)
                times[name].append(time.perf_counter() - t0)
            if "ref" in got:
                if [x[0] for x in got["ref"]] != [x[0] for x in got["ours"]]:
                    mismatch += 1
                else:
                    for a, b in zip(got["ref"], got["ours"], strict=True):
                        maxdiff = max(maxdiff, *(abs(a[i] - b[i]) for i in range(1, 5)))
            for name, rows in got.items():
                for tid, x0, y0, x1, y1 in rows:
                    w, h = x1 - x0, y1 - y0
                    if w * h > MIN_AREA and not w / h > MAX_ASPECT:
                        lines[name].append(f"{k},{tid},{x0:.2f},{y0:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        for name, rows in lines.items():
            (out / name / "data").mkdir(parents=True, exist_ok=True)
            (out / name / "data" / f"{seq}-FRCNN.txt").write_text("".join(rows))
        res[seq] = {"frames": n, "id_mismatch_frames": mismatch, "max_box_diff": maxdiff}
        res[seq].update({f"{k}_ms": 1e3 * float(np.median(v)) for k, v in times.items()})
        print(seq, res[seq], flush=True)
    (out / "compare.json").write_text(json.dumps(res, indent=1))


def score(out: Path, gt: Path) -> None:
    """TrackEval HOTA / CLEAR / Identity of `<out>/{ref,ours}` -> `<out>/metrics.json`."""
    import scipy.optimize  # noqa: F401, PLC0415  # before the numpy aliases TrackEval needs

    for name, t in (("float", float), ("int", int), ("bool", bool)):
        setattr(np, name, t)
    sys.path.insert(0, os.environ["TRACKEVAL_SRC"])
    import trackeval  # noqa: PLC0415  # ty: ignore[unresolved-import]

    names = [n for n in ("ref", "ours") if (out / n).exists()]
    quiet = {"PRINT_CONFIG": False}
    ev = trackeval.Evaluator(
        {"USE_PARALLEL": False, "PRINT_RESULTS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False}
        | {"PLOT_CURVES": False, "TIME_PROGRESS": False}
        | quiet
    )
    ds = trackeval.datasets.MotChallenge2DBox(
        {"GT_FOLDER": str(gt / "MOT17-train"), "SEQMAP_FILE": str(gt / "seqmaps/MOT17-train.txt")}
        | {"TRACKERS_FOLDER": str(out), "TRACKER_SUB_FOLDER": "data", "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train"}
        | {"TRACKERS_TO_EVAL": names, "SKIP_SPLIT_FOL": True}
        | quiet
    )
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(quiet), trackeval.metrics.Identity(quiet)]
    res, _ = ev.evaluate([ds], metrics)
    table = {}
    for n in names:
        c = res["MotChallenge2DBox"][n]["COMBINED_SEQ"]["pedestrian"]
        hota = {k: 100 * float(np.mean(c["HOTA"][k])) for k in ("HOTA", "DetA", "AssA")}
        table[n] = hota | {"MOTA": 100 * c["CLEAR"]["MOTA"], "IDF1": 100 * c["Identity"]["IDF1"]}
        table[n]["IDSW"] = int(c["CLEAR"]["IDSW"])
    print(json.dumps(table, indent=1))
    (out / "metrics.json").write_text(json.dumps(table, indent=1))


def main() -> None:
    """Command line (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("detect", "track", "score"))
    p.add_argument("--model", default="dfine-x-coco")
    p.add_argument("--dets", type=Path)
    p.add_argument("--gt", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.step == "detect":
        detect(a.model, a.out)
    elif a.step == "track":
        track(a.dets, a.out)
    else:
        score(a.out, a.gt)


if __name__ == "__main__":
    main()
