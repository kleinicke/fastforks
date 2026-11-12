"""COCO val2017 accuracy of the detection and promptable-segmentation models, per tier, on the same images.

    FASTFORKS_KERNEL_PATH=build/kernels python tools/eval_coco.py run --task detection --every 10 --budget 140 \
        --models mask-rcnn-r50-fpn-3x-coco --tiers portable,max --out raw/

    python tools/eval_coco.py summarize --out raw/        # mAP / mIoU table from the raw files

Needs `pycocotools` and `scipy` (not project dependencies; `uv pip install pycocotools scipy` in the dev venv).

Images: `val2017/*.jpg` sorted by file name (= sorted image ids), every `--every`-th (default 10: 500 images); they are
processed in order and the run resumes where the last one stopped (`--budget` seconds per invocation fits one GPU lock).
Each tier and model writes `<out>/<task>__<model>__<tier>.jsonl` (progress rows: images, per-image ms, peak MiB) and
`.dets.json` / `.ious.json` (raw detections / per-instance IoUs, appended per chunk).

Pose (`--task pose`, `person_keypoints_val2017.json`): top-down on the GT person boxes (non-crowd, at least one
labelled keypoint) in one call per image; the 17 COCO body keypoints (the first 17 of a whole-body model) with their
scores as visibility, the mean of the 17 keypoint scores as the person score, to `COCOeval` (keypoints, OKS). Not the
mmpose `CocoMetric` ranking (box score x mean of the keypoint scores above 0.2, then OKS-NMS at 0.9) and no flip test,
so the AP is not the papers' number even on the same boxes; GT boxes give higher AP than the detector boxes of the
papers' headline numbers. Bottom-up models (RTMO) instead run on the image without GT boxes, including images
without labelled people, and are ranked by their predicted person scores after their own NMS. Timing is the call
per image (all its people).

Detection: the model's own call on the RGB uint8 image; boxes (and for Mask R-CNN the pasted masks) of the top 100
detections go to pycocotools `COCOeval` (bbox, and segm for masks); the 80 classes map to the 91 COCO category ids by
name. A one-class detector (the RTMDet person detectors) gets the box AP of its class alone (`person_ap`, dataset
`...-person`), on the same images.
Promptable segmentation (SAM / SAM 2 point prompts): one point per non-crowd GT instance with area >= 1 px, the
interior point farthest from the mask border (Euclidean distance transform on the mask padded by one zero pixel; first
maximum in row-major order); label 1. One call per image with all points as separate prompts. IoU against the GT mask of
(a) the candidate with the best predicted `iou_scores` of the three (`multimask_output=True`) and (b) the single mask
(`multimask_output=False`); mIoU = mean over instances. Timing is per image (detection: the call; SAM: embedding and
prompt call), synchronised, after one untimed warm-up image. Memory: peak allocated / reserved since model build.
Concept segmentation (SAM 3): per image the image embedding once, then one text prompt per COCO category with a
non-crowd GT instance in the image (its COCO name), threshold 0.5 (the default); the top 100 instances per prompt with
boxes and masks go to `COCOeval` (bbox, segm) under that category; timing = embedding + all prompts of the image.

Tiers: fastforks `portable32` (portable, float32), `portable`, `kernels` (card default dtype); the authors' code
(`tools/_original_coco.py`, run in the family's venv) `upstream` (fp32), `upstream16` (the authors' option, else
autocast float16), `upstreamhalf` (`.half()`). Each raw record carries `mode`, `dtype`, `how`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from _data import DATA

CONCEPT_THRESHOLD = 0.5  # SAM 3's default (fastforks CONCEPT_THRESHOLD, transformers' post-processing example)
MIN_LEFT_S = 20  # seconds of budget below which no new model is started
COCO = str(DATA / "coco")


def _images(every: int) -> list[Path]:
    return sorted((Path(COCO) / "val2017").glob("*.jpg"))[::every]


def _gt(task: str = "detection") -> Any:  # noqa: ANN401 - pycocotools.coco.COCO
    from pycocotools.coco import COCO as Coco  # noqa: PLC0415, N811

    name = "person_keypoints" if task == "pose" else "instances"
    return Coco(f"{COCO}/annotations/{name}_val2017.json")


def _f(stem: Path, ext: str) -> Path:
    """`stem` + `ext` (model ids contain dots, so no `with_suffix`)."""
    return stem.parent / (stem.name + ext)


def _done(stem: Path) -> int:
    f = _f(stem, ".jsonl")
    if not f.exists():
        return 0
    rows = [json.loads(line) for line in f.read_text().splitlines()]
    return max((r["stop"] for r in rows if "stop" in r), default=0)


def _point(mask: np.ndarray) -> tuple[float, float]:
    """The interior point farthest from the mask border: first maximum of the distance transform (x, y)."""
    from scipy.ndimage import distance_transform_edt  # noqa: PLC0415

    d = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(x), float(y)


def _sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _load(task: str, ref: str, tier: str) -> Any:  # noqa: ANN401 - a model of either task
    if tier.startswith("upstream"):  # the authors' code (tools/_original_coco.py), run in the family's venv
        import _original_coco  # noqa: PLC0415

        return _original_coco.load(task, ref, tier)
    import fastforks.detection as det  # noqa: PLC0415
    import fastforks.pose as pos  # noqa: PLC0415
    import fastforks.promptable_segmentation as seg  # noqa: PLC0415

    mode, dtype = ("portable", "float32") if tier == "portable32" else (tier, None)
    if task == "pose":
        # DETR OKS evaluation keeps the top 100 query/class pairs, then only the person class.
        # Public inference defaults to threshold 0.5.
        opts = {"score_threshold": 0.0} if ref.startswith("rf-detr-keypoint") else None
        return pos.load(ref, mode=mode, device="cuda", dtype=dtype, options=opts)

    # RT-DETRv2 / D-FINE mAP: the official postprocessor (all 300 queries, no threshold); defaults: 0.05 / 100
    all300 = ref.startswith(("rtdetrv2", "dfine", "deim", "rf-detr", "owlv2", "grounding-dino"))
    opts = {"score_threshold": 0.0, "max_detections": 300} if all300 else None
    if ref.startswith("rtmdet") and ref.endswith("-coco"):  # the reference configs' COCO test settings
        opts = {"score_threshold": 0.001, "max_detections": 300}
    return (det if task == "detection" else seg).load(ref, mode=mode, device="cuda", options=opts, dtype=dtype)


def _cat_map(model: Any, coco: Any) -> list[int]:  # noqa: ANN401
    ids = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    return [ids[n] for n in model.class_names]


def _detect(model: Any, coco: Any, files: list[Path], rec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float]]:  # noqa: ANN401
    from pycocotools import mask as mu  # noqa: PLC0415

    cats = _cat_map(model, coco)
    dets, ms = [], []
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB"))
        t = _sync()
        r = model(img)
        ms.append(1e3 * (_sync() - t))
        k = min(100, len(r.scores))
        b = r.boxes[:k].cpu().numpy()
        s, lab = r.scores[:k].cpu().numpy(), r.labels[:k].cpu().numpy()
        rles = None
        if r.masks is not None:
            m = r.masks[:k].cpu().numpy().astype(np.uint8)
            rles = mu.encode(np.asfortranarray(m.transpose(1, 2, 0)))
        for i in range(k):
            d = {
                "image_id": int(f.stem),
                "category_id": cats[int(lab[i])],
                "bbox": [float(b[i, 0]), float(b[i, 1]), float(b[i, 2] - b[i, 0]), float(b[i, 3] - b[i, 1])],
                "score": float(s[i]),
            }
            if rles is not None:
                d["segmentation"] = {"size": rles[i]["size"], "counts": rles[i]["counts"].decode()}
            dets.append(d)
    rec["masks"] = bool(dets and "segmentation" in dets[0])
    rec["only"] = model.class_names[0] if len(cats) == 1 else None  # a one-class detector: AP of that class alone
    return dets, ms


def _pose(model: Any, coco: Any, files: list[Path], rec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float]]:  # noqa: ANN401
    """Evaluate top-down GT-box or bottom-up pose using the 17 COCO body keypoints."""
    dets, ms = [], []
    rec["pose_gt_boxes"] = getattr(model, "needs_boxes", True)
    if rec.get("model", "").startswith("rf-detr-keypoint"):
        rec["pose_protocol"] = (
            "all images; top 100 query/class pairs, person class only; score_threshold=0; no NMS; "
            "uncertainty-fused person score (trace_alpha=0.2), joint confidence=findability"
        )
    for f in files:
        iid = int(f.stem)
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False)) if a["num_keypoints"] > 0]
        if not anns and rec["pose_gt_boxes"]:
            continue
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                          for a in anns], dtype=np.float32)  # fmt: skip
        img = np.asarray(Image.open(f).convert("RGB"))
        t = _sync()
        r = model(img, boxes=boxes) if rec["pose_gt_boxes"] else model(img)
        ms.append(1e3 * (_sync() - t))
        kp = np.asarray(r.keypoints.float().cpu())[:, :17]  # whole-body models: the body keypoints come first
        ks = np.asarray(r.keypoint_scores.float().cpu())[:, :17]
        scores = ks.mean(axis=1) if rec["pose_gt_boxes"] else np.asarray(r.scores.float().cpu())
        for i in range(len(kp)):
            k = np.concatenate([kp[i], ks[i, :, None]], 1)
            dets.append({"image_id": iid, "category_id": 1, "keypoints": [float(x) for x in k.ravel()],
                         "score": float(scores[i])})  # fmt: skip
    rec["people"] = len(dets) + rec.get("people", 0)
    return dets, ms


def _concept(model: Any, coco: Any, files: list[Path], rec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float]]:  # noqa: ANN401
    """SAM 3: one text prompt per COCO category present in the image (its name), card threshold; boxes + masks."""
    from pycocotools import mask as mu  # noqa: PLC0415

    names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    dets, ms = [], []
    for f in files:
        iid = int(f.stem)
        cats = sorted({a["category_id"] for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False))})
        img = np.asarray(Image.open(f).convert("RGB"))
        t0 = _sync()
        emb = model.get_image_embeddings(img)
        t1 = _sync()
        tp = 0.0
        for c in cats:
            t2 = _sync()
            r = model(emb, text=names[c], threshold=CONCEPT_THRESHOLD)
            tp += _sync() - t2
            k = min(100, len(r.scores))
            b, s = r.boxes[:k].float().cpu().numpy(), r.scores[:k].float().cpu().numpy()
            m = r.masks[:k].cpu().numpy().astype(np.uint8)
            rles = mu.encode(np.asfortranarray(m.transpose(1, 2, 0))) if k else []
            for i in range(k):
                dets.append({  # noqa: PERF401
                    "image_id": iid, "category_id": c, "score": float(s[i]),
                    "bbox": [float(b[i, 0]), float(b[i, 1]), float(b[i, 2] - b[i, 0]), float(b[i, 3] - b[i, 1])],
                    "segmentation": {"size": rles[i]["size"], "counts": rles[i]["counts"].decode()},
                })  # fmt: skip
        ms.append([1e3 * (t1 - t0), 1e3 * tp, len(cats)])
    rec["masks"] = True
    rec["concept"] = True
    return dets, ms


def _prompt(model: Any, coco: Any, files: list[Path], rec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[float]]:  # noqa: ANN401
    if "text" in model.prompts:
        return _concept(model, coco, files, rec)
    if "points" not in model.prompts:
        rec["skipped"] = f"no point prompts (takes {sorted(model.prompts)})"
        return [], []
    rows, ms = [], []
    for f in files:
        iid = int(f.stem)
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False)) if a["area"] > 0]
        gts = [coco.annToMask(a).astype(bool) for a in anns]
        gts = [g for g in gts if g.any()]
        img = np.asarray(Image.open(f).convert("RGB"))
        t0 = _sync()
        emb = model.get_image_embeddings(img)
        t1 = _sync()
        if not gts:
            ms.append([1e3 * (t1 - t0), 0.0])
            continue
        pts = torch.tensor([[_point(g)] for g in gts])  # [P, 1, 2]
        lab = torch.ones(len(gts), 1)
        t2 = _sync()
        multi = model(emb, input_points=pts, input_labels=lab, multimask_output=True)
        t3 = _sync()
        single = model(emb, input_points=pts, input_labels=lab, multimask_output=False)
        t4 = _sync()
        ms.append([1e3 * (t1 - t0), 1e3 * (t3 - t2)])
        g = torch.from_numpy(np.stack(gts)).cuda()
        best = multi.iou_scores.argmax(1)
        pm = multi.masks[torch.arange(len(gts)), best]
        sm = single.masks[:, 0]
        for name, m in (("best", pm), ("single", sm)):
            inter = (m & g).flatten(1).sum(1).float()
            union = (m | g).flatten(1).sum(1).float()
            for j, v in enumerate((inter / union.clamp(min=1)).tolist()):
                rows.append({"image_id": iid, "ann": anns[j]["id"], "kind": name, "iou": v})
        del t4
    return rows, ms


def _run(a: argparse.Namespace, ref: str, tier: str) -> bool:  # noqa: PLR0915
    """Process images of one model and tier until the budget is spent; True if all images are done."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{a.task}__{ref}__{tier}"
    files = _images(a.every)
    start = _done(stem)
    if start >= len(files):
        return True
    deadline = time.perf_counter() + a.budget
    coco = _gt(a.task)
    rec: dict[str, Any] = {"model": ref, "tier": tier, "task": a.task, "every": a.every}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        model = _load(a.task, ref, tier)
    except Exception as e:  # noqa: BLE001 - recorded; the queue goes on
        rec.update(stop=len(files), error=f"{type(e).__name__}: {str(e)[:300]}")
        _append(stem, rec)
        return True
    rec["backend"] = f"{model.backend}"
    be = model.backend
    rec["mode"] = "upstream" if tier.startswith("upstream") else be.mode
    rec["dtype"] = str(be.dtype).removeprefix("torch.")
    rec["how"] = getattr(be, "how", None)
    rec["build_s"] = round(time.perf_counter() - t0, 1)
    fn = {"detection": _detect, "pose": _pose}.get(a.task, _prompt)
    if start == 0:  # warm-up on the first image, untimed
        try:
            fn(model, coco, files[:1], {})
        except Exception as e:  # noqa: BLE001
            rec.update(stop=len(files), error=f"{type(e).__name__}: {str(e)[:300]}")
            _append(stem, rec)
            return True
    pos, items, times = start, [], []
    while pos < len(files) and time.perf_counter() < deadline:
        try:
            r, ms = fn(model, coco, files[pos : pos + a.step], rec)
        except Exception as e:  # noqa: BLE001
            rec.update(stop=len(files), error=f"{type(e).__name__}: {str(e)[:300]}")
            _append(stem, rec)
            return True
        if "skipped" in rec:
            rec["stop"] = len(files)
            _append(stem, rec)
            return True
        items += r
        times += ms
        pos += a.step
    pos = min(pos, len(files))
    rec.update(
        start=start,
        stop=pos,
        images=pos - start,
        ms=[round(m, 3) if not isinstance(m, list) else [round(x, 3) for x in m] for m in times],
        alloc_mb=torch.cuda.max_memory_allocated() >> 20,
        res_mb=torch.cuda.max_memory_reserved() >> 20,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        commit=subprocess.run(
            ["git", "-C", str(Path(__file__).parent.parent), "log", "-1", "--format=%h", "--", "src"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        date=time.strftime("%Y-%m-%d %H:%M"),
        command=" ".join(["python", *sys.argv]),
    )
    ext = ".dets.json" if a.task in ("detection", "pose") or rec.get("concept") else ".ious.json"
    f = _f(stem, ext)
    f.write_text(json.dumps((json.loads(f.read_text()) if f.exists() else []) + items))
    _append(stem, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "ms"}), flush=True)
    return pos >= len(files)


def _append(stem: Path, rec: dict[str, Any]) -> None:
    with _f(stem, ".jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _coco_map(coco: Any, dets: list[dict[str, Any]], kind: str, ids: list[int], only: str = "") -> float:  # noqa: ANN401
    """COCO AP (0.50:0.95, in %) over the images `ids`; `only`: of that one category (else the mean over all)."""
    from pycocotools.cocoeval import COCOeval  # noqa: PLC0415

    if not dets:
        return 0.0
    ev = COCOeval(coco, coco.loadRes(dets), kind)
    ev.params.imgIds = ids
    if only:
        ev.params.catIds = coco.getCatIds(catNms=[only])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return round(100 * float(ev.stats[0]), 2)


def _summarize(a: argparse.Namespace) -> None:  # noqa: PLR0915 - one branch per task
    out = Path(a.out)
    cocos = {}
    ids = [int(f.stem) for f in _images(a.every)]
    lines = ["| task | model | tier | metric | ms/image | MiB alloc / reserved | images |", "|---" * 7 + "|"]
    docs = []
    for jl in sorted(out.glob("*.jsonl")):
        rows = [json.loads(x) for x in jl.read_text().splitlines()]
        stem = jl.parent / jl.name.removesuffix(".jsonl")
        task, ref, tier = stem.name.split("__")
        coco = cocos.setdefault(task, _gt(task))
        last = rows[-1]
        mode = "upstream-fp16" if tier.startswith("upstream") and tier != "upstream" else last.get("mode", tier)
        row: dict[str, Any] = {"ref": ref, "task": task, "tier": tier, "mode": mode,
                               "dtype": last.get("dtype"), "how": last.get("how"),
                               "dataset": f"coco-val2017-every{a.every}" + (("-keypoints-gtbox" if last.get("pose_gt_boxes", True) else "-keypoints") if task == "pose" else "")}  # fmt: skip  # noqa: E501
        if last.get("pose_protocol"):
            row["protocol"] = last["pose_protocol"]
        if err := next((r for r in rows if "error" in r or "skipped" in r), None):
            lines.append(f"| {task} | {ref} | {tier} | skipped: {err.get('error') or err['skipped']} | | | |")
            docs.append(row | {"error": err.get("error") or err["skipped"], "date": err.get("date")})
            continue
        n = sum(r["images"] for r in rows)
        ms = [m for r in rows for m in r["ms"]]
        alloc, res = max(r["alloc_mb"] for r in rows), max(r["res_mb"] for r in rows)
        met: dict[str, Any] = {}
        if task == "pose":
            dets = json.loads(_f(stem, ".dets.json").read_text())
            met["keypoint_ap"] = _coco_map(coco, dets, "keypoints", ids)
            box_source = "GT boxes" if last.get("pose_gt_boxes", True) else "predicted people"
            metric = f"keypoint AP {met['keypoint_ap']} ({box_source}, {len(dets)} people)"
            t = f"{statistics.median(ms):.1f}"
            tm = {"ms": round(statistics.median(ms), 2)}
        elif task == "detection" or rows[0].get("concept"):
            dets = json.loads(_f(stem, ".dets.json").read_text())
            if only := rows[0].get("only"):  # a one-class detector (RTMDet person): the box AP of its class
                row["dataset"] += f"-{only}"
                met[f"{only}_ap"] = _coco_map(coco, dets, "bbox", ids, only)
                metric = f"{only} box AP {met[f'{only}_ap']}"
            else:
                met["box_map"] = _coco_map(coco, dets, "bbox", ids)
                metric = f"box mAP {met['box_map']}"
            if rows[0].get("masks"):
                met["mask_map"] = _coco_map(coco, dets, "segm", ids)
                metric += f", mask mAP {met['mask_map']}"
            if rows[0].get("concept"):
                met["prompts"] = int(sum(m[2] for m in ms))
                t = f"{statistics.median(m[0] for m in ms):.1f} embed + {statistics.median(m[1] for m in ms):.1f} prompts"  # noqa: E501
                tm = {"ms_embed": round(statistics.median(m[0] for m in ms), 2),
                      "ms_prompts": round(statistics.median(m[1] for m in ms), 2)}  # fmt: skip
            else:
                t = f"{statistics.median(ms):.1f}"
                tm = {"ms": round(statistics.median(ms), 2)}
        else:
            ious = json.loads(_f(stem, ".ious.json").read_text())
            mi = {k: 100 * float(np.mean([r["iou"] for r in ious if r["kind"] == k])) for k in ("best", "single")}
            cnt = sum(r["kind"] == "best" for r in ious)
            met = {"miou_best": round(mi["best"], 2), "miou_single": round(mi["single"], 2), "instances": cnt}
            metric = f"mIoU best-of-3 {mi['best']:.2f}, single {mi['single']:.2f} ({cnt} instances)"
            t = f"{statistics.median(m[0] for m in ms):.1f} embed + {statistics.median(m[1] for m in ms):.1f} prompts"
            tm = {"ms_embed": round(statistics.median(m[0] for m in ms), 2),
                  "ms_prompts": round(statistics.median(m[1] for m in ms), 2)}  # fmt: skip
        lines.append(f"| {task} | {ref} | {tier} | {metric} | {t} | {alloc} / {res} | {n} |")
        docs.append(row | {"n": n, "metrics": met, **tm, "peak_alloc_mb": alloc, "peak_reserved_mb": res,
                           "backend": last.get("backend"), "build_s": rows[0].get("build_s")}
                    | {k: last.get(k) for k in ("commit", "gpu", "torch", "date", "command")})  # fmt: skip
    print("\n".join(lines))
    if a.rows:
        Path(a.rows).write_text("".join(json.dumps(d) + "\n" for d in docs))


def main() -> None:
    """Command line entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--task", choices=["detection", "promptable_segmentation", "pose"], default="detection")
    ap.add_argument("--models", default="")
    ap.add_argument("--tiers", default="portable,max")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--step", type=int, default=5, help="images between budget checks")
    ap.add_argument("--budget", type=float, default=140.0, help="seconds per invocation (all models and tiers)")
    ap.add_argument("--per-run", type=float, default=1e9, help="seconds per model and tier (a pilot's probe)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="", help="summarize: also write one jsonl row per raw file here")
    a = ap.parse_args()
    if a.cmd == "summarize":
        _summarize(a)
        return
    t0 = time.perf_counter()
    todo = False
    for ref in a.models.split(","):
        for tier in a.tiers.split(","):
            left = a.budget - (time.perf_counter() - t0)
            if left < MIN_LEFT_S:
                todo = True
                continue
            b = argparse.Namespace(**{**vars(a), "budget": min(left, a.per_run)})
            todo |= not _run(b, ref, tier)
    sys.exit(3 if todo else 0)


if __name__ == "__main__":
    main()
