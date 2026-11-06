"""ADE20K-150 mIoU of fastforks segmentation models and of the transformers models they extend, on the same images.

    FASTFORKS_KERNEL_PATH=build/kernels python tools/eval_segmentation.py --models segformer-b0-ade20k \
        --tiers upstream,portable,max --every 4 --out raw.jsonl

Images: `ADEChallengeData2016/images/validation/*.jpg` sorted, every `--every`-th, then `[--start:--stop]` (chunks fit
one GPU lock; rows of one model and tier add up: the jsonl keeps per-class intersection / union pixel counts).
Labels: the annotation PNGs with 0 = ignore and class `k` = index `k - 1` (mmseg `reduce_zero_label`). Each image goes
in at its own size as float RGB in [0, 1] on the GPU; prediction = argmax of the logits at the image size.

`upstream` is the transformers model itself in float32 (`from_pretrained` at the card's pinned revision) under the
authors' test protocol as written out independently in `tools/golden/seginfer_segmentation.py` (the goldens'
reference); for DINOv2 linear the upstream facebookresearch/dinov2 ViT (`DINOV2` checkout, pinned commit) with the
protocol of `tools/golden/dinov2_linear.py`; for the mmseg classics mmsegmentation 1.2.2 itself (`MMSEG`
site-packages appended to `sys.path`) with the protocol of `tools/golden/mmseg_classics.py`. fastforks tiers run the
model's runner (`segmentation.load(..., mode=tier)`). One JSON line per model, tier and chunk: images, mIoU / aAcc
of the chunk, per-class counts, ms per image, peak allocated / reserved MB.

Tiers of the 2026-11-05 campaign: the authors' code through the bench adapters (`tools/_original.py`, run in the
family's venv: `upstream` fp32, `upstream16` autocast float16, `upstreamhalf` `.half()`; each record has `mode`,
`dtype`, `how`), and fastforks `portable32` (portable, float32), `portable`, `kernels`. `--chunk N --max-seconds S`
runs every model x tier x chunk not yet in `--out` (one model build per model and tier per invocation; exit 3 =
call again).

Sapiens (Goliath body parts, no ADE20K classes) is scored on COCO val2017 persons instead: the images of the every-10th
subset (`tools/eval_coco.py`) with a non-crowd person instance; label = union of the person masks (crowd person regions
ignored); prediction = argmax != 0 (any body part); two classes (background, person), `person_iou` = IoU of class 1.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from _data import DATA

sys.path.insert(0, str(Path(__file__).parent / "golden"))

ADE = str(DATA / "segmentation/ADE20K/ADEChallengeData2016")
DINOV2 = os.environ.get("FASTFORKS_DINOV2_REPO", "")  # facebookresearch/dinov2 checkout (pinned commit)
MMSEG = os.environ.get("FASTFORKS_MMSEG_SITE", "")  # site-packages with mmseg 1.2.2 + mmcv
CLASSES = 150
COCO = DATA / "coco"


def _files(root: str, every: int, start: int, stop: int | None) -> list[tuple[Path, Path]]:
    ims = sorted((Path(root) / "images" / "validation").glob("*.jpg"))[::every][start:stop]
    return [(f, Path(root) / "annotations" / "validation" / f"{f.stem}.png") for f in ims]


class _Set(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, files: list[tuple[Path, Path]]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        im, ann = self.files[i]
        x = torch.from_numpy(np.asarray(Image.open(im).convert("RGB")).copy()).permute(2, 0, 1)
        y = torch.from_numpy(np.asarray(Image.open(ann)).astype(np.int64)) - 1  # 0 (ignore) -> -1
        return x, y


class _Persons(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """COCO val2017 person foreground (1), background (0), crowd persons ignored (-1)."""

    def __init__(self, items: list[tuple[Path, int]]) -> None:
        from pycocotools.coco import COCO as Coco  # noqa: PLC0415, N811

        self.items = items
        self.coco = Coco(str(COCO / "annotations/instances_val2017.json"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        f, iid = self.items[i]
        x = torch.from_numpy(np.asarray(Image.open(f).convert("RGB")).copy()).permute(2, 0, 1)
        y = np.zeros(x.shape[1:], np.int64)
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid, catIds=[1]))
        for a in anns:
            if a["iscrowd"]:
                y[self.coco.annToMask(a).astype(bool) & (y == 0)] = -1
        for a in anns:
            if not a["iscrowd"]:
                y[self.coco.annToMask(a).astype(bool)] = 1
        return x, torch.from_numpy(y)


def _person_items() -> list[tuple[Path, int]]:
    from pycocotools.coco import COCO as Coco  # noqa: PLC0415, N811

    coco = Coco(str(COCO / "annotations/instances_val2017.json"))
    out = []
    for f in sorted((COCO / "val2017").glob("*.jpg"))[::10]:
        iid = int(f.stem)
        if coco.getAnnIds(imgIds=iid, catIds=[1], iscrowd=False):
            out.append((f, iid))
    return out


def _upstream_dinov2(card: Any, variant: Any) -> Any:  # noqa: ANN401 - registry records -> a callable
    import dinov2_linear as gold  # noqa: PLC0415

    vit, head, arch = gold.reference(Path(DINOV2), card, variant)
    vit = vit.cuda()
    head = {k: v.cuda() for k, v in head.items()}

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return gold.whole(vit, head, x, arch)

    return fwd


def _upstream_mmseg(card: Any, variant: Any) -> Any:  # noqa: ANN401 - registry records -> a callable
    import mmseg_classics as gold  # noqa: PLC0415

    model = gold.reference(MMSEG, card, variant).cuda()

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return gold.whole(model, x)

    return fwd


def _upstream(ref: str) -> Any:  # noqa: ANN401 - a callable on [1, 3, H, W] float images
    import seginfer_segmentation as gold  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    card, variant = next(cv for k, cv in registry.cards("segmentation").items() if k.split("@")[0] == ref)
    if card.family == "dinov2_linear":
        return _upstream_dinov2(card, variant)
    if card.family == "mmseg":
        return _upstream_mmseg(card, variant)
    params = {**card.params, **variant.params}
    m = re.match(r"https://huggingface.co/(.+?)/resolve/([0-9a-f]{40})/", variant.weights[-1].url or "")
    assert m, ref  # noqa: S101
    net = getattr(transformers, params["arch"]).from_pretrained(m[1], revision=m[2]).float().eval()
    task = gold._task(m[1], m[2], net, params) if "task_tokens" in params else None  # noqa: SLF001
    net = net.cuda()
    task = None if task is None else task.cuda()

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return gold._whole(net, x, params, task)  # noqa: SLF001

    return fwd


def _build(ref: str, tier: str) -> tuple[Any, dict[str, Any]]:
    """-> (fwd(uint8 CUDA [3, H, W]) -> logits [1, C, H, W], record fields)."""
    t0 = time.perf_counter()
    if tier.startswith("upstream"):
        import _original  # noqa: PLC0415

        o = _original.load("segmentation", ref, tier)
        rec: dict[str, Any] = {"mode": "upstream", "dtype": o.dtype, "how": o.how,
                               "backend": f"original {o.how} ({o.dtype}); {o.note}".rstrip("; ")}  # fmt: skip

        def fwd(x: torch.Tensor) -> torch.Tensor:
            return o(*o.prep((x,))).float()[None]

    elif tier == "upstream-legacy":  # the 2026-10-31 path (fastforks venv), kept for reuse checks
        f0 = _upstream(ref)
        rec = {
            "mode": "upstream",
            "dtype": "float32",
            "how": "fp32",
            "backend": "upstream fp32 (tools/golden reference)",
        }

        def fwd(x: torch.Tensor) -> torch.Tensor:
            return f0(x[None].float().div_(255))

    else:
        from fastforks import segmentation  # noqa: PLC0415

        mode, dtype = ("portable", "float32") if tier == "portable32" else (tier, None)
        model = segmentation.load(ref, mode=mode, device="cuda", dtype=dtype)
        runner = model._runner  # noqa: SLF001 - logits at the image size, argmax here as for upstream
        be = model.backend
        rec = {"mode": be.mode, "dtype": str(be.dtype).removeprefix("torch."), "how": None,
               "backend": f"fastforks {be.mode} {be.dtype}", "kernels": list(be.kernels)}  # fmt: skip

        def fwd(x: torch.Tensor) -> torch.Tensor:
            return runner(x[None].float().div_(255))["logits"]

    rec["build_s"] = round(time.perf_counter() - t0, 1)
    return fwd, rec


def _is_sapiens(ref: str) -> bool:
    return ref.startswith("sapiens")


def _eval(fwd: Any, ds: torch.utils.data.Dataset[Any], classes: int, person: bool) -> dict[str, Any]:  # noqa: ANN401, FBT001
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    inter = torch.zeros(classes, dtype=torch.int64)
    union = torch.zeros(classes, dtype=torch.int64)
    label_area = torch.zeros(classes, dtype=torch.int64)
    t_run, n, nonfinite = 0.0, 0, 0
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=6, prefetch_factor=4)
    with torch.inference_mode():
        for x, y in loader:
            xc = x.cuda(non_blocking=True)
            torch.cuda.synchronize()
            t = time.perf_counter()
            logits = fwd(xc)
            pred = logits.argmax(1)[0]
            torch.cuda.synchronize()
            t_run += time.perf_counter() - t
            nonfinite += int(not bool(torch.isfinite(logits).all()))
            if person:
                pred = (pred != 0).long()
            yc = y.cuda(non_blocking=True)
            valid = yc >= 0
            p, g = pred[valid], yc[valid]
            ai = torch.bincount(p[p == g], minlength=classes).cpu()
            ap = torch.bincount(p, minlength=classes).cpu()
            ag = torch.bincount(g, minlength=classes).cpu()
            inter += ai
            union += ap + ag - ai
            label_area += ag
            n += 1
    seen = union > 0
    rec: dict[str, Any] = {
        "images": n,
        "miou": round(100 * float((inter[seen] / union[seen]).mean()), 3),
        "aacc": round(100 * float(inter.sum() / label_area.sum()), 3),
        "intersect": inter.tolist(),
        "union": union.tolist(),
        "label_area": label_area.tolist(),
        "ms_per_image": round(1e3 * t_run / n, 3),
        "alloc_mb": torch.cuda.max_memory_allocated() >> 20,
        "res_mb": torch.cuda.max_memory_reserved() >> 20,
        "nonfinite_images": nonfinite,
    }
    if person:
        rec["person_iou"] = round(100 * float(inter[1] / union[1].clamp(min=1)), 3)
    return rec


def _run(ref: str, tier: str, files: list[tuple[Path, Path]]) -> dict[str, Any]:
    torch.cuda.empty_cache()
    fwd, rec = _build(ref, tier)
    if _is_sapiens(ref):
        return rec | _eval(fwd, _Persons(_person_items()), 2, person=True)
    return rec | _eval(fwd, _Set(files), CLASSES, person=False)


def _chunked(a: argparse.Namespace) -> int:  # noqa: C901
    """`--chunk`: every model x tier x chunk of the selection not yet in `--out`, one record each; 3 = budget hit."""
    t0 = time.perf_counter()
    ade = _files(a.ade, a.every, 0, None)
    out = Path(a.out)
    done = set()
    if out.exists():
        for r in map(json.loads, out.read_text().splitlines()):
            if "error" not in r:  # failed chunks are retried by the next call
                done.add((r["model"], r["tier"], r["start"], r["stop"]))
    for ref in a.models.split(","):
        person = _is_sapiens(ref)
        items: list[Any] = _person_items() if person else ade
        items = items[: a.stop]  # --stop caps the selection (pilots); chunk bounds stay those of the full run
        for tier in a.tiers.split(","):
            built = None
            for s0 in range(0, len(items), a.chunk):
                s1 = min(s0 + a.chunk, len(items))
                if (ref, tier, s0, s1) in done:
                    continue
                if time.perf_counter() - t0 > a.max_seconds:
                    return 3
                rec: dict[str, Any] = {"model": ref, "tier": tier, "every": a.every, "start": s0, "stop": s1,
                                       "dataset": "coco-val2017-persons-every10" if person else "ade20k-val"}  # fmt: skip  # noqa: E501
                try:
                    if built is None:
                        torch.cuda.empty_cache()
                        built = _build(ref, tier)
                    fwd, base = built
                    ds = _Persons(items[s0:s1]) if person else _Set(items[s0:s1])
                    rec.update(base | _eval(fwd, ds, 2 if person else CLASSES, person))
                except Exception as e:  # noqa: BLE001 - recorded; the queue goes on
                    rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                    if "out of memory" in str(e).lower():
                        rec["error"] = f"OOM, {rec['error']}"
                rec.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, commit=_commit(),
                           date=time.strftime("%Y-%m-%d %H:%M"), command=" ".join(["python", *sys.argv]))  # fmt: skip
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(ref, tier, s0, s1, rec.get("miou", rec.get("error")), rec.get("ms_per_image"), flush=True)
                if "error" in rec:
                    break
            del built
    return 0


def _commit() -> str:
    """The last library commit (src/) of this checkout."""
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(root), "log", "-1", "--format=%h", "--", "src"], capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> None:
    """Command line entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tiers", default="upstream,portable,max")
    ap.add_argument("--ade", default=ADE)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--chunk", type=int, default=0, help="chunks of N images, skip chunks in --out")
    ap.add_argument("--max-seconds", type=float, default=1e9, help="with --chunk: exit 3 (call again) after this")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.chunk:
        sys.exit(_chunked(a))
    files = _files(a.ade, a.every, a.start, a.stop)
    rev = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    for ref in a.models.split(","):
        for tier in a.tiers.split(","):
            rec: dict[str, Any] = {"model": ref, "tier": tier, "every": a.every, "start": a.start, "stop": a.stop}
            try:
                rec.update(_run(ref, tier, files))
            except Exception as e:  # noqa: BLE001 - recorded; the queue goes on
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rec.update(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                commit=rev,
                date=time.strftime("%Y-%m-%d %H:%M"),
                command=" ".join(["python", *sys.argv]),
            )
            line = json.dumps(rec)
            print(
                json.dumps({k: v for k, v in rec.items() if k not in {"intersect", "union", "label_area"}}), flush=True
            )
            if a.out:
                with Path(a.out).open("a") as f:
                    f.write(line + "\n")


if __name__ == "__main__":
    main()
