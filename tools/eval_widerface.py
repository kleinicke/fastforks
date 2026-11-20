"""Face detection accuracy: WIDER FACE validation AP (easy / medium / hard), as libfacedetection.train evaluates YuNet.

`FASTFORKS_DATA`/widerface/WIDER_val/images/<event>/<file>.jpg and widerface/gt/wider_{face,easy,medium,hard}_val.mat
(the official evaluation toolkit's ground truth). `--every N`: every N-th image in the ground truth's order (dataset
`widerface-val-everyN`); the AP counts only those images' faces. Each image at its own size (`eval_widerface --mode
origin`: BGR 0..255, zero-padded to multiples of 32), score threshold 0.02 and NMS IoU 0.45 (upstream's evaluation
defaults, not the card's), the AP code of upstream's `yunet_train.tasks.face.evaluation` (scores normalised over the
set, 1000 thresholds, IoU 0.5) on the subset (`FF_YUNET_REPO`: a libfacedetection.train checkout).

Tiers: `upstream` (`build_yunet` + `YuNetPostprocessor`, float32, as `tools/golden/yunet.py`; `FF_YUNET_WEIGHTS`),
`portable32`, `portable`, `kernels` (the library's network and decoding at the evaluation thresholds).

    python tools/eval_widerface.py yunet-n --tiers upstream,portable32,portable,kernels --every 4 --out out/x
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import DATA
from _prefetch import prefetch

ROOT = Path(__file__).resolve().parents[1]
SCORE, NMS = 0.02, 0.45
TIERS = {  # tier -> (mode, dtype, how, column)
    "upstream": (
        "upstream",
        "float32",
        "authors' code (YuNet: build_yunet + YuNetPostprocessor; RetinaFace: facexlib detect_faces), float32",
        "original fp32",
    ),
    "portable32": ("portable32", "float32", None, "portable32"),
    "portable": ("portable", "float16", None, "portable"),
    "kernels": ("kernels", "float16", None, "kernels"),
}


def _golden() -> Any:  # noqa: ANN401 - tools/golden/yunet.py
    sys.path.insert(0, str(ROOT / "tools" / "golden"))
    import yunet  # noqa: PLC0415

    return yunet


def _family(ref: str) -> str:
    from fastforks._core import registry  # noqa: PLC0415

    return next(c.family for c, v in registry.cards("pose").values() if v.id == ref)


def _retinaface(ref: str, tier: str) -> Any:  # noqa: ANN401
    """RetinaFace (`FF_FACEXLIB_REPO`, `FF_RETINAFACE_WEIGHTS`): facexlib `detect_faces`, own size (conf 0.02)."""
    if tier == "upstream":
        sys.path.insert(0, str(ROOT / "tools" / "golden"))
        import retinaface as g  # noqa: PLC0415

        from fastforks._core import registry  # noqa: PLC0415

        card, variant = next((c, v) for c, v in registry.cards("pose").values() if v.id == ref)
        tree = g.reference_tree(Path(os.environ["FF_FACEXLIB_REPO"]), g.REF_COMMIT)
        net = g.build(
            tree,
            Path(os.environ["FF_RETINAFACE_WEIGHTS"]) / variant.weights[0].name,
            {**card.params, **variant.params}["backbone"],
            "cuda",
        ).cuda()

        def up(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            det = torch.from_numpy(
                g.run(net, np.ascontiguousarray(x[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)), SCORE, 0.4)
            ).reshape(-1, 15)
            return det[:, :4], det[:, 4]

        return up
    from fastforks import pose  # noqa: PLC0415
    from fastforks.pose.models.retinaface.model import faces  # noqa: PLC0415

    m = pose.load(
        ref,
        mode="kernels" if tier == "kernels" else "portable",
        device="cuda",
        **({"dtype": "float32"} if tier == "portable32" else {}),
    )

    def lib(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = faces(m._runner.dense(x.flip(1) / 255), *x.shape[-2:], SCORE, 0.4)  # noqa: SLF001 - the evaluation thresholds
        return out["boxes"], out["scores"]

    return lib


def predictor(ref: str, tier: str, tree: Path) -> Any:  # noqa: ANN401
    """`[1, 3, H, W]` BGR 0..255 float CUDA -> (`[n, 4]` x0 y0 x1 y1, `[n]` scores) of `tier`."""
    if _family(ref) == "retinaface":
        return _retinaface(ref, tier)
    if tier == "upstream":
        g = _golden()
        from fastforks._core import registry  # noqa: PLC0415

        variant = next(v for c, v in registry.cards("pose").values() if v.id == ref and c.family == "yunet")
        net = g.build(tree, Path(os.environ["FF_YUNET_WEIGHTS"]) / variant.weights[0].name, ref, "cuda")

        def up(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            det = g.run(tree, net, x, SCORE, NMS)
            return det.boxes, det.scores

        return up
    from fastforks import pose  # noqa: PLC0415
    from fastforks.pose.models.yunet.model import faces  # noqa: PLC0415

    mode = "kernels" if tier == "kernels" else "portable"
    m = pose.load(ref, mode=mode, device="cuda", **({"dtype": "float32"} if tier == "portable32" else {}))

    def lib(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = faces(m._runner.dense(x.flip(1) / 255), SCORE, NMS)  # noqa: SLF001 - the evaluation thresholds
        return out["boxes"], out["scores"]

    return lib


def subset(gt: dict, every: int) -> list[tuple[str, str]]:
    """(event, image name) of every `every`-th image in the ground truth's order."""
    items = [
        (str(e[0][0]), str(f[0][0]))
        for e, files in zip(gt["event_list"], gt["file_list"], strict=True)
        for f in files[0]
    ]
    return items[::every]


def _read_image(root: Path, item: tuple[str, str]) -> tuple[str, str, np.ndarray]:
    event, image = item
    path = root / event / f"{image}.jpg"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        msg = f"cannot decode WIDER FACE image {path}"
        raise ValueError(msg)
    return event, image, bgr


def average_precisions(ev: Any, gt_dir: Path, preds: dict, keep: set) -> dict[str, float]:  # noqa: ANN401
    """Upstream `wider_evaluation` restricted to the images in `keep` (the others count neither faces nor boxes)."""
    from scipy.io import loadmat  # noqa: PLC0415

    preds = ev.norm_score(preds)
    face = loadmat(gt_dir / "wider_face_val.mat")
    out = {}
    for setting in ("easy", "medium", "hard"):
        gt_list = loadmat(gt_dir / f"wider_{setting}_val.mat")["gt_list"]
        count, curve = 0, np.zeros((1000, 2))
        for e, event in enumerate(face["event_list"]):
            name = str(event[0][0])
            for i, f in enumerate(face["file_list"][e][0]):
                image = str(f[0][0])
                if (name, image) not in keep:
                    continue
                pred = preds.get(name, {}).get(image, np.zeros((0, 5), np.float32))
                boxes = face["face_bbx_list"][e][0][i][0].astype(np.float32)
                index = np.asarray(gt_list[e][0][i][0], dtype=np.int64).reshape(-1)
                count += len(index)
                if len(boxes) == 0 or len(pred) == 0:
                    continue
                ignore = np.zeros(boxes.shape[0], dtype=np.int64)
                if len(index):
                    ignore[index - 1] = 1
                recall, proposals = ev.image_eval(pred, boxes, ignore, 0.5)
                curve += ev.img_pr_info(1000, pred, proposals, recall)
        curve = ev.dataset_pr_info(1000, curve, count)
        out[f"ap_{setting}"] = round(float(ev.voc_ap(curve[:, 1], curve[:, 0])), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--tiers", default="portable,kernels")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = DATA / "widerface"
    read = partial(_read_image, data / "WIDER_val" / "images")
    tree = _golden().reference_tree(Path(os.environ["FF_YUNET_REPO"]), _golden().REF_COMMIT)
    sys.path.insert(0, str(tree))
    import yunet_train.tasks.face.evaluation as ev  # noqa: PLC0415  # ty: ignore[unresolved-import] - upstream AP code
    from scipy.io import loadmat  # noqa: PLC0415

    items = subset(loadmat(data / "gt" / "wider_face_val.mat"), a.every)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    for ref in a.refs:
        for tier in a.tiers.split(","):
            mode, dtype, how, column = TIERS[tier]
            fn = predictor(ref, tier, tree)
            fn(torch.rand(1, 3, 480, 640, device="cuda") * 255)
            torch.cuda.reset_peak_memory_stats()
            preds: dict[str, dict[str, np.ndarray]] = {}
            times = []
            with closing(prefetch(read, items, workers=4)) as inputs:
                for event, image, bgr in inputs:
                    x = torch.from_numpy(bgr).cuda().permute(2, 0, 1)[None].float()
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    with torch.inference_mode():
                        boxes, scores = fn(x)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t)
                    b, s = boxes.float().cpu().numpy(), scores.float().cpu().numpy()
                    xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], s], 1).astype(np.float32)
                    preds.setdefault(event, {})[image] = xywh
            metrics = average_precisions(ev, data / "gt", preds, set(items))
            row = {
                "ref": ref, "task": "pose", "tier": tier, "mode": mode, "dtype": dtype, "how": how,
                "commit": commit.stdout.strip(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "command": "python tools/eval_widerface.py " + " ".join(sys.argv[1:]),
                "dataset": "widerface-val" + (f"-every{a.every}" if a.every > 1 else ""), **metrics,
                "n": len(items), "metrics": metrics, "ms_per_image": round(float(np.median(times)) * 1e3, 2),
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20), "column": column,
            }  # fmt: skip
            print(json.dumps(row), flush=True)
            with (a.out / "rows.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            del fn
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
