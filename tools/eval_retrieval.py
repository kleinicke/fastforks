"""Retrieval accuracy: visual place recognition on St Lucia (recall@1 / @5 at 25 m).

`FASTFORKS_DATA`/retrieval/st_lucia/images/test/{database,queries} (the gmberton/VPR-datasets-downloader layout: 1549
database and 1464 query images, 640 x 480, UTM east / north in the file names `@east@north@...`), sorted by name;
`--every 5`: every 5th query against the full database (dataset `st-lucia-every5`). Every image at its own size, one
call each; the descriptors of the queries against all database descriptors with `fastforks.retrieval.top_k`
(cosine similarity, exact). A query counts as found at k when one of its k nearest database images is within 25 m.

Tiers: `upstream` / `upstream16` / `upstreamhalf` (the authors' model code with the README's pre-processing, the
family's `tools/golden/<family>.py`: fp32 / `torch.autocast` float16 / `half()`, as `tools/_original.py`; neither
author ships a precision option; `FF_<FAMILY>_REPO` = the upstream checkout), `portable32`, `portable`, `kernels`.

    python tools/eval_retrieval.py megaloc --tiers upstream,portable,kernels --every 5 --out results/raw/x
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RADIUS = 25.0  # metres
KS = (1, 5)
TIERS = {  # tier -> (mode, dtype, how, column)
    "upstream": ("upstream", "float32", "fp32", "original fp32"),
    "upstream16": ("upstream-fp16", "float16", "autocast", "original fp16"),
    "upstreamhalf": ("upstreamhalf", "float16", "half", "original half"),
    "portable32": ("portable32", "float32", None, "portable32"),
    "portable": ("portable", "float16", None, "portable"),
    "kernels": ("kernels", "float16", None, "kernels"),
}


def images(split: str, every: int = 1) -> tuple[list[Path], np.ndarray]:
    """Paths of `split` (`database` / `queries`) sorted by name, every `every`-th, and their UTM `[N, 2]`."""
    root = Path(os.environ["FASTFORKS_DATA"]) / "retrieval" / "st_lucia" / "images" / "test" / split
    paths = sorted(root.glob("*.jpg"))[::every]
    return paths, np.array([[float(v) for v in p.name.split("@")[1:3]] for p in paths])


def read(p: Path) -> np.ndarray:
    """RGB uint8."""
    return np.ascontiguousarray(cv2.imread(str(p))[..., ::-1])


def predictor(ref: str, tier: str) -> Any:  # noqa: ANN401
    """`[1, 3, H, W]` float CUDA in [0, 1] -> descriptor `[D]` of `tier`."""
    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve("retrieval", ref)
    if tier.startswith("upstream"):
        sys.path.insert(0, str(ROOT / "tools" / "golden"))
        g = importlib.import_module(card.family)
        (weights,) = registry.weights(card, variant)
        net = g.network(Path(os.environ[f"FF_{card.family.upper()}_REPO"]), Path(weights)).cuda()
        if tier == "upstreamhalf":
            net = net.half()

        def call(x: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=tier == "upstream16"):
                return g.predict(net, x)[0]

        return call
    from fastforks import retrieval  # noqa: PLC0415

    mode = "kernels" if tier == "kernels" else "portable"
    m = retrieval.load(ref, mode=mode, device="cuda", **({"dtype": "float32"} if tier == "portable32" else {}))
    return lambda x: m(x).descriptor[0]


def describe(fn: Any, paths: list[Path], times: list[float]) -> torch.Tensor:  # noqa: ANN401
    """Descriptors `[N, D]` float32 of `paths`, one call each (timed, decoding overlapped)."""
    out = []
    with ThreadPoolExecutor(4) as pool:
        for img in pool.map(read, paths):
            x = torch.from_numpy(img).cuda().permute(2, 0, 1)[None].float() / 255
            torch.cuda.synchronize()
            t = time.perf_counter()
            d = fn(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            out.append(d.float())
    return torch.stack(out)


def recalls(q: torch.Tensor, db: torch.Tensor, q_utm: np.ndarray, db_utm: np.ndarray) -> dict[str, float]:
    """Recall@k (%) of the queries: one of the k nearest database descriptors within `RADIUS`."""
    from fastforks.retrieval import top_k  # noqa: PLC0415

    _, idx = top_k(q, db, k=max(KS))
    dist = np.linalg.norm(db_utm[idx.cpu().numpy()] - q_utm[:, None], axis=-1)  # [Q, k]
    return {f"r{k}": round(float((dist[:, :k] <= RADIUS).any(1).mean()) * 100, 2) for k in KS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--tiers", default="portable,kernels")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    db_paths, db_utm = images("database")
    q_paths, q_utm = images("queries", a.every)
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    )
    dataset = f"st-lucia-every{a.every}" if a.every > 1 else "st-lucia"
    for ref in a.refs:
        for tier in a.tiers.split(","):
            mode, dtype, how, column = TIERS[tier]
            fn = predictor(ref, tier)
            for _ in range(3):
                fn(torch.rand(1, 3, 480, 640, device="cuda"))
            torch.cuda.reset_peak_memory_stats()
            times: list[float] = []
            db = describe(fn, db_paths, times)
            q = describe(fn, q_paths, times)
            metrics = recalls(q, db, q_utm, db_utm)
            row = {
                "ref": ref, "task": "retrieval", "tier": tier, "mode": mode, "dtype": dtype, "how": how,
                "commit": commit.stdout.strip(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "command": "python tools/eval_retrieval.py " + " ".join(sys.argv[1:]),
                "dataset": dataset, "n_images": len(db_paths) + len(q_paths), **metrics, "n": len(q_paths),
                "metrics": metrics, "ms_per_img": round(float(np.median(times)) * 1e3, 2),
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
