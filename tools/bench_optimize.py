"""Parity + speed bench of `fastforks._optimize.optimize` on the fast-vision-models classification zoo.

    FASTFORKS_KERNEL_PATH=build/kernels python tools/bench_optimize.py --fvm <fast-vision-models dir> \
        --models resnet50,vit_b16 --tiers upstream,portable,max --out raw.jsonl

The models come from the fast-vision-models zoo builders (torchvision / timm / HF checkpoints, upstream fp32). One
process per call: each model is built once on the CPU and deep-copied per tier. Per tier: b1 and b64 latency (median
after warm-up, synchronized wall clock, as fast-vision-models bench/bench.py), peak allocated / reserved MB (weights
included), first-call time, and top-1 agreement with upstream fp32 on 256 ImageNet val images (every 195th file of the
sorted class folders, as fast-vision-models). Tier `fvm-max` runs the fast-vision-models cuda-kernels tier itself
(reference; its JIT-built kernels must be in the fast-vision-models cache). One JSON line per model and tier.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch

from _data import DATA

IMAGES = 256
STRIDE = 195


def _open_clip_names(path: str) -> None:
    """Provide `open_clip.zero_shot_metadata` (SigLIP 2 class names) from a file, without installing open_clip."""
    spec = importlib.util.spec_from_file_location("open_clip.zero_shot_metadata", path)
    assert spec is not None  # noqa: S101
    assert spec.loader is not None  # noqa: S101
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pkg = types.ModuleType("open_clip")
    pkg.zero_shot_metadata = mod
    sys.modules["open_clip"] = pkg
    sys.modules["open_clip.zero_shot_metadata"] = mod


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _timed(fn: Any, x: torch.Tensor, n: int) -> list[float]:  # noqa: ANN401
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return ts


def _measure(fn: Any, x: torch.Tensor, n: int, warm: int) -> dict[str, Any]:  # noqa: ANN401
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    first = _timed(fn, x, 1)[0]
    _timed(fn, x, warm)
    probe = statistics.median(_timed(fn, x, 2))
    n = max(5, n // 4) if probe > 1.0 else (max(5, n // 2) if probe > 0.2 else n)  # noqa: PLR2004
    med = statistics.median(_timed(fn, x, n))
    return {
        "first_ms": round(first * 1e3, 2),
        "ms": round(med * 1e3, 3),
        "n": n,
        "alloc_mb": torch.cuda.max_memory_allocated() >> 20,
        "res_mb": torch.cuda.max_memory_reserved() >> 20,
    }


def _images(root: str, transform: Any) -> torch.Tensor:  # noqa: ANN401
    from PIL import Image  # noqa: PLC0415

    r = Path(root)
    files = [p for c in sorted(r.iterdir()) for p in sorted(c.iterdir())][::STRIDE][:IMAGES]
    return torch.stack([transform(Image.open(p).convert("RGB")) for p in files])


def _logits(fn: Any, xs: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
    with torch.inference_mode():
        return torch.cat([fn(xs[i : i + 32].cuda()).float().cpu() for i in range(0, len(xs), 32)])


def _build(tier: str, base: torch.nn.Module, zoo: Any, name: str) -> tuple[Any, dict[str, Any]]:  # noqa: ANN401
    if tier == "upstream":
        return copy.deepcopy(base).cuda(), {"backend": "upstream fp32"}
    if tier == "fvm-max":
        return zoo.load(name, "opt").model, {"backend": "fast-vision-models cuda-kernels"}
    from fastforks._optimize import optimize  # noqa: PLC0415

    m = optimize(copy.deepcopy(base), mode=tier, device="cuda")
    return m, {"backend": f"fastforks {tier}", "report": m.report}


def bench_model(name: str, tiers: list[str], zoo: Any, val: str, out: str | None) -> None:  # noqa: ANN401
    """Build `name` once, run every tier, append the JSON records."""
    t0 = time.perf_counter()
    # SigLIP 2 zero-shot: the class text embeddings are computed at load, on the GPU
    loaded = zoo.load(name, "base", device="cuda" if zoo.BY_NAME[name].src == "siglip2" else "cpu")
    base, size = loaded.model.eval().cpu(), loaded.input_size
    xs = _images(val, loaded.transform)
    ref = None
    for tier in tiers:
        _free()
        rec: dict[str, Any] = {"model": name, "tier": tier, "input": list(size)}
        try:
            t = time.perf_counter()
            fn, info = _build(tier, base, zoo, name)
            torch.cuda.synchronize()
            rec["build_s"] = round(time.perf_counter() - t, 3)
            rec.update(info)
            _free()
            g = torch.Generator(device="cuda").manual_seed(0)
            with torch.inference_mode():
                for i, b0 in enumerate((1, 64)):
                    b = b0
                    while True:
                        try:
                            x = torch.randn(b, *size, device="cuda", generator=g)
                            r = _measure(fn, x, 30 if i == 0 else 10, 12 if i == 0 else 3)
                            break
                        except torch.OutOfMemoryError:
                            _free()
                            b //= 2
                    rec[f"b{b}"] = r
                    del x
            out_l = _logits(fn, xs)
            if tier == "upstream":
                ref = out_l
            if ref is not None:
                ok = bool(torch.isfinite(out_l).all())
                agree = float((out_l.argmax(1) == ref.argmax(1)).float().mean()) if ok else 0.0
                rec["top1_agree_pct"] = round(100 * agree, 2)
                rec["logits_rel_l2"] = round(float((out_l - ref).norm() / ref.norm()), 5) if ok else float("nan")
                rec["images"] = len(xs)
            del fn
        except Exception as e:  # noqa: BLE001 - one failing tier must not stop the queue; recorded
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec.update(
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            commit=_git_rev(),
            date=time.strftime("%Y-%m-%d %H:%M"),
            command=" ".join(["python", *sys.argv]),
            model_total_s=round(time.perf_counter() - t0, 1),
        )
        line = json.dumps(rec)
        print(line, flush=True)
        if out:
            with Path(out).open("a") as f:
                f.write(line + "\n")


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "n/a"


def main() -> None:
    """Command line entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fvm", required=True, help="fast-vision-models checkout (zoo builders)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--tiers", default="upstream,portable,max")
    ap.add_argument("--val", default=str(DATA / "classification/imagenet1k_val"))
    ap.add_argument("--open-clip-names", help="open_clip/zero_shot_metadata.py (SigLIP 2 zero-shot class names)")
    ap.add_argument("--out")
    a = ap.parse_args()
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    sys.path.insert(0, a.fvm)
    if a.open_clip_names:
        _open_clip_names(a.open_clip_names)
    from fast_vision_models.classification import zoo  # noqa: PLC0415  # ty: ignore[unresolved-import]

    torch.set_grad_enabled(False)
    for name in a.models.split(","):
        bench_model(name, a.tiers.split(","), zoo, a.val, a.out)


if __name__ == "__main__":
    main()
    os._exit(0)
