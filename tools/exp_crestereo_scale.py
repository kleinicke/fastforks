"""CREStereo accuracy against input scale and iteration count (rows of `docs/data/experiments_resolution.jsonl`).

    python tools/exp_crestereo_scale.py out/cre [--ref crestereo-eth3d] [--limit N]   # rows -> out/cre/rows.jsonl

Datasets, validity masks and thresholds are those of `tools/eval_stereo_depth.py` (kernels mode). The pair is resized
by `scale` (bilinear, back to uint8), the disparity resized back (bilinear) and divided by `scale`; the timed call
includes both resizes. `epe` / `bad` are means over images; `*_small` / `*_large` are pooled over the valid pixels
with GT disparity below / at least 64 px.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import eval_stereo_depth as ev

SETTINGS = ((1.0, 20), (1.5, 20), (2.0, 20), (1.0, 40))  # (scale, iters)
SPLIT = 64.0
NOTE = "split at GT disparity 64 px of the original scale, pooled over pixels; epe / bad are means over images"


def predict(model: Any, left: Any, right: Any, scale: float) -> Any:  # noqa: ANN401
    """Disparity at the input size from a pair resized by `scale`."""
    import torch.nn.functional as F  # noqa: PLC0415

    if scale == 1.0:
        return model(left, right).disparity
    h, w = left.shape[-2:]
    size = (round(h * scale), round(w * scale))
    up = [F.interpolate(x[None].float(), size, mode="bilinear", align_corners=False)[0] for x in (left, right)]
    d = model(*(x.round().clamp(0, 255).byte() for x in up)).disparity
    return F.interpolate(d.reshape(1, 1, *size).float(), (h, w), mode="bilinear", align_corners=False)[0, 0] / scale


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--ref", default="crestereo-eth3d")
    ap.add_argument("--limit", type=int, default=0, help="first N items per dataset (pilot runs)")
    a = ap.parse_args()
    import torch  # noqa: PLC0415

    from fastforks import stereo  # noqa: PLC0415

    a.out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"ref": a.ref, "task": "stereo", "commit": ev._commit(),  # noqa: SLF001
                            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": "kernels",
                            "command": " ".join(["python", "tools/exp_crestereo_scale.py", *sys.argv[1:]])}  # fmt: skip
    first = True
    with (a.out / "rows.jsonl").open("w") as f:
        for iters in dict.fromkeys(i for _, i in SETTINGS):
            model = stereo.load(a.ref, mode="kernels", device="cuda", options={"iters": iters})
            meta["dtype"] = str(model.backend.dtype).removeprefix("torch.")
            for ds in ev.DATASETS["stereo"]:
                thr, vmax = ev.BAD[ds]
                items = ev.stereo_items(ds)[: a.limit or None]
                for scale in [s for s, i in SETTINGS if i == iters]:
                    epe, bad, ms, alloc, reserved, errors = [], [], [], 0, 0, 0
                    pool = {k: torch.zeros(3, dtype=torch.float64) for k in ("small", "large")}  # n, sum epe, n bad
                    for _, ld in items:
                        left, right, gt = ld()
                        xs = [torch.from_numpy(i).permute(2, 0, 1).contiguous().cuda() for i in (left, right)]
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                        try:
                            pred = predict(model, *xs, scale)
                            torch.cuda.synchronize()
                        except Exception as e:  # noqa: BLE001 - counted in n_errors
                            print("ERROR", ds, scale, iters, type(e).__name__, str(e)[:200], flush=True)
                            errors += 1
                            continue
                        if not first:
                            ms.append(1e3 * (time.perf_counter() - t0))
                        first = False
                        alloc = max(alloc, torch.cuda.max_memory_allocated())
                        reserved = max(reserved, torch.cuda.max_memory_reserved())
                        g = torch.from_numpy(gt).cuda()
                        p = torch.nan_to_num(pred.float().reshape(g.shape), nan=0.0, posinf=0.0, neginf=0.0)
                        v = torch.isfinite(g) & (g < vmax)
                        e = (p - g).abs()
                        epe.append(e[v].mean().item())
                        bad.append((e[v] > thr).float().mean().item())
                        for k, part in (("small", v & (g < SPLIT)), ("large", v & (g >= SPLIT))):
                            x = e[part].double()
                            pool[k] += torch.tensor([x.numel(), x.sum().item(), (x > thr).sum().item()])
                    metrics: dict[str, Any] = {"epe": round(sum(epe) / len(epe), 4)}
                    metrics["bad"] = round(100 * sum(bad) / len(bad), 2)
                    for k, (n, s, b) in ((k, v.tolist()) for k, v in pool.items()):
                        metrics |= {f"n_px_{k}": int(n), f"epe_{k}": round(s / n, 4) if n else None,
                                    f"bad_{k}": round(100 * b / n, 2) if n else None}  # fmt: skip
                    out = meta | {
                        "dataset": ds, "setting": {"scale": scale, "iters": iters}, "bad_thr": thr, "n": len(epe),
                        "n_errors": errors, "metrics": metrics, "ms_per_img": round(sum(ms) / len(ms), 2),
                        "peak_alloc_mb": round(alloc / 2**20), "peak_reserved_mb": round(reserved / 2**20),
                        "note": NOTE,
                    }  # fmt: skip
                    f.write(json.dumps(out) + "\n")
                    f.flush()
                    print(ds, scale, iters, metrics, out["ms_per_img"], flush=True)
            del model


if __name__ == "__main__":
    main()
