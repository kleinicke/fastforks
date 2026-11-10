"""Depth Anything 3 accuracy against the network's `long_side` (rows of `docs/data/experiments_resolution.jsonl`).

    python tools/exp_da3_longside.py out/da3 [--ref da3mono-large] [--limit N]   # rows -> out/da3/rows.jsonl

Datasets, alignment, timing and base metrics are those of `tools/eval_stereo_depth.py` (kernels mode). One model is
loaded and its `long_side` changed between calls; the run first checks on one image that this gives the same output
as a model loaded with that `long_side`. `"native"` sets `long_side` to the longer side of each image.

Edge metrics, at the GT size on the aligned prediction: a jump is a pixel whose depth differs by more than 10 % from
a valid 4-neighbour. `edge_band_absrel` = AbsRel of the valid pixels within 2 px of a GT jump; `edge_prec` /
`edge_rec` = predicted jumps within 1 px of a GT jump / GT jumps within 1 px of a predicted jump (predicted jumps
counted only within 1 px of valid GT), `edge_f1` their harmonic mean. Each is the mean over the images with GT jumps.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import eval_stereo_depth as ev

SETTINGS = (504, 756, 1008, "native")
JUMP, BAND, TOL = 0.10, 2, 1
NOTE = "edge metrics: see the script (GT depth jump > 10 % between 4-neighbours; band 2 px; F1 tolerance 1 px)"


def set_long_side(model: Any, value: int) -> None:  # noqa: ANN401
    """Change `long_side` of a loaded model (the runner reads it from its context on every call)."""
    r = model._runner  # noqa: SLF001 - an experiment: the option is fixed at load time in the public API
    r.ctx = dataclasses.replace(r.ctx, params=dataclasses.replace(r.ctx.params, long_side=value))


def _jumps(d: Any, valid: Any) -> Any:  # noqa: ANN401
    """Pixels whose depth differs by more than `JUMP` (relative to the smaller) from a valid 4-neighbour."""
    import torch  # noqa: PLC0415

    out = torch.zeros_like(valid)
    for dim in (0, 1):
        n = d.shape[dim] - 1
        a, b = d.narrow(dim, 0, n), d.narrow(dim, 1, n)
        j = ((a - b).abs() > JUMP * torch.minimum(a, b)) & valid.narrow(dim, 0, n) & valid.narrow(dim, 1, n)
        out.narrow(dim, 0, n).logical_or_(j)
        out.narrow(dim, 1, n).logical_or_(j)
    return out


def _dilate(m: Any, r: int) -> Any:  # noqa: ANN401
    import torch.nn.functional as F  # noqa: PLC0415

    return F.max_pool2d(m[None, None].float(), 2 * r + 1, 1, r)[0, 0] > 0


def edges(pred: Any, space: str, gt: Any) -> dict[str, float]:  # noqa: ANN401
    """The edge metrics of one image (empty without GT jumps)."""
    import torch  # noqa: PLC0415

    g = torch.from_numpy(gt).cuda()
    p, _ = ev._hw(pred, tuple(g.shape))  # noqa: SLF001
    fin = torch.isfinite(p)
    p = torch.where(fin, p, torch.zeros_like(p))
    m = torch.isfinite(g) & (g > 0) & fin
    a = ev._align(p, g, m, space)  # noqa: SLF001
    gj = _jumps(g, m)
    if not gj.any():
        return {}
    band = _dilate(gj, BAND) & m
    pj = _jumps(a, torch.ones_like(m)) & _dilate(m, TOL)
    prec = (pj & _dilate(gj, TOL)).sum().item() / max(int(pj.sum()), 1)
    rec = (gj & _dilate(pj, TOL)).sum().item() / int(gj.sum())
    return {
        "edge_band_absrel": ((a - g).abs() / g)[band].mean().item(),
        "edge_f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "edge_prec": prec,
        "edge_rec": rec,
    }


def check(ref: str, model: Any, image: Any) -> None:  # noqa: ANN401
    """A model loaded with `long_side=756` and one changed to 756 after loading give the same output."""
    import torch  # noqa: PLC0415

    from fastforks import depth  # noqa: PLC0415

    other = depth.load(ref, mode="kernels", device="cuda", options={"long_side": 756})
    set_long_side(model, 756)
    a, b = (r.inverse_depth if r.inverse_depth is not None else r.depth for r in (model(image), other(image)))
    if not torch.equal(a, b):
        msg = f"long_side changed after loading differs from long_side at load: max {(a - b).abs().max().item():.3g}"
        raise SystemExit(msg)
    print("check: long_side=756 at load == changed after loading (bit-equal)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--ref", default="da3mono-large")
    ap.add_argument("--limit", type=int, default=0, help="first N items per dataset (pilot runs)")
    a = ap.parse_args()
    import torch  # noqa: PLC0415

    pr = ev.Predictor("depth", a.ref, "kernels")
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {"ref": a.ref, "task": "depth", "commit": ev._commit(), "gpu": torch.cuda.get_device_name(),  # noqa: SLF001
            "torch": torch.__version__, "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": pr.mode,
            "dtype": pr.dtype, "command": " ".join(["python", "tools/exp_da3_longside.py", *sys.argv[1:]])}  # fmt: skip
    check(a.ref, pr.model, pr.prep([ev.depth_items("ibims")[0][1]()[0]])[0])
    first = True
    with (a.out / "rows.jsonl").open("w") as f:
        for ds in ev.DATASETS["depth"]:
            items = ev.depth_items(ds)[: a.limit or None]
            for setting in SETTINGS:
                rows, sides = [], set()
                for _, ld in items:
                    image, gt = ld()
                    side = max(image.shape[:2]) if setting == "native" else setting
                    sides.add(int(side))
                    set_long_side(pr.model, int(side))
                    args = pr.prep([image])
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    pred, space = pr.call(args)
                    torch.cuda.synchronize()
                    row = {"ms": 1e3 * (time.perf_counter() - t0), "first": first}
                    row |= {"alloc": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
                    row |= ev._item(pred, space, gt, ds, "depth", "relative") | edges(pred, space, gt)  # noqa: SLF001
                    rows.append(row)
                    first = False

                def mean(key: str, n: int, rows: list[dict[str, Any]] = rows) -> float | None:
                    v = [r[key] for r in rows if key in r and not (key == "ms" and r["first"])]
                    return round(sum(v) / len(v), n) if v else None

                keys = ("rel_absrel", "rel_d1", "edge_band_absrel", "edge_f1", "edge_prec", "edge_rec")
                out = meta | {
                    "dataset": ds, "setting": {"long_side": setting},
                    "long_side_values": sorted(sides) if setting == "native" else None, "n": len(rows),
                    "metrics": {k: mean(k, 4) for k in keys}, "ms_per_img": mean("ms", 2),
                    "peak_alloc_mb": round(max(r["alloc"] for r in rows) / 2**20),
                    "peak_reserved_mb": round(max(r["reserved"] for r in rows) / 2**20), "note": NOTE,
                }  # fmt: skip
                f.write(json.dumps(out) + "\n")
                f.flush()
                print(ds, setting, out["metrics"], out["ms_per_img"], flush=True)


if __name__ == "__main__":
    main()
