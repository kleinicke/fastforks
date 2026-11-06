"""Find, per model family, the smallest square size at which `module()` runs (CPU, float32), as the export traces it.

Usage: `python tools/probe_min_size.py OUT_JSONL [TASK ...]`. One JSON line per family (its smallest-weights variant):
`{"card": ..., "ref": ..., "min": N or null, "tried": {size: "ok" | error}}`. Models whose weights are not local are
skipped. Run it in a RAM-capped unit, as one process loads one model at a time.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

from fastforks._core import registry

LADDER = (8, 14, 16, 28, 32, 42, 56, 64, 96, 112, 128, 160, 224, 256, 384, 512)
TASKS = ("stereo", "depth", "detection", "promptable_segmentation", "multiview")


def _runs(mod, model, s: int) -> str:  # noqa: ANN001 - torch module and fastforks Model
    """`"ok"` if `module()` runs at `s x s` with finite outputs, else the error."""
    try:
        with torch.no_grad():
            out = mod(*model.example_inputs(s, s))
        leaves = [t for t in (out if isinstance(out, (tuple, list)) else [out]) if torch.is_tensor(t)]
        return "ok" if all(torch.isfinite(t.float()).all() for t in leaves) else "non-finite"
    except Exception as e:  # noqa: BLE001 - any failure means "too small"
        return f"{type(e).__name__}: {str(e)[:120]}"


def probe(model) -> tuple[int | None, dict[str, str]]:  # noqa: ANN001 - a loaded fastforks Model
    """Smallest square side that runs with finite outputs (ladder, then bisection below the first hit), and the log.

    Assumes that a size that runs makes every larger size run.
    """
    mod = model.module().eval()
    tried: dict[str, str] = {}
    prev = 0
    for s in LADDER:
        tried[str(s)] = _runs(mod, model, s)
        if tried[str(s)] == "ok":
            lo, hi = prev, s  # lo fails (or 0), hi runs
            while hi - lo > 1:
                mid = (lo + hi) // 2
                tried[str(mid)] = _runs(mod, model, mid)
                lo, hi = (lo, mid) if tried[str(mid)] == "ok" else (mid, hi)
            return hi, tried
        prev = s
    return None, tried


def main() -> None:
    """Probe every family of the given tasks and append the results."""
    out = Path(sys.argv[1])
    tasks = tuple(sys.argv[2:]) or TASKS
    done = {json.loads(line)["card"] for line in out.read_text().splitlines()} if out.exists() else set()
    for task in tasks:
        seen = set()
        for card, _ in registry.cards(task).values():
            if card.name in seen or card.name in done:
                continue
            seen.add(card.name)
            best = min(card.variants, key=lambda v: sum(w.size or 0 for w in v.weights))
            row = {"task": task, "card": card.name, "ref": best.ref, "min": None, "tried": {}}
            try:
                from fastforks._core import registry as r  # noqa: PLC0415

                r.weights(card, best, allow_download=False)
                import fastforks  # noqa: PLC0415

                api = getattr(fastforks, task)
                model = api.load(best.ref, device="cpu", options={"iters": 2} if "iters" in card.options else {})
                row["min"], row["tried"] = probe(model)
                del model
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            gc.collect()
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(row["card"], row["min"], row.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
