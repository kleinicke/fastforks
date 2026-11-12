"""Speed + peak memory of the authors' original code per precision tier (`tools/_original.py`), bench_upstream inputs.

    VENV/python tools/speed_original.py TASK REF[,REF...] --tiers upstream,upstream16,upstreamhalf --out rows.jsonl

Same inputs and timing as `bench/bench_upstream.py run` (its `inputs(ref)` / `inputs_for(task)`, `measure`: median of
`--runs` after `--warmup` calls, CUDA-synchronized; peak_first = first call after a reset; peak = over all calls), but
one row per tier: `mode` "upstream" with `dtype` and `how` (`fp32`, `authors flag`, `autocast`, `half`) of the tier as
the accuracy evals run it, so the docs' "original fp32" / "original fp16" columns time exactly the measured code path.
`commit` = the card's upstream pin, `ff_commit` = the fastforks commit (`git log -1 -- src`). Resumable: (ref, tier,
input) rows already in `--out` are skipped; exit 3 when `--max-seconds` is spent (call again). Failures are rows with
`error` (OOM prefixed "OOM, ").
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

sys.path.insert(0, str(Path(__file__).parent))
import _original


def _commit() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(root), "log", "-1", "--format=%h", "--", "src"], capture_output=True, text=True, check=False
    ).stdout.strip()


class _Features:
    """eval_features' original builder as an `Original`-like callable (prep = identity)."""

    def __init__(self, ref: str, tier: str) -> None:
        import eval_features  # noqa: PLC0415

        self.fn, meta = eval_features._original(ref, tier)  # noqa: SLF001
        self.dtype, self.how = meta["dtype"], meta["how"]
        self.note = "transformers pooler_output" if ref.startswith("dinov3") else "dinov2 forward_features CLS"

    def prep(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return args

    def __call__(self, *args: Any) -> Any:  # noqa: ANN401
        import torch  # noqa: PLC0415

        with torch.no_grad():
            return self.fn(*args)


def _pin(adapter: Any) -> str:  # noqa: ANN401
    if adapter is None:
        return ""
    try:
        return getattr(adapter, "COMMIT", None) or adapter.mod().REF_COMMIT
    except Exception:  # noqa: BLE001 - the pin is informational
        return ""


def main() -> int:  # noqa: C901, PLR0912, PLR0915 - one resumable loop
    """Command line entry; 3 = budget spent, call again."""
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("refs")
    ap.add_argument("--tiers", default="upstream,upstream16,upstreamhalf")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="call", choices=("call", "network"), help="network: pose network alone")
    a = ap.parse_args()
    t0 = time.perf_counter()
    bu = _original.bench()
    import torch  # noqa: PLC0415

    measure = getattr(bu, "measure", None)
    if measure is None:
        from bench_gaps import measure  # noqa: PLC0415  # ty: ignore[unresolved-import]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if out.exists():
        rows = [r for r in map(json.loads, out.read_text().splitlines()) if r.get("part", "call") == a.part]
        have = {(r["ref"], r["tier"], r.get("input")) for r in rows}
    torch.zeros(1, device="cuda").sum().item()
    for ref in a.refs.split(","):
        full, card, _ = _original.full_ref(a.task, ref)
        adapter = bu.FAM.get(card.family) or _original.LOCAL.get(card.family)
        for tier in a.tiers.split(","):
            own = a.task != "features" and _original.own_option(a.task, ref)
            if (tier == "upstreamhalf" and own) or (tier == "upstreamac" and not own):
                continue  # .half() only without an own option; upstreamac = upstream16 without one
            if any(k[:2] == (ref, tier) and (k[2] is None) for k in have):
                continue  # a recorded failure
            if time.perf_counter() - t0 > a.max_seconds:
                return 3
            base: dict[str, Any] = {"ref": ref, "task": a.task, "mode": "upstream", "tier": tier}
            if a.part != "call":
                base["part"] = a.part
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
                tb = time.perf_counter()
                if a.task == "features":  # the accuracy eval's builder (tools/eval_features.py), batch 1
                    o = _Features(ref, tier)
                    ins = [("224x224", "synthetic (fastforks bench)", (bu.synth_u8(224, 224)[None],))]
                else:
                    o = _original.load(a.task, ref, tier, a.part)
                    ins = adapter.inputs(full) if hasattr(adapter, "inputs") else bu.inputs_for(a.task)
                torch.cuda.synchronize()
                load_s = time.perf_counter() - tb
            except Exception as e:  # noqa: BLE001 - recorded
                err = f"{type(e).__name__}: {str(e)[:300]}"
                with out.open("a") as f:
                    f.write(json.dumps(base | {"error": ("OOM, " if "out of memory" in err else "") + err}) + "\n")
                continue
            base |= {"dtype": o.dtype, "how": o.how, "note": o.note}
            for label, scene, args in ins:
                if (ref, tier, label) in have:
                    continue
                row = base | {"input": label, "scene": scene}
                try:
                    pa = o.prep(args)
                    torch.cuda.empty_cache()
                    first, pf, ts = measure(lambda pa=pa, o=o: o(*pa), a.warmup, a.runs)
                    row |= {
                        "ms": round(1e3 * statistics.median(ts), 2), "ms_min": round(1e3 * min(ts), 2),
                        "runs": len(ts), "load_s": round(load_s, 2), "first_call_s": round(first, 3),
                        "peak_first_alloc_mb": round(pf[0]), "peak_first_reserved_mb": round(pf[1]),
                        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
                    }  # fmt: skip
                except Exception as e:  # noqa: BLE001 - recorded
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    row["error"] = ("OOM, " if "out of memory" in err else "") + err
                    torch.cuda.empty_cache()
                row |= {
                    "protocol": f"median of {a.runs} after {a.warmup} warm-up, uint8 CUDA input, original code at the "
                    "pinned commit, torch.no_grad; peak = max over the first call and the timed calls",
                    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                    "date": time.strftime("%Y-%m-%d"),
                    "commit": _pin(adapter), "ff_commit": _commit(),
                    "source": "", "command": " ".join(["python", "tools/speed_original.py", *sys.argv[1:]]),
                }  # fmt: skip
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row.get(k) for k in ("ref", "tier", "input", "ms", "peak_reserved_mb", "error")}))
            del o
    return 0


if __name__ == "__main__":
    sys.exit(main())
