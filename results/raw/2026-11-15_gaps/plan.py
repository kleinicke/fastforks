"""Plan of the gap-fill queue: lists of refs per stage, restricted to refs whose weights are cached (no downloads)."""
import csv, json, sys
from pathlib import Path
D = Path(__file__).resolve().parent
ROOT = D.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fastforks._core import registry
R = list(csv.DictReader(open(D / "gaps.csv")))
cached, nocache = {}, []
cards = {m: cv for t in registry.TASKS for m, cv in registry.cards(t).items()}
def have(ref):
    if ref not in cached:
        card, v = cards[ref]
        try:
            registry.weights(card, v, allow_download=False); cached[ref] = True
        except Exception as e:
            cached[ref] = False; nocache.append((ref, type(e).__name__))
    return cached[ref]
def mk(name, refs):
    ok = [r for r in refs if have(r)]
    (D / f"queue_{name}.txt").write_text("".join(f"{r}\n" for r in ok))
    print(name, len(refs), "queued", len(ok))
portable_ok = [r["ref"] for r in R if r["portable"] == "ok"]
mk("trt_missing", [r["ref"] for r in R if r["tensorrt"] == "missing" and r["portable"] == "ok"])
mk("compile_missing", [r["ref"] for r in R if r["compile"] == "missing" and r["portable"] == "ok"])
mk("trt_backfill", [r["ref"] for r in R if r["tensorrt"] == "ok" and r["tensorrt_start_cached"] == "False"])
mk("compile_backfill", [r["ref"] for r in R if r["compile"] == "ok" and r["compile_start_warm"] == "False"])
(D / "weights_not_cached.txt").write_text("".join(f"{a}\t{b}\n" for a, b in nocache))
