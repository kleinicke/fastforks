"""Gap matrix: every model ref of the library vs docs/data/speed.jsonl + speed_compile.jsonl (no torch)."""
import csv, json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from fastforks._core import registry
assert "torch" not in sys.modules, "torch imported"
refs = {}
for task in registry.TASKS:
    for mid, (card, v) in registry.cards(task).items():
        refs[mid] = (task, card.family)
def rd(p): return [json.loads(l) for l in open(ROOT / p) if l.strip()]
sp, cp = rd("docs/data/speed.jsonl"), rd("docs/data/speed_compile.jsonl")
def ok(r): return "error" not in r and not r.get("fails") and r.get("ms") is not None and r.get("ms")==r.get("ms")
def has(rows, ref, mode, side=None):
    return [r for r in rows if r["ref"] == ref and r["mode"] == mode and (side is None or r.get("side") == side)]
def cls(rs):
    if not rs: return "missing"
    if any(ok(r) for r in rs): return "ok"
    e = " | ".join(str(r.get("error") or r.get("fails") or "no ms")[:90] for r in rs[:1])
    return "OOM" if "OOM" in e or "out of memory" in e.lower() else "error: " + e
out = []
for ref, (task, fam) in sorted(refs.items()):
    row = dict(ref=ref, task=task, family=fam)
    for m in ("upstream", "upstream-default", "portable", "kernels"):
        row[m] = cls(has(sp, ref, m))
    orig = [r for r in has(sp, ref, "upstream") + has(sp, ref, "upstream-default") if ok(r)]
    row["orig_ms"] = bool(orig)
    row["orig_mem"] = any(r.get("peak_alloc_mb") is not None for r in orig)
    for m in ("compile", "tensorrt"):
        row[m] = cls(has(cp, ref, m, "lib"))
        good = [r for r in has(cp, ref, m, "lib") if ok(r)]
        row[m + "_startcold"] = any((r.get("compile_s") if m == "compile" else r.get("engine_build_s")) is not None for r in good)
    row["tensorrt_start_cached"] = any(r.get("engine_load_s") is not None for r in has(cp, ref, "tensorrt", "lib"))
    row["compile_start_warm"] = any(r.get("compile_warm_s") is not None for r in has(cp, ref, "compile", "lib"))
    out.append(row)
fn = list(out[0])
with open(ROOT / "results/raw/2026-11-15_gaps/gaps.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fn); w.writeheader(); w.writerows(out)
def stat(col): return collections.Counter(r[col].split(":")[0] for r in out)
print(len(out), "refs", len({r["family"] for r in out}), "families")
for c in ("upstream", "upstream-default", "portable", "kernels", "compile", "tensorrt"): print(c, dict(stat(c)))
print("no orig ms", sum(not r["orig_ms"] for r in out), "no orig mem", sum(r["orig_ms"] and not r["orig_mem"] for r in out))
print("orig OOM only", sum(not r["orig_ms"] and "OOM" in (r["upstream"]+r["upstream-default"]) for r in out))
for c in ("compile_startcold","tensorrt_start_cold" if 0 else "tensorrt_startcold","tensorrt_start_cached","compile_start_warm"): print(c, sum(r[c] for r in out))
