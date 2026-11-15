"""Round-2 merge: bootstrap / master2 / VDA metric-s / tapnext-pp original rows -> docs/data (new keys only, never an
existing OK row). Run from the worktree root."""
import json
from pathlib import Path
SV = Path("results/raw/2026-11-15_gaps/solve")
rd = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []
SKIP = ("load: Weights", "load: Licence", "OSError")
def merge(dst, rows, src, cmd, key):
    old = rd(dst)
    ok = {key(r) for r in old if not r.get("fails") and "error" not in r}
    new = {}
    for r in rows:
        if key(r) in ok or str(r.get("fails", "")).startswith(SKIP): continue
        r = dict(r); r["source"] = src; r.setdefault("command", cmd); new[key(r)] = r
    keep = [r for r in old if key(r) not in new]
    with open(dst, "w") as f:
        for r in keep + list(new.values()): f.write(json.dumps(r) + "\n")
    print(dst, "appended/replaced", len(new))
k3 = lambda r: (r["ref"], r["mode"], r.get("side"))
k5 = lambda r: (r["ref"], r["mode"], r.get("side"), r.get("tier"), r.get("input"), r.get("how"))
vda = {"video-depth-anything-metric-s"}
cmdb = "results/raw/2026-11-15_gaps/solve/bootstrap_speed.py REF --modes MODE --sides lib --no-profile --prefix 'gpulock-verylow timeout 175' (q_boot.sh)"
RT0 = rd(SV / "rows_retry.jsonl")
P = [r for r in rd(SV / "rows_boot.jsonl") if r["ref"] not in vda] + [r for r in RT0 if r["mode"] in ("portable", "kernels")]
merge("docs/data/speed.jsonl", P, "results/raw/2026-11-15_gaps/solve/rows_boot.jsonl", cmdb, k3)
RT = [r for r in rd(SV / "rows_retry.jsonl")]
C = [r for r in RT if r["mode"] in ("compile", "tensorrt")] + [r for r in rd(SV / "rows_boot_ct.jsonl") if r["ref"] not in vda] + rd(SV / "rows_compile2.jsonl") + rd(SV / "rows_compile3.jsonl")
merge("docs/data/speed_compile.jsonl", C, "results/raw/2026-11-15_gaps/solve/rows_boot_ct.jsonl + rows_compile2.jsonl", cmdb, k3)
V = {k5(r): r for r in rd(SV / "rows_vda.jsonl")}
for r in rd(SV / "rows_vda_rerun.jsonl"): V[k5(r)] = r
merge("docs/data/speed.jsonl", list(V.values()), "results/raw/2026-11-15_gaps/solve/rows_vda.jsonl", "", k5)
O = rd(SV / "rows_original2.jsonl")
merge("docs/data/speed.jsonl", O, "results/raw/2026-11-15_gaps/solve/rows_original2.jsonl", "tools/speed_original.py tracking tapnext-pp --tiers upstream (master2.sh)", k5)
