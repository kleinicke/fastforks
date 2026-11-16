"""Round-2 merge: bootstrap / master2 / VDA metric-s / tapnext-pp original rows -> docs/data (new keys only, never an
existing OK row). Run from the worktree root."""
import json
from pathlib import Path
SV = Path("results/raw/2026-11-15_weights3")
rd = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []
SKIP = ("load: Weights", "load: Licence", "OSError")
def merge(dst, rows, src, cmd, key):
    if dst.endswith("speed.jsonl"):  # the floor file only takes measured rows (a failed row with input "-" breaks speed_check jobs())
        rows = [r for r in rows if r.get("ms") is not None and r.get("input") not in (None, "-")]
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
def conv(r):
    f = str(r.get("fails", ""))
    if f.startswith("worker rc=124"):
        r["fails"] = "unsupported: cold " + r["mode"] + " build exceeds the 175 s slice of gpulock (worker timeout); no row within the 5 min lock rule"
    elif f and not f.startswith("unsupported"):
        r["fails"] = "unsupported: " + re.sub(r"\x1b\[[0-9;]*m", "", f)[:300]
    return r
import re
R = [conv(r) for r in rd(SV / "rows_ct.jsonl")]
cmd = "tools/speed_check.py run --only REF --modes MODE [--warm-start] --sides lib --no-profile --prefix 'gpulock-verylow timeout 175' (queue.sh, results/raw/2026-11-15_weights3/)"
merge("docs/data/speed_compile.jsonl", R, "results/raw/2026-11-15_weights3/rows_ct.jsonl", cmd, k3)
