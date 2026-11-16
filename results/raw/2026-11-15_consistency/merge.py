"""Merge the 2026-11-15 consistency re-measures into docs/data/speed.jsonl (replace by ref, mode, input, tier, how).

Run from the worktree root: ../fastforks/.venv/bin/python results/raw/2026-11-15_consistency/merge.py
"""

import json
from pathlib import Path

D = Path("results/raw/2026-11-15_consistency")
DST = Path("docs/data/speed.jsonl")
CMD = ("tools/speed_check.py run --only <ref> --modes <modes> --sides lib --no-profile --force "
       "--prefix 'gpulock-low timeout 175' --timeout 190 (results/raw/2026-11-15_consistency/queue*.sh)")
# the last run of each finding: C run 2 (rows_c2, rows_c2s), OWLv2 after the duplicate-weight fix (rows_b_after)
FILES = ["rows_a.jsonl", "rows_b2.jsonl", "rows_b_after.jsonl", "rows_c2.jsonl", "rows_c2s.jsonl", "rows_d.jsonl", "rows_e_before.jsonl"]


def rd(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def key(r):
    return (r["ref"], r["mode"], r.get("input"), r.get("tier"), r.get("how"))


new = {}
for f in FILES:
    for r in rd(D / f):
        if r.get("ms") is None or r.get("input") in (None, "-") or r.get("side", "lib") != "lib":
            continue
        r = {k: v for k, v in r.items() if k != "error"}  # the error of a failed input of the same worker
        r["source"] = str(D / f)
        r.setdefault("command", CMD)
        new[key(r)] = r
old = rd(DST)
keep = [r for r in old if key(r) not in new]
DST.write_text("".join(json.dumps(r) + "\n" for r in keep + list(new.values())))
print(f"{DST}: replaced {len(old) - len(keep)}, added {len(new) - (len(old) - len(keep))}")
