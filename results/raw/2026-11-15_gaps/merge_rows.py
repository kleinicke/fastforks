"""Append the gap-fill rows (results/raw/2026-11-15_gaps/speed_compile_new.jsonl) to docs/data/speed_compile.jsonl.

Only (ref, mode, side) keys the target does not have yet; never touches an existing row. Rows get `source` / `command`.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
src = ROOT / "results/raw/2026-11-15_gaps/speed_compile_new.jsonl"
dst = ROOT / "docs/data/speed_compile.jsonl"
if not src.exists():
    sys.exit(0)
have = {(r["ref"], r["mode"], r.get("side")) for r in map(json.loads, filter(None, dst.read_text().splitlines()))}
new = {}
for r in map(json.loads, filter(None, src.read_text().splitlines())):
    k = (r["ref"], r["mode"], r.get("side"))
    if k in have or r.get("fails", "").startswith(("load: WeightsError", "load: LicenceError")):
        continue
    r["source"] = "results/raw/2026-11-15_gaps/speed_compile_new.jsonl"
    r["command"] = "tools/speed_check.py run --modes %s --sides lib --no-profile --warm-start (queue results/raw/2026-11-15_gaps/queue.sh)" % r["mode"]
    new[k] = r  # the last row of a key wins
with dst.open("a") as f:
    for r in new.values():
        f.write(json.dumps(r) + "\n")
print("appended", len(new))
