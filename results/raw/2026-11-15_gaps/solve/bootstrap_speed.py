"""speed_check for refs without floor rows: stub floor rows (sibling variant's inputs, else the task's STANDARD input).

    python bootstrap_speed.py REF [speed_check run args...]   (cwd = repo root, PYTHONPATH=src, tools/ on path)
Only the in-memory FLOOR list is patched (a temp file); docs/data/speed.jsonl is not touched.
"""
import json, sys, tempfile
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))
import speed_check as sc
from fastforks._core import registry
real = [json.loads(l) for l in sc.FLOOR.read_text().splitlines() if l.strip()]
have = {r["ref"] for r in real if r["mode"] == "portable"}
refs = sys.argv[1].split(",")
cards = {m: cv for t in registry.TASKS for m, cv in registry.cards(t).items()}
task_of = {m: t for t in registry.TASKS for m in registry.cards(t)}
stubs = []
for ref in refs:
    if ref in have:
        continue
    card, v = cards[ref]
    sib = [r for r in real if cards.get(r["ref"]) and cards[r["ref"]][0].family == card.family and r["mode"] == "portable"]
    ins = sorted({(r["input"], r.get("scene", "synthetic")) for r in sib})
    if not ins:
        ins = [(sc.STANDARD.get(task_of[ref], "480x640"), "synthetic")]
    for lab, sc_ in ins:
        stubs.append({"ref": ref, "task": task_of[ref], "mode": "portable", "input": lab, "scene": sc_, "ms": 1.0})
tmp = Path(tempfile.mkdtemp()) / "speed.jsonl"
tmp.write_text("".join(json.dumps(r) + "\n" for r in real + stubs))
sc.FLOOR = tmp
sys.argv = ["speed_check.py", "run", "--only", *refs, *sys.argv[2:]]
sys.exit(sc.main())
