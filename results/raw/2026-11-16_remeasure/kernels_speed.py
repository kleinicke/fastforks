"""speed_check run for refs that have a portable floor row but no kernels row: stub kernels floor rows (temp file only).

    python kernels_speed.py REF[,REF] [speed_check run args...]   (cwd = repo root, PYTHONPATH=src:tools)
"""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tools"))
import speed_check as sc
real = [json.loads(l) for l in sc.FLOOR.read_text().splitlines() if l.strip()]
refs = sys.argv[1].split(",")
import os
MODES=[sys.argv[sys.argv.index("--modes")+1]] if "--modes" in sys.argv else ["kernels"]
stubs = [dict(r, mode="kernels", ms=1.0, status=None) for r in real if r["ref"] in refs and r["mode"] == "portable" and not any(x["ref"]==r["ref"] and x["mode"]=="kernels" and x.get("input")==r.get("input") for x in real)]
tmp = Path(tempfile.mkdtemp()) / "speed.jsonl"
tmp.write_text("".join(json.dumps(r) + "\n" for r in real + stubs))
sc.FLOOR = tmp
sys.argv = ["speed_check.py", "run", "--only", *refs, *sys.argv[2:]]
sys.exit(sc.main())
