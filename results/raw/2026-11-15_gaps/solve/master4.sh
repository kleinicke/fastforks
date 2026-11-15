#!/bin/bash
# Retry the bootstrap (ref, mode) steps that failed only by the 190 s step timeout (gpulock-verylow wait counted): --timeout 7000, TMPDIR on SSD.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_DATA=/home/florian/projects/data TMPDIR=/home/florian/scratch_tests/port-gaps-tmp
python3 - <<'P' > $SV/q4.txt
import json
last={}
for f in ("rows_boot.jsonl","rows_boot_ct.jsonl"):
    for l in open("results/raw/2026-11-15_gaps/solve/"+f):
        r=json.loads(l)
        if r["ref"]=="video-depth-anything-metric-s": continue
        last[(r["ref"],r["mode"])]=r.get("fails") or r.get("error") or ""
for (ref,mode),e in last.items():
    if "timeout" in e: print(ref,mode)
P
while read ref mode; do
  extra=""; [ $mode = compile ] && extra="--warm-start"
  echo "$(date +%T) $ref $mode" >> $SV/master4.log
  timeout 3300 $PY $SV/bootstrap_speed.py $ref --modes $mode --sides lib --no-profile $extra --out $SV/rows_retry.jsonl --prefix "$LOCK timeout 175" --timeout 7000 >> $SV/master4_steps.log 2>&1
done < $SV/q4.txt
echo "$(date +%T) master4 done" >> $SV/master4.log
