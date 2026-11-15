#!/bin/bash
# Re-run (ref, mode) steps of master2 that failed with "Disk quota exceeded" (/tmp tmpfs full of stale ONNX dirs), TMPDIR on the SSD.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_DATA=/home/florian/projects/data TMPDIR=/home/florian/scratch_tests/port-gaps-tmp
LOG=$SV/master3.log
python3 - <<'P' > $SV/q3.txt
import re
cur=None
for l in open("results/raw/2026-11-15_gaps/solve/master2_steps.log"):
    m=re.match(r"== lib (\S+) (\S+) ",l)
    if m: cur=m.groups()
    elif "Disk quota exceeded" in l and cur: print(*cur); cur=None
P
while read ref mode; do
  extra=""; [ $mode = compile ] && extra="--warm-start"
  echo "$(date +%T) $ref $mode" >> $LOG
  timeout 3300 $PY tools/speed_check.py run --out $SV/rows_compile3.jsonl --only $ref --modes $mode --sides lib --no-profile $extra --prefix "$LOCK timeout 175" --timeout 7000 >> $SV/master3_steps.log 2>&1
done < $SV/q3.txt
echo "$(date +%T) master3 done" >> $LOG
