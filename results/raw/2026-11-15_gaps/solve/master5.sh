#!/bin/bash
# portable rows of the 6 refs whose 190 s bootstrap step timed out (lock wait counted); --timeout 7000, TMPDIR on SSD.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_DATA=/home/florian/projects/data TMPDIR=/home/florian/scratch_tests/port-gaps-tmp
for ref in $(cat $SV/q5.txt); do
  echo "$(date +%T) $ref portable" >> $SV/master5.log
  timeout 3300 $PY $SV/bootstrap_speed.py $ref --modes portable --sides lib --no-profile --out $SV/rows_retry2.jsonl --prefix "$LOCK timeout 175" --timeout 7000 >> $SV/master5_steps.log 2>&1
done
echo "$(date +%T) master5 done" >> $SV/master5.log
