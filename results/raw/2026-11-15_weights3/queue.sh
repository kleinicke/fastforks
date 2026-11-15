#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-weights3 || exit 1
S=results/raw/2026-11-15_weights3; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_DATA=/home/florian/projects/data TMPDIR=/home/florian/scratch_tests/port-w3-tmp
for ref in $(cat $S/q_ct.txt); do
  echo "$(date +%T) $ref tensorrt" >> $S/queue.log
  timeout 1500 $PY tools/speed_check.py run --out $S/rows_ct.jsonl --only $ref --modes tensorrt --sides lib --no-profile --prefix "$LOCK timeout 175" --timeout 1400 >> $S/steps.log 2>&1
  echo "$(date +%T) $ref compile" >> $S/queue.log
  timeout 1500 $PY tools/speed_check.py run --out $S/rows_ct.jsonl --only $ref --modes compile --warm-start --sides lib --no-profile --prefix "$LOCK timeout 175" --timeout 1400 >> $S/steps.log 2>&1
done
echo "$(date +%T) done" >> $S/queue.log
