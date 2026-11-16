#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock
export TMPDIR=/home/florian/scratch_tests/port-gaps-tmp FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
for r in dfine-x-coco rtdetrv2-x-coco; do
timeout 3300 $PY tools/speed_check.py run --out $SV/rows_dfine.jsonl --only $r --modes tensorrt --sides lib --no-profile --prefix "$LOCK timeout 295" --timeout 7000 --force >> $SV/dfine_steps.log 2>&1
done
echo "$(date +%T) dfine fix done" >> $SV/special.log
