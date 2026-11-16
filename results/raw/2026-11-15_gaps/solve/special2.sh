#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock
export TMPDIR=/home/florian/scratch_tests/port-gaps-tmp FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
timeout 3300 $PY tools/speed_check.py run --out $SV/rows_special.jsonl --only c-foundationstereo-s --modes compile --sides lib --no-profile --warm-start --prefix "$LOCK timeout 295" --timeout 7000 --force >> $SV/special_steps.log 2>&1
echo "$(date +%T) cfs done" >> $SV/special.log
$LOCK timeout 290 $PY $SV/dfine_dbg.py 24 > $SV/dfine_dbg1.log 2>&1
echo "$(date +%T) dfine dbg1 done" >> $SV/special.log
