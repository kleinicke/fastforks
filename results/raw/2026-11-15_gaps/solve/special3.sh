#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock
export TMPDIR=/home/florian/scratch_tests/port-gaps-tmp FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
$LOCK timeout 290 $PY $SV/dfine_dbg.py 44 616 662 > $SV/dfine_dbg2.log 2>&1
echo "$(date +%T) dfine dbg2 done" >> $SV/special.log
