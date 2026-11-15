#!/bin/bash
W=/home/florian/projects/experiments/017-fastforks-lib/wt-origgaps
cd $W || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-low
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data TMPDIR=/home/florian/scratch_tests/origgaps/tmp
mkdir -p $TMPDIR
O=/home/florian/scratch_tests/origgaps
timeout 600 $PY $SV/bootstrap_speed.py resplat-base-re10k --modes portable --sides lib --no-profile --out $O/rows_boot.jsonl --prefix "/home/florian/projects/shared/bin/gpulock timeout 290" --timeout 300 > $O/q1.log 2>&1
echo done >> $O/q1.log
