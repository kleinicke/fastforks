#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps
export TMPDIR=/home/florian/scratch_tests/gtmp
S=results/raw/2026-11-15_gaps/solve
TAPNET_SRC=../upstream/tapnet /home/florian/projects/shared/bin/gpulock timeout 290 ../fastforks/.venv/bin/python tools/speed_original.py tracking tapnext-pp --tiers upstream --out $S/rows_original.jsonl > $S/orig3.log 2>&1
echo rc=$? >> $S/orig3.log
