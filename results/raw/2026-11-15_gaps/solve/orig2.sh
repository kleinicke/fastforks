#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps
export TMPDIR=/home/florian/scratch_tests/gtmp
S=results/raw/2026-11-15_gaps/solve
FF_MEMFOF_REPO=../upstream/memfof /home/florian/projects/shared/bin/gpulock timeout 290 ../fastforks/.venv/bin/python tools/speed_original.py flow memfof-kitti,memfof-sintel,memfof-spring --tiers upstream --out $S/rows_original.jsonl > $S/orig2.log 2>&1
echo rc=$? >> $S/orig2.log
