#!/bin/bash
# TensorRT rows of dinov2-vitg14-reg / dinov3-vith16plus (master2 failed with "problem writing element": /tmp tmpfs full); TMPDIR on SSD.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_DATA=/home/florian/projects/data TMPDIR=/home/florian/scratch_tests/port-gaps-tmp
for ref in dinov2-vitg14-reg dinov3-vith16plus; do
  timeout 3300 $PY tools/speed_check.py run --out $SV/rows_compile3.jsonl --only $ref --modes tensorrt --sides lib --no-profile --prefix "$LOCK timeout 175" --timeout 7000 >> $SV/master3_steps.log 2>&1
done
