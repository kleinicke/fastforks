#!/bin/bash
# Portable/kernels/compile/TRT rows for refs without floor rows (bootstrap_speed.py: stub floor in memory). Unit gapfill-boot.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export FASTFORKS_NONCOMMERCIAL=1 PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data TMPDIR=/home/florian/scratch_tests/port-gaps-tmp
LOG=$SV/boot.log
for r in $(cat ${1:-$SV/q_boot_refs.txt}); do
  echo "$(date +%T) $r portable kernels" >> $LOG
  timeout 1200 $PY $SV/bootstrap_speed.py $r --modes portable kernels --sides lib --no-profile --out $SV/rows_boot.jsonl --prefix "$LOCK timeout 175" --timeout 190 >> $SV/boot_steps.log 2>&1
  for m in compile tensorrt; do
    echo "$(date +%T) $r $m" >> $LOG
    extra=""; [ $m = compile ] && extra="--warm-start"
    timeout 1500 $PY $SV/bootstrap_speed.py $r --modes $m --sides lib --no-profile $extra --out $SV/rows_boot_ct.jsonl --prefix "$LOCK timeout 175" --timeout 190 >> $SV/boot_steps.log 2>&1
  done
done
echo "$(date +%T) done" >> $LOG
