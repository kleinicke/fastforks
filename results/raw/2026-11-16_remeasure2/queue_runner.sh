#!/bin/bash
# usage: queue_runner.sh JOBFILE   (lines "ref mode"); re-measure queue of 2026-11-16 remeasure2
E=/home/florian/projects/experiments/017-fastforks-lib; W=$E/wt-remeasure2; cd $W || exit 1
O=/home/florian/scratch_tests/remeasure2; U=/mnt/hdd/projects/upstream
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_OFFLINE=1 HF_HUB_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
export PYTHONPATH=src:$U/_pkgs:tools TMPDIR=$O/tmp
. $E/review/kernel_path.sh; export FASTFORKS_KERNEL_PATH=$KP
PY=$E/fastforks/.venv/bin/python
while read r m; do
  [ -z "$r" ] && continue
  /home/florian/projects/shared/bin/gpulock-low $PY results/raw/2026-11-16_remeasure2/kernels_speed.py $r --modes $m --out $O/rows_${r}_$m.jsonl --prefix "timeout 500" --sides lib --force
done < $1
echo QUEUE_DONE
