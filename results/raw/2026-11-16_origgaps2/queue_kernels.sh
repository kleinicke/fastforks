#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-origgaps
O=/home/florian/scratch_tests/origgaps3; U=/mnt/hdd/projects/upstream
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=src:$U/_pkgs:tools TMPDIR=$O/tmp FASTFORKS_KERNEL_PATH=$PWD/build/kernels
for r in memfof-sintel; do
 /home/florian/projects/shared/bin/gpulock-low ../fastforks/.venv/bin/python results/raw/2026-11-16_origgaps2/kernels_speed.py $r --out $O/rows_$r.jsonl --prefix "timeout 170" --sides lib >> $O/q.log 2>&1
done
echo done >> $O/q.log
