#!/bin/bash
# usage: kq.sh REF [more speed_check args]; kernels row of a ref
E=/home/florian/projects/experiments/017-fastforks-lib; W=$E/wt-kmem; cd $W || exit 1
O=/home/florian/scratch_tests/kmem; U=/mnt/hdd/projects/upstream
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_OFFLINE=1 HF_HUB_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
export PYTHONPATH=src:$U/_pkgs:tools TMPDIR=$O/tmp
. $E/review/kernel_path.sh; export FASTFORKS_KERNEL_PATH=$KP
r=$1; shift
exec /home/florian/projects/shared/bin/gpulock-low $E/fastforks/.venv/bin/python results/raw/2026-11-16_origgaps2/kernels_speed.py $r --out $O/krows_$r.jsonl --prefix "timeout 250" --sides lib "$@"
