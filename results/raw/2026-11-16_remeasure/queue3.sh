#!/bin/bash
# queue of the 2026-11-16 re-measure (stale rows after the depth-pro / hrnet / cotracker / graph fixes)
E=/home/florian/projects/experiments/017-fastforks-lib; W=$E/wt-remeasure; cd $W || exit 1
O=/home/florian/scratch_tests/remeasure; U=/mnt/hdd/projects/upstream
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_OFFLINE=1 HF_HUB_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
export PYTHONPATH=src:$U/_pkgs:tools TMPDIR=$O/tmp
. $E/review/kernel_path.sh; export FASTFORKS_KERNEL_PATH=$KP
PY=$E/fastforks/.venv/bin/python
run() { # ref mode
  /home/florian/projects/shared/bin/gpulock-low $PY results/raw/2026-11-16_remeasure/kernels_speed.py $1 --modes $2 --out $O/rows_$1_$2.jsonl --prefix "timeout 400" --sides lib --force
}
for r in great-igev-eth3d great-igev-kitti great-igev-middlebury great-igev-sceneflow las2-l las2-m raft-stereo-eth3d raft-stereo-middlebury raft-stereo-sceneflow film segformer-b0-ade20k segformer-b5-ade20k segnext-mscan-t-ade20k segnext-mscan-s-ade20k segnext-mscan-b-ade20k segnext-mscan-l-ade20k; do run $r kernels; done
echo QUEUE_DONE
