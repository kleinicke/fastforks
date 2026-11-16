#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-kmem
O=/home/florian/scratch_tests/kmem; U=/mnt/hdd/projects/upstream
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=$O/site153:src:$U/_pkgs:tools TMPDIR=$O/tmp
export FF_RESPLAT_REPO=/home/florian/projects/experiments/013-joint-view-depth/third_party/resplat FF_RESPLAT_SITE=$O/site
export CUDA_HOME=/home/florian/projects/experiments/001-gsplat-runtime/.venv/lib/python3.12/site-packages/nvidia/cu13
PY=/home/florian/projects/experiments/001-gsplat-runtime/.venv/bin/python
exec /home/florian/projects/shared/bin/gpulock-low timeout 280 $PY tools/speed_original.py multiview ${2:-resplat-base-re10k} --tiers "${1:-upstream,upstream16}" --runs 3 --max-seconds 200 --out $O/rows_${2:-resplat-base-re10k}.jsonl
