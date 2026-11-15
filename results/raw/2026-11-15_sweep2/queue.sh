#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-sweep2 || exit 1
D=results/raw/2026-11-15_sweep2
export FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data PYTHONPATH=/home/florian/projects/experiments/017-fastforks-lib/wt-ports-sweep2/src
PFX="/home/florian/projects/experiments/017-fastforks-lib/review/cpuslot --gpu systemd-run --user --wait --collect -P -d -p MemoryMax=15G -E HOME=$HOME -E FASTFORKS_OFFLINE=1 -E FASTFORKS_DATA=/home/florian/projects/data -E PYTHONPATH=/home/florian/projects/experiments/017-fastforks-lib/wt-ports-sweep2/src /home/florian/projects/shared/bin/gpulock-verylow timeout 175"
timeout 6500 /home/florian/projects/experiments/017-fastforks-lib/fastforks/.venv/bin/python tools/speed_check.py run --out $D/rows.jsonl --changed-since HEAD~2 --modes kernels --sides lib --no-profile --prefix "$PFX" --timeout 1400 > $D/queue.log 2>&1
echo done >> $D/queue.log
