#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-portable2 || exit 1
D=results/raw/2026-11-15_portable2
export FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data PYTHONPATH=$PWD/src
for r in efficientvit_b2.r288_in1k fastvit_t8.apple_dist_in1k; do
../fastforks/.venv/bin/python tools/speed_check.py run --out $D/prof_rows.jsonl --only $r --modes portable --sides lib --force --prefix "/home/florian/projects/shared/bin/gpulock-verylow timeout 175" --timeout 600
done
echo done > $D/prof.done
