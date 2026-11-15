#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-portable2 || exit 1
D=results/raw/2026-11-15_portable2
export FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data PYTHONPATH=$PWD/src
timeout 3000 ../fastforks/.venv/bin/python tools/speed_check.py run --out $D/chg_rows.jsonl --changed-since HEAD --modes kernels --sides lib --no-profile --prefix "/home/florian/projects/shared/bin/gpulock-verylow timeout 175" --timeout 1400 > $D/chg.log 2>&1
echo done >> $D/chg.log
