#!/bin/bash
source "$(dirname "$0")/env.sh"
FASTFORKS_DATA=/home/florian/projects/data/stereo /home/florian/projects/experiments/017-fastforks-lib/fastforks/.venv/bin/python tools/speed_check.py run --out docs/data/speed_compile.jsonl --only pointdit-b-512 --modes tensorrt --sides lib --force --prefix "$L" --timeout 600 > $R/q6.log 2>&1
