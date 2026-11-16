#!/bin/bash
source "$(dirname "$0")/env.sh"
/home/florian/projects/experiments/017-fastforks-lib/fastforks/.venv/bin/python tools/speed_check.py run --out docs/data/speed_compile.jsonl --only pointdit-l-512 --modes compile tensorrt --sides lib --force --prefix "$L" --timeout 600 > $R/q5.log 2>&1
