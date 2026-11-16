#!/bin/bash
. /home/florian/projects/experiments/017-fastforks-lib/wt-accrows/results/raw/2026-11-16_accuracy_rows/env.sh
$L $PY tools/eval_prompt_depth.py --ref prompt-depth-anything-l --out $R/promptda.jsonl || echo "FAIL promptda"
echo DONE
