#!/bin/bash
. /home/florian/projects/experiments/017-fastforks-lib/wt-accrows/results/raw/2026-11-16_accuracy_rows/env.sh
for r in wd-vit-tagger-v3 wd-convnext-tagger-v3 wd-vit-large-tagger-v3 wd-eva02-large-tagger-v3; do
  $L $PY $R/agree_wd.py $r 500 $R/wd_agree.jsonl || echo "FAIL $r"
done
echo DONE
