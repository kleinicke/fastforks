#!/bin/bash
. /home/florian/projects/experiments/017-fastforks-lib/wt-accrows/results/raw/2026-11-16_accuracy_rows/env.sh
echo KP=$KP
for i in $(seq 40); do
  $L $PY tools/eval_features.py --models pe-core-g14-448 --tiers upstream,upstream16,portable,kernels --chunk 1000 --max-seconds 200 --cache /home/florian/.cache/ff_r5_tmp/acc/feat_cache.npy --out $R/features; rc=$?
  echo "rc=$rc"; [ $rc = 3 ] || break
done
echo DONE
