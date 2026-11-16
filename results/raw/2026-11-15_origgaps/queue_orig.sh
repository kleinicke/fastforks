#!/bin/bash
W=/home/florian/projects/experiments/017-fastforks-lib/wt-origgaps
cd $W || exit 1
O=/home/florian/scratch_tests/origgaps; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-low
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 PYTHONPATH=src FASTFORKS_OFFLINE=1 HF_HUB_OFFLINE=1
S=/home/florian/scratch_tests/sfm2
export FF_OMNI_DC_REPO=/home/florian/projects/experiments/013-joint-view-depth/third_party/OMNI-DC FF_FLEXGLUE_REPO=$S/src/flexglue FF_FLEXGLUE_LIGHTGLUE=$S/pkgs FF_LOMA_REPO=$S/src/loma
for spec in "matching loma-b"; do
  set -- $spec; echo "== $2 $(date +%T)" >> $O/q4.log
  timeout 175 $LOCK $PY tools/speed_original.py $1 $2 --tiers upstream --out $O/orig_rows.jsonl >> $O/q4.log 2>&1
  echo "rc $?" >> $O/q4.log
done
echo done4 >> $O/q4.log
