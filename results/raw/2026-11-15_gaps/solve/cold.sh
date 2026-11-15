#!/bin/bash
# cold TRT builds at plain gpulock (normal tier, <= 5 min), one at a time (unit gapfill-cold)
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock
export TMPDIR=/home/florian/scratch_tests/port-gaps-tmp FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
for r in stablenormal-turbo defom-stereo-vitl-sceneflow dinov2-vitg14 great-igev-middlebury waft-stereo-b igev-stereo-kitti12 igev-stereo-kitti15 mocha-v1-sceneflow selective-igev-eth3d-train selective-igev-kitti12 selective-igev-kitti15 selective-igev-middlebury-finetune selective-igev-middlebury-train selective-igev-sceneflow tapnext mvsanywhere sam-vit-h; do
  [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 6 ] || { echo "$(date +%T) SKIP disk $r" >> $SV/cold.log; continue; }
  echo "$(date +%T) start $r" >> $SV/cold.log
  timeout 3300 $PY tools/speed_check.py run --out $SV/rows_cold.jsonl --only $r --modes tensorrt --sides lib --no-profile \
    --prefix "$LOCK timeout 295" --timeout 7000 --force >> $SV/cold_steps.log 2>&1
  echo "$(date +%T) end $r rc=$?" >> $SV/cold.log
done
echo "$(date +%T) cold done" >> $SV/cold.log
