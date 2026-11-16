#!/bin/bash
# unit pp2: re-measure portable/kernels rows (and classification originals), one locked step each (<3 min)
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-portable2 || exit 1
D=results/raw/2026-11-15_portable2; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data PYTHONPATH=$PWD/src
for r in efficientvit_b2.r288_in1k fastvit_t8.apple_dist_in1k squeezenet1_1.tv_in1k ben2 flowformer-small-things moge-3-vitl gmflow-scale1-things swin2sr-x4 moge-2-vits-normal seedvr2-3b; do
  echo "$(date +%T) start $r" >> $D/master.log
  timeout 1500 $PY tools/speed_check.py run --out $D/lib_rows.jsonl --only $r --modes portable kernels --sides lib --force --no-profile \
    --prefix "$LOCK timeout 175" --timeout 1400 >> $D/steps.log 2>&1
  echo "$(date +%T) end $r rc=$?" >> $D/master.log
done
for r in efficientvit_b2.r288_in1k fastvit_t8.apple_dist_in1k squeezenet1_1.tv_in1k; do
  timeout 400 $LOCK timeout 170 $PY tools/speed_original.py classification $r --tiers upstream --out $D/orig_rows.jsonl >> $D/steps.log 2>&1
  echo "$(date +%T) orig $r rc=$?" >> $D/master.log
done
echo done >> $D/master.log
