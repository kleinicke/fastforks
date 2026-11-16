#!/bin/bash
# Consistency re-measure 2026-11-15 (audit findings A-D): speed_check portable/kernels rows at the floor inputs on the
# current commit, gpulock-low per worker (< 3 min). Run as one capped unit (cpuslot --gpu systemd-run ... MemoryMax=20G).
cd /home/florian/projects/experiments/017-fastforks-lib/wt-consistency || exit 1
D=results/raw/2026-11-15_consistency; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-low
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights/perception_encoder:/home/florian/projects/data/ff-weights/seedvr2 FASTFORKS_NONCOMMERCIAL=1 PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
LOG=$D/queue.log
step() {  # step OUT "MODES" REF...
  local out=$1 modes=$2; shift 2
  echo "$(date +%T) start $modes $*" >> $LOG
  timeout 1500 $PY tools/speed_check.py run --out $out --only "$@" --modes $modes --sides lib --no-profile --force \
    --prefix "$LOCK timeout 175" --timeout 190 >> $D/steps.log 2>&1
  echo "$(date +%T) end rc=$?" >> $LOG
}
echo "$(date +%T) queue start $(git rev-parse --short HEAD) ${1:-all}" >> $LOG
case ${1:-all} in all|1)
for r in pe-spatial-g14-448 pe-core-g14-448; do step $D/rows_b2.jsonl "portable kernels" $r; done
step $D/rows_c1s.jsonl "portable kernels" seedvr2-3b
step $D/rows_c2s.jsonl "kernels portable" seedvr2-3b
;; esac
echo "$(date +%T) queue done" >> $LOG
