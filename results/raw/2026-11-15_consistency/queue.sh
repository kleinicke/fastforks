#!/bin/bash
# Consistency re-measure 2026-11-15 (audit findings A-D): speed_check portable/kernels rows at the floor inputs on the
# current commit, gpulock-low per worker (< 3 min). Run as one capped unit (cpuslot --gpu systemd-run ... MemoryMax=20G).
cd /home/florian/projects/experiments/017-fastforks-lib/wt-consistency || exit 1
D=results/raw/2026-11-15_consistency; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-low
export FASTFORKS_NONCOMMERCIAL=1 PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
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
# C: kernels vs portable noise, run 1
for r in wd-vit-tagger-v3 rtmw-x-384 geocalib-pinhole seedvr2-3b; do step $D/rows_c1.jsonl "portable kernels" $r; done
# B: kernels vs portable memory after the duplicate-weight fixes
for r in owlv2-l14 pe-spatial-g14-448 pe-core-g14-448; do step $D/rows_b.jsonl "portable kernels" $r; done
# A: flowformer portable (no kernels mode)
for r in flowformer-small-sintel flowformer-small-things flowformer-small-chairs; do step $D/rows_a.jsonl portable $r; done
# C run 2 (reverse order)
for r in seedvr2-3b geocalib-pinhole rtmw-x-384 wd-vit-tagger-v3; do step $D/rows_c2.jsonl "portable kernels" $r; done
# D: portable vs original
for r in efficientvit_b2.r288_in1k fastvit_t8.apple_dist_in1k squeezenet1_1.tv_in1k shufflenet_v2_x1_0.tv_in1k pspnet-r101-d8-ade20k rife-4.25 rife-4.26 s2m2-m s2m2-l s2m2-xl; do step $D/rows_d.jsonl portable $r; done
;; esac
echo "$(date +%T) queue done" >> $LOG
