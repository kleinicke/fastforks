#!/bin/bash
# Gap-solve master loop (unit gapfill-m2). One step = one (ref, mode); logs and continues on failure.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
D=results/raw/2026-11-15_gaps; SV=$D/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
LOG=$SV/master2.log; log() { echo "$(date +%F_%T) $*" >> $LOG; }
step() {  # step OUT MODE REF [flags]
  local out=$1 mode=$2 ref=$3; shift 3
  [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 6 ] || { log "SKIP (disk < 6 GB) $mode $ref"; return; }
  log "start $mode $ref"
  timeout 3300 $PY tools/speed_check.py run --out $out --only $ref --modes $mode --sides lib --no-profile \
    --prefix "$LOCK timeout 175" --timeout 7000 "$@" >> $SV/master2_steps.log 2>&1
  log "end $mode $ref rc=$?"
}
log "master2 start $(git rev-parse --short HEAD)"
C=$SV/rows_compile2.jsonl
UP=/home/florian/projects/experiments/017-fastforks-lib/upstream
FASTFORKS_NONCOMMERCIAL=1 TAPNET_SRC=$UP/tapnet timeout 900 $LOCK timeout 170 $PY tools/speed_original.py tracking tapnext-pp --tiers upstream --out $SV/rows_original2.jsonl >> $SV/master2_steps.log 2>&1; log "original tapnext-pp rc=$?"
export FASTFORKS_NONCOMMERCIAL=1
for r in $(cat $SV/q2_trt.txt); do step $C tensorrt $r; done
for r in $(cat $SV/q2_comp.txt); do step $C compile $r --warm-start; done
log "master2 done"
