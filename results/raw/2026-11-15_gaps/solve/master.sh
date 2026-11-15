#!/bin/bash
# Gap-solve master loop (unit gapfill-solve). One step = one (ref, mode); logs and continues on failure.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
D=results/raw/2026-11-15_gaps; SV=$D/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
LOG=$SV/master.log; log() { echo "$(date +%F_%T) $*" >> $LOG; }
step() {  # step OUT MODE REF [flags]
  local out=$1 mode=$2 ref=$3; shift 3
  [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 6 ] || { log "SKIP (disk < 6 GB) $mode $ref"; return; }
  log "start $mode $ref"
  timeout 3300 $PY tools/speed_check.py run --out $out --only $ref --modes $mode --sides lib --no-profile \
    --prefix "$LOCK timeout 175" --timeout 7000 "$@" >> $SV/master_steps.log 2>&1
  log "end $mode $ref rc=$?"
}
log "master start $(git rev-parse --short HEAD)"
C=$SV/rows_compile.jsonl; P=$SV/rows_portable.jsonl
# romav1: cold compile cache built unlocked (capped unit), then timed under the lock
if grep -qx romav1 $SV/dl_done.txt || ls ~/.cache/fastforks 2>/dev/null | grep -qi roma; then
  timeout 3000 $PY tools/speed_check.py run --out $SV/romav1_cold.jsonl --only romav1 --modes compile --sides lib --no-profile \
    --cache-dir $SV/cache_romav1 --timeout 3000 >> $SV/master_steps.log 2>&1; log "romav1 cold build rc=$?"
  step $C compile romav1 --cache-dir $SV/cache_romav1 --cold-from $SV/romav1_cold.jsonl --force
fi
for r in $(cat $SV/q_fixed.txt); do step $C tensorrt $r --force; done
for r in $(cat $SV/q_fixed.txt); do step $C compile $r --warm-start --force; done
for r in $(cat $SV/q_new_tensorrt.txt); do step $C tensorrt $r; done
for r in $(cat $SV/q_new_compile.txt); do step $C compile $r --warm-start; done
for r in $(cat $SV/q_new_portable.txt); do step $P portable $r; done
for r in $(cat $SV/q_new_kernels.txt); do step $P kernels $r; done
# original rows of the memfof / tapnext refs (adapters need the upstream clones)
UP=/home/florian/projects/experiments/017-fastforks-lib/upstream
for spec in flow:memfof-kitti,memfof-sintel,memfof-spring tracking:tapnext-pp; do
  FF_MEMFOF_REPO=$UP/memfof TAPNET_SRC=$UP/tapnet timeout 900 $LOCK timeout 170 $PY tools/speed_original.py ${spec%%:*} ${spec#*:} --tiers upstream --out $SV/rows_original.jsonl >> $SV/master_steps.log 2>&1
  log "original $spec rc=$?"
done
log "master done"
