#!/bin/bash
# Gap-fill queue 2026-11-15 (unit gapfill-trt, verylow GPU tier, one capped unit). One ref = one step; a failing step is
# logged and the queue continues. Stages: (1) missing original rows [blocked: no upstream clones, see the report],
# (2) missing TRT rows (cold build + cached-engine start), (3) missing compile rows (cold + warm-cache start),
# (4) backfill of the start columns for existing TRT / compile rows into results/raw (existing rows stay untouched).
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports || exit 1
D=results/raw/2026-11-15_gaps; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export FASTFORKS_OFFLINE=1 FASTFORKS_DATA=${FASTFORKS_DATA:-/home/florian/projects/data}
LOG=$D/queue.log
log() { echo "$(date +%F_%T) $*" >> $LOG; }
disk_ok() { [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 6 ]; }
step() {  # step OUT MODE REF [extra flags]
  local out=$1 mode=$2 ref=$3; shift 3
  disk_ok || { log "STOP: SSD < 6 GB free"; exit 3; }
  log "start $mode $ref"
  timeout 900 $PY tools/speed_check.py run --out $out --only $ref --modes $mode --sides lib --no-profile \
    --prefix "$LOCK timeout 175" --timeout 190 "$@" >> $D/queue_steps.log 2>&1
  log "end $mode $ref rc=$?"
}
log "queue start commit $(git rev-parse --short HEAD) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
log "stage 1 original rows: skipped (memfof-kitti/-sintel/-spring need FF_MEMFOF_REPO, tapnext-pp the tapnet clone, VDA-metric-s has no adapter; no downloads in the queue)"
OUT=$D/speed_compile_new.jsonl
for r in $(cat $D/queue_trt_missing.txt); do step $OUT tensorrt $r; done
for r in $(cat $D/queue_compile_missing.txt); do step $OUT compile $r --warm-start; done
$PY $D/merge_rows.py >> $LOG 2>&1
bash $D/publish.sh >> $LOG 2>&1
for r in $(cat $D/queue_trt_backfill.txt); do step $D/backfill_trt.jsonl tensorrt $r --force; done
for r in $(cat $D/queue_compile_backfill.txt); do step $D/backfill_compile.jsonl compile $r --warm-start --force; done
log "queue done"
