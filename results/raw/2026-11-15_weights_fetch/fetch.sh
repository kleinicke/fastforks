#!/bin/bash
# sequential, resumable weights fetch via the library downloader; stops when / free < 10 GB.
PY=/home/florian/projects/experiments/017-fastforks-lib/wt-ports/.venv/bin/python
LOG=/home/florian/projects/experiments/017-fastforks-lib/wt-diskclean/results/raw/2026-11-15_weights_fetch/fetch.log
COLD=/mnt/hdd/projects/fastforks_cache_cold
export FASTFORKS_NONCOMMERCIAL=1
# ref:size_MB:target(ssd|cold)
for e in totalsegmentator-total@1:1200:ssd nninteractive@1:411:ssd nnfoundation-cnn@1:410:ssd seedvr2-3b@1:7300:ssd yolox-x@1:793:ssd resplat-base-re10k@1:889:ssd pointdit-b-512@1:1098:ssd span-x2@1:1327:ssd span-x4@1:10:ssd monster-plusplus-mix-all@1:1555:ssd monster-sceneflow@1:1555:ssd depth-pro@1:1904:ssd nnfoundation-vit@1:2698:ssd pointdit-l-512@1:3747:ssd da3-giant-1.1@1:5423:ssd pe-spatial-g14-448@1:7408:ssd pe-core-g14-448@1:7528:cold seedvr2-7b@1:16483:cold; do
  ref=${e%%:*}; r=${e#*:}; mb=${r%%:*}; tgt=${r##*:}
  if [ $tgt = ssd ]; then
    avail=$(df --output=avail -m / | tail -1)
    if [ $((avail-mb)) -lt 10240 ]; then echo "$(date +%T) STOP-SPACE before $ref (avail ${avail}MB need ${mb}MB)" >> $LOG; continue; fi
    $PY -m fastforks download $ref >> $LOG 2>&1; echo "$(date +%T) $ref rc=$?" >> $LOG
  else
    avail=$(df --output=avail -m /mnt/hdd | tail -1)
    if [ $((avail-mb)) -lt 3000 ]; then echo "$(date +%T) STOP-HDD before $ref" >> $LOG; continue; fi
    FASTFORKS_CACHE=$COLD $PY -m fastforks download $ref >> $LOG 2>&1; rc=$?
    for f in $(find $COLD/blobs -type f 2>/dev/null); do
      sha=$(basename $f); d=$HOME/.cache/fastforks/blobs/sha256/${sha:0:2}; mkdir -p $d; ln -sfn $f $d/$sha
    done
    echo "$(date +%T) $ref (cold) rc=$rc" >> $LOG
  fi
done
echo "$(date +%T) DONE" >> $LOG
