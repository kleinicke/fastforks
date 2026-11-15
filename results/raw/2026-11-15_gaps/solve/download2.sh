#!/bin/bash
# Download the uncached weights smallest first, outside any gpulock, only while / keeps >= 30 GB free.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
S=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; export PYTHONPATH=src
while IFS=$'\t' read -r ref mb; do
  free=$(df --output=avail -BM / | tail -1 | tr -dc 0-9)
  if [ $((free - mb)) -lt 30720 ]; then echo "$(date +%T) STOP before $ref: ${free} MB free, needs ${mb} MB, floor 30720 MB" >> $S/download2.log; echo "$ref	$mb	stopped: / would drop below 30 GB" >> $S/dl_skipped2.txt; continue; fi
  echo "$(date +%T) $ref ${mb} MB (free ${free} MB)" >> $S/download2.log
  if timeout 1800 $PY -m fastforks download $ref >> $S/download2.log 2>&1; then echo "$ref" >> $S/dl_done2.txt; else echo "$ref" >> $S/dl_failed2.txt; fi
done < $S/dl_order2.txt
echo "$(date +%T) download done" >> $S/download2.log
