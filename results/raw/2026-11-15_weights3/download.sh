#!/bin/bash
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-weights3 || exit 1
S=results/raw/2026-11-15_weights3; PY=../fastforks/.venv/bin/python; export PYTHONPATH=src FASTFORKS_NONCOMMERCIAL=1
while IFS=$'\t' read -r ref mb; do
  free=$(df --output=avail -BM / | tail -1 | tr -dc 0-9)
  if [ $((free - mb)) -lt 30720 ]; then echo "$ref	$mb	stopped: / would drop below 30 GB (free ${free} MB)" >> $S/dl_skipped.txt; continue; fi
  echo "$(date +%T) $ref ${mb} MB (free ${free} MB)" >> $S/download.log
  if timeout 1800 $PY -m fastforks download $ref >> $S/download.log 2>&1; then echo "$ref" >> $S/dl_done.txt; else echo "$ref" >> $S/dl_failed.txt; fi
done < $S/dl_order.txt
echo "$(date +%T) done" >> $S/download.log
