#!/bin/bash
# Consistency fixes 2026-11-15, after the OWLv2 duplicate-weight fix: A/B speed rows, OWLv2 goldens, and the general
# kernels-vs-portable load-memory test over the smallest variant of every family, in chunks (gpulock-low, < 5 min each).
# Run as one capped unit (cpuslot --gpu systemd-run ... MemoryMax=20G).
cd /home/florian/projects/experiments/017-fastforks-lib/wt-consistency || exit 1
D=results/raw/2026-11-15_consistency; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-low
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
E=/home/florian/projects/experiments/017-fastforks-lib; W=$E/wt-consistency; . $E/review/kernel_path.sh; export FASTFORKS_KERNEL_PATH=$KP
LOG=$D/queue.log
echo "$(date +%T) queue3 start $(git rev-parse --short HEAD) ${1:-all}" >> $LOG
case ${1:-all} in all|1)
for r in owlv2-l14 owlv2-b16; do
  echo "$(date +%T) start ab $r" >> $LOG
  timeout 1500 $PY tools/speed_check.py run --out $D/rows_b_after.jsonl --only $r --modes portable kernels --sides lib \
    --no-profile --force --prefix "$LOCK timeout 175" --timeout 190 >> $D/steps.log 2>&1
  echo "$(date +%T) end rc=$?" >> $LOG
done
echo "$(date +%T) start owlv2 goldens" >> $LOG
$LOCK timeout 290 $PY -m pytest -q -x tests/test_golden.py -k "test_gpu and owlv2" tests/test_detection_owlv2.py \
  > $D/owlv2_tests.log 2>&1; echo "$(date +%T) end rc=$?" >> $LOG
;; esac
case ${1:-all} in all|2)
$PY -m pytest -q --collect-only --slow tests/test_no_duplicate_weights.py -k kernels_load_holds 2>/dev/null \
  | grep "::" > $D/memtest_ids.txt
split -l 12 -d $D/memtest_ids.txt $D/memtest_chunk_
for c in $D/memtest_chunk_*; do
  echo "$(date +%T) start memtest $c" >> $LOG
  $LOCK timeout 290 $PY -m pytest -q --slow $(cat $c) >> $D/memtest.log 2>&1
  echo "$(date +%T) end rc=$?" >> $LOG
done
;; esac
echo "$(date +%T) queue3 done" >> $LOG
