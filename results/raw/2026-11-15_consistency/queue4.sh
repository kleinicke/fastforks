#!/bin/bash
# OWLv2 goldens (GPU portable + kernels, CPU portable) and the family tests after the duplicate-weight fix (gpulock-low).
cd /home/florian/projects/experiments/017-fastforks-lib/wt-consistency || exit 1
D=results/raw/2026-11-15_consistency; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-low
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 PYTHONPATH=src FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
E=/home/florian/projects/experiments/017-fastforks-lib; W=$E/wt-consistency; . $E/review/kernel_path.sh; export FASTFORKS_KERNEL_PATH=$KP
export FASTFORKS_GOLDEN_PATH=/home/florian/projects/experiments/017-fastforks-lib/golden_ports
echo "$(date +%T) queue4 start $(git rev-parse --short HEAD)" >> $D/queue.log
$LOCK timeout 290 $PY -m pytest -q -rs --slow tests/test_golden.py -k "(test_gpu or test_cpu_portable or test_kernels_validated) and owlv2" \
  > $D/owlv2_tests.log 2>&1; echo "$(date +%T) goldens rc=$?" >> $D/queue.log
$LOCK timeout 290 $PY -m pytest -q -rs --slow tests/test_detection_owlv2.py >> $D/owlv2_tests.log 2>&1
echo "$(date +%T) queue4 done rc=$?" >> $D/queue.log
