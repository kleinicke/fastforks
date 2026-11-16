#!/bin/bash
# diagnosis: failing refs, full errors into solve/diag1.log; waits for the other queue. Outer timeout is large: it includes the lock wait.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
S=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-verylow
export FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
while systemctl --user is-active --quiet gapfill-trt; do sleep 20; done
for spec in "$@"; do mode=${spec%%:*}; r=${spec#*:}
  echo "=== $mode $r $(date +%T)" >> $S/diag1.log
  timeout 1500 $PY tools/speed_check.py run --out $S/diag1.jsonl --only $r --modes $mode --sides lib --no-profile --force \
    --prefix "$LOCK timeout 175" --timeout 1400 >> $S/diag1.log 2>&1
done
echo "=== done $(date +%T)" >> $S/diag1.log
