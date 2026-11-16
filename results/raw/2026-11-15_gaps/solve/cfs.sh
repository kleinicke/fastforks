#!/bin/bash
# c-foundationstereo-s compile: cold build in <= 295 s gpulock slices on a persistent cache, then the warm timing.
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps || exit 1
SV=results/raw/2026-11-15_gaps/solve; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock
export TMPDIR=/home/florian/scratch_tests/port-gaps-tmp FASTFORKS_OFFLINE=1 FASTFORKS_DATA=/home/florian/projects/data
C=$SV/cache_cfs; mkdir -p $C; tot=0
for i in 1 2 3 4; do
  t0=$(date +%s)
  timeout 3300 $PY tools/speed_check.py run --out $SV/cfs_slice$i.jsonl --only c-foundationstereo-s --modes compile --sides lib --no-profile \
    --cache-dir $C --prefix "$LOCK timeout 295" --timeout 7000 --force >> $SV/cfs_steps.log 2>&1
  t1=$(date +%s); tot=$((tot + t1 - t0)); echo "$(date +%T) slice $i wall $((t1-t0)) s total $tot" >> $SV/cfs.log
  if python3 -c "import json,sys; r=[json.loads(l) for l in open('$SV/cfs_slice$i.jsonl')]; sys.exit(0 if r and not r[-1].get('fails') else 1)"; then
    echo "{\"ref\":\"c-foundationstereo-s\",\"mode\":\"compile\",\"compile_s\":$tot}" > $SV/cfs_cold.json; echo "$(date +%T) ok" >> $SV/cfs.log; break; fi
done
