#!/bin/bash
W=/home/florian/projects/experiments/017-fastforks-lib/wt-origgaps
cd $W || exit 1
O=/home/florian/scratch_tests/origgaps2; PY=../fastforks/.venv/bin/python; LOCK=/home/florian/projects/shared/bin/gpulock-low
U=/mnt/hdd/projects/upstream
export FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_NONCOMMERCIAL=1 FASTFORKS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=src:$U/_pkgs TMPDIR=$O/tmp; mkdir -p $TMPDIR
export FF_ANY2FULL_REPO=$U/Any2Full FF_LINGBOT_DEPTH_REPO=$U/lingbot-depth FF_LINGBOT_MAP_REPO=$U/lingbot-map FF_DEPTHFM_REPO=$U/depth-fm
while read -r task ref tiers; do
  for i in 1 2 3 4 5 6; do
    echo "== $ref try $i $(date +%T)" >> $O/q.log
    $LOCK timeout 280 $PY tools/speed_original.py $task $ref --tiers $tiers --runs 3 --max-seconds 150 --out $O/rows.jsonl >> $O/q.log 2>&1
    rc=$?; echo "rc $rc" >> $O/q.log
    [ $rc -ne 3 ] && break
  done
done <<'L'
depth any2full-s upstream
depth any2full-b upstream
depth any2full-l upstream
depth lingbot-depth upstream,upstream16
depth lingbot-depth-dc upstream,upstream16
multiview lingbot-map upstream,upstream16
depth depthfm-v1 upstream,upstream16
L
echo done >> $O/q.log
