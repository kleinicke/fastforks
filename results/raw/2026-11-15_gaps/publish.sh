#!/bin/bash
# After the missing-row stages: test gate in a capped unit, then commit the data separately and push (never force, never main).
cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports || exit 1
git diff --quiet -- docs/data/speed_compile.jsonl && { echo "publish: no new rows"; exit 0; }
systemd-run --user --wait --collect -p MemoryMax=8G --unit=gapfill-trt-tests --working-directory=$PWD \
  -E PYTHONPATH=src ../fastforks/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_docs.py tests/test_policy.py tests/test_speed_floor.py -m "not gpu" \
  > results/raw/2026-11-15_gaps/tests.log 2>&1 || { echo "publish: tests FAILED, data left uncommitted (see tests.log)"; exit 1; }
git add docs/data/speed_compile.jsonl results/raw/2026-11-15_gaps results/2026-11-15_measurement_gaps.md
git commit -q -m "Gap-fill rows 2026-11-15: TRT / compile rows with cold and cached start

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" || exit 1
git pull --rebase origin work/ports || { git rebase --abort; echo "publish: rebase failed, commit stays local"; exit 1; }
git push origin HEAD:work/ports
