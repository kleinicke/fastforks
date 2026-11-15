cd /home/florian/projects/experiments/017-fastforks-lib/wt-ports-gaps
export PYTHONPATH=src
for r in $(cat results/raw/2026-11-15_gaps/solve/uncached_refs.txt); do ../fastforks/.venv/bin/python -m fastforks download $r --dry-run 2>&1 | grep -v "^note"; done > results/raw/2026-11-15_gaps/solve/dry.txt
