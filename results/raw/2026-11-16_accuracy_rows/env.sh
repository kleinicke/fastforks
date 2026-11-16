E=/home/florian/projects/experiments/017-fastforks-lib; W=$E/wt-accrows; R=$W/results/raw/2026-11-16_accuracy_rows
export FASTFORKS_DATA=/home/florian/projects/data FASTFORKS_WEIGHTS_PATH=/home/florian/projects/data/ff-weights FASTFORKS_GOLDEN_PATH=$E/golden PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FASTFORKS_NONCOMMERCIAL=1
source $E/review/kernel_path.sh; export FASTFORKS_KERNEL_PATH=$KP
export PYTHONPATH=$W/src:$W/tools
PY=$E/wt-ports/.venv/bin/python; L="/home/florian/projects/shared/bin/gpulock-low timeout 290"
cd $W
