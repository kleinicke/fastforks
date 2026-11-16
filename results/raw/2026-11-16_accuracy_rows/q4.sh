#!/bin/bash
. /home/florian/projects/experiments/017-fastforks-lib/wt-accrows/results/raw/2026-11-16_accuracy_rows/env.sh
SV=/home/florian/.cache/ff_r5_tmp/accrows_sv; RP=$E/ref-venvs/seedvr/.venv/bin/python; FW=/home/florian/projects/data/ff-weights
cat $FW/seedvr2/seedvr2-3b.safetensors $FW/seedvr2/seedvr2-vae.safetensors $FW/seedvr2-src/seedvr2_ema_3b.pth > /dev/null
for c in d0801 d0826 d0851 d0876; do
  [ -f $SV/ref/3b_$c.npz ] || $L $RP $W/tools/seedvr2_reference.py $E/upstream/SeedVR $FW/seedvr2 --src $FW/seedvr2-src --model 3b --input $SV/in/$c.npz --scale 2 --out $SV/ref/3b_$c.npz > $R/ref_3b_$c.log 2>&1 || echo "FAIL ref 3b $c"
done
$L $PY $R/seedvr2_eval.py 3b $R/seedvr2.jsonl --crops d0801 d0826 d0851 d0876 --clips synth car-turn breakdance mallard-water || echo "FAIL eval 3b"
cat $FW/seedvr2/seedvr2-7b.safetensors > /dev/null
for c in d0801 d0826; do
  [ -f $SV/ref/7b_$c.npz ] || $L $RP $W/tools/seedvr2_reference.py $E/upstream/SeedVR $FW/seedvr2 --src $FW/seedvr2-src --model 7b --offload --input $SV/in/$c.npz --scale 2 --out $SV/ref/7b_$c.npz > $R/ref_7b_$c.log 2>&1 || echo "FAIL ref 7b $c"
done
$L $PY $R/seedvr2_eval.py 7b $R/seedvr2.jsonl --crops d0801 d0826 --clips synth car-turn || echo "FAIL eval 7b"
echo DONE
