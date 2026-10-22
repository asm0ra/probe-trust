#!/usr/bin/env bash
# Ablation "absolute rule vs cohort-relative" and a simplified FedMID: probe2_rel (same probes, scored by the cohort median/MAD), fedmid_like (random inputs, relational distances).
# Attacks hide/exagg/bias, shares 0.2/0.4/0.6, 5 seeds, 3 datasets + clean runs. Results results/abl_<dataset>.jsonl. Waits for run_sens.sh to finish (one GPU).
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-../venv/bin/python}
while [ ! -f ../results/sens_done.flag ] && [ -z "$NOWAIT" ]; do sleep 30; done
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24); M=probe2_rel,fedmid_like
for DS in metr_la pems_bay pemsd7m; do
 O=../results/abl_$DS.jsonl; C="--dataset $DS --norm 1 --delta 0.5 --tau 0.1 --chunk ${CH[$DS]} --out $O --seeds 0,1,2,3,4 --methods $M"
 [ -f ../results/abl_${DS}_clean.flag ] || { $PY driver.py $C --clean 1 && touch ../results/abl_${DS}_clean.flag; }
 for A in hide exagg bias; do [ -f ../results/abl_${DS}_$A.flag ] || { $PY driver.py $C --attack $A && touch ../results/abl_${DS}_$A.flag; }; done
done
echo DONE > ../results/abl_done.flag
