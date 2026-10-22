#!/usr/bin/env bash
# Sensitivity of Probe-Trust to delta and tau (in fractions of sigma): one parameter at a time around the base (0.5; 0.1), the base is in the main grid final_*, attacks hide/exagg/bias, shares 0.4/0.6, 5 seeds, 3 datasets. Results results/sens_<dataset>.jsonl; flags results/sens_<dataset>_<delta>_<tau>_<attack>.flag
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-../venv/bin/python}
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24)
for DS in metr_la pems_bay pemsd7m; do
 for DT in "0.25 0.1" "1.0 0.1" "0.5 0.05" "0.5 0.2"; do set -- $DT; D=$1; T=$2; for A in hide exagg bias; do
  F=../results/sens_${DS}_${D}_${T}_${A}.flag; [ -f $F ] && continue
  $PY driver.py --dataset $DS --norm 1 --delta $D --tau $T --chunk ${CH[$DS]} --out ../results/sens_$DS.jsonl --seeds 0,1,2,3,4 --methods probe2 --shares 0.4,0.6 --attack $A && touch $F
 done; done
done
echo DONE > ../results/sens_done.flag
