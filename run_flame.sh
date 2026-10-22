#!/usr/bin/env bash
# FLAME-like baseline (clustering of cosine distances + norm clipping, no noise): attacks hide/exagg/bias/flip, shares 0.2/0.4/0.6, 5 seeds, 3 datasets + clean. Results results/flame_<dataset>.jsonl.
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-../venv/bin/python}
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24); M=flame_like
for DS in metr_la pems_bay pemsd7m; do
 O=../results/flame_$DS.jsonl; C="--dataset $DS --norm 1 --delta 0.5 --tau 0.1 --chunk ${CH[$DS]} --out $O --seeds 0,1,2,3,4 --methods $M"
 [ -f ../results/flame_${DS}_clean.flag ] || { $PY driver.py $C --clean 1 && touch ../results/flame_${DS}_clean.flag; }
 for A in hide exagg bias flip; do [ -f ../results/flame_${DS}_$A.flag ] || { $PY driver.py $C --attack $A && touch ../results/flame_${DS}_$A.flag; }; done
done
