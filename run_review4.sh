#!/usr/bin/env bash
# Review response, part 4: Zeno++ and FLAME (with noise) from the papers; main grid 5 attacks x 0.4/0.6 x 5 seeds x 3 datasets. Results results/rev4_<dataset>.jsonl.
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-/venv/main/bin/python}
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24); B="--norm 1 --delta 0.5 --tau 0.1"
for DS in metr_la pems_bay pemsd7m; do for A in hide exagg bias flip alie; do F=../results/rev4_${DS}_$A.flag; [ -f $F ] || { $PY driver.py --dataset $DS $B --chunk ${CH[$DS]} --out ../results/rev4_$DS.jsonl --seeds 0,1,2,3,4 --methods zenopp,flame_full,flame_like --shares 0.4,0.6 --attack $A && touch $F; }; done; done
echo DONE > ../results/rev4_done.flag
