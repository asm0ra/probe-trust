#!/usr/bin/env bash
# Review response, part 5: graph model (PT_ARCH=gcn: GRU + a "mean of graph neighbours" channel), main grid + clean runs. Results results/rev5_G_<dataset>.jsonl
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-../venv/bin/python}; mkdir -p ../results ../logs
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24); B="--norm 1 --delta 0.5 --tau 0.1"; S5=0,1,2,3,4
for DS in metr_la pems_bay pemsd7m; do C="--dataset $DS $B --chunk ${CH[$DS]}"
 for A in hide exagg bias; do F=../results/rev5_G_${DS}_$A.flag; [ -f $F ] || { PT_ARCH=gcn $PY driver.py $C --out ../results/rev5_G_$DS.jsonl --seeds $S5 --methods fedavg,median,krum,fltrust,probe2,probe2_fl --shares 0.4,0.6 --attack $A && touch $F; }; done
 F=../results/rev5_G_${DS}_clean.flag; [ -f $F ] || { PT_ARCH=gcn $PY driver.py $C --out ../results/rev5_G_$DS.jsonl --seeds $S5 --methods fedavg,fltrust,probe2,probe2_fl --clean 1 && touch $F; }
done; echo DONE > ../results/rev5_done.flag
