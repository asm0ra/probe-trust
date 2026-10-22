#!/usr/bin/env bash
# Review response (2026-10-21). Phases by priority: P1 sweep of attack strength (including inside the dead zone delta=0.5 sigma = 6.4 mph), P2 misspecified probe ranges, P3 Min-Max, P4 15 more seeds (5-19) for the key cells at 60 %.
# Results results/rev_<phase>_<dataset>.jsonl; flags results/rev_<phase>_<dataset>_<...>.flag; resumable. Usage: bash run_review.sh [phases, e.g. "P1 P2"]
export PT_COMPILE=${PT_COMPILE:-1} CPATH=/mnt/projects/tools/pydev/python3.14:$CPATH  # 1.75x speed-up: torch.compile of the forward pass + fused Adam (Python headers unpacked without sudo)
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-../venv/bin/python}
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24); PH=${1:-"P1 P2 P3 P4"}; B="--norm 1 --delta 0.5 --tau 0.1"
for P in $PH; do for DS in metr_la pems_bay pemsd7m; do C="--dataset $DS $B --chunk ${CH[$DS]}"
 case $P in
 P1) for A in hide exagg bias; do for M in 4 8 12; do F=../results/rev_P1_${DS}_${A}_$M.flag; [ -f $F ] || { $PY driver.py $C --out ../results/rev_P1_$DS.jsonl --seeds 0,1,2,3,4 --methods fedavg,median,fltrust,probe2 --shares 0.4,0.6 --attack $A --hide_mag $M && touch $F; }; done; done;;
 P2) for R in "10,25,40,75" "0.5,3,85,99"; do for A in hide exagg bias; do F=../results/rev_P2_${DS}_${R//,/_}_$A.flag; [ -f $F ] || { PT_PROBERANGE=$R $PY driver.py $C --out ../results/rev_P2_${DS}_${R//,/_}.jsonl --seeds 0,1,2,3,4 --methods probe2 --shares 0.4,0.6 --attack $A && touch $F; }; done; done;;
 P3) F=../results/rev_P3_${DS}.flag; [ -f $F ] || { $PY driver.py $C --out ../results/rev_P3_$DS.jsonl --seeds 0,1,2,3,4 --methods fedavg,median,krum,fltrust,probe2,probe2_fl --shares 0.4,0.6 --attack minmax && touch $F; };;
 P4) for A in hide exagg bias; do F=../results/rev_P4_${DS}_$A.flag; [ -f $F ] || { $PY driver.py $C --out ../results/rev_P4_$DS.jsonl --seeds 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --methods fedavg,median,fltrust,probe2,probe2_rel --shares 0.6 --attack $A && touch $F; }; done;;
 esac
done; done
echo DONE > ../results/rev_done.flag
