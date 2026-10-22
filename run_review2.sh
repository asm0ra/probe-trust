#!/usr/bin/env bash
# Review response, part 2. Q1: an attacker that learns to distinguish synthetic probes from real windows using ITS OWN probes (adapt_unk = the same family as the server's; adapt_gen = another family: noise 0.05-0.35 sigma, shapes, ranges) with penalty lam in {0,5,20}. Q2: attacks not built around the rule: fang, scaleup, noise (+Min-Max already in rev_P3).
# Results results/rev2_<phase>_<dataset>.jsonl; resumable by flags. Usage: bash run_review2.sh [phases]
export PT_COMPILE=${PT_COMPILE:-1} CPATH=/mnt/projects/tools/pydev/python3.14:$CPATH
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-../venv/bin/python}
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24); PH=${1:-"Q1 Q2"}; B="--norm 1 --delta 0.5 --tau 0.1"
for P in $PH; do for DS in metr_la pems_bay pemsd7m; do C="--dataset $DS $B --chunk ${CH[$DS]}"
 case $P in
 Q1) for A in adapt_unk adapt_gen; do for L in 0 5 20; do F=../results/rev2_Q1_${DS}_${A}_$L.flag; [ -f $F ] || { $PY driver.py $C --out ../results/rev2_Q1_$DS.jsonl --seeds 0,1,2,3,4 --methods fedavg,fltrust,probe2,probe2_fl --shares 0.4,0.6 --attack $A --lam $L && touch $F; }; done; done;;
 Q2) for A in fang scaleup noise; do F=../results/rev2_Q2_${DS}_$A.flag; [ -f $F ] || { $PY driver.py $C --out ../results/rev2_Q2_$DS.jsonl --seeds 0,1,2,3,4 --methods fedavg,median,krum,fltrust,probe2,probe2_fl --shares 0.4,0.6 --attack $A && touch $F; }; done;;
 esac
done; done
echo DONE > ../results/rev2_done.flag
