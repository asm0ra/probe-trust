#!/usr/bin/env bash
# Review response, part 3 (on a rented L40S). Q3: sweep of the probe share in the adaptive attacker's training (aware_k), the "attack strength vs stealth" boundary; N: strong non-IID (--noniid 1), attacks + clean runs; A: another architecture (PT_ARCH=tf|mlp), main grid.
# Results results/rev3_<phase>_<dataset>.jsonl; resumable by flags. Usage: bash run_review3.sh [phases: Q3 N A]
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-/venv/main/bin/python}; mkdir -p ../results ../logs
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24); PH=${1:-"Q3 N A"}; B="--norm 1 --delta 0.5 --tau 0.1"; S5=0,1,2,3,4
for P in $PH; do for DS in metr_la pems_bay pemsd7m; do C="--dataset $DS $B --chunk ${CH[$DS]}"
 case $P in
 Q3) for A in adapt_known adapt_unk adapt_gen; do for K in 4 8 16 48; do F=../results/rev3_Q3_${DS}_${A}_$K.flag; [ -f $F ] || { $PY driver.py $C --out ../results/rev3_Q3_$DS.jsonl --seeds $S5 --methods fedavg,fltrust,probe2,probe2_fl --shares 0.4,0.6 --attack $A --lam 0 --aware_k $K && touch $F; }; done; done;;
 N) for A in hide exagg bias; do F=../results/rev3_N_${DS}_$A.flag; [ -f $F ] || { $PY driver.py $C --out ../results/rev3_N_$DS.jsonl --seeds $S5 --methods fedavg,median,krum,fltrust,probe2,probe2_fl --shares 0.4,0.6 --attack $A --noniid 1 && touch $F; }; done
     F=../results/rev3_N_${DS}_clean.flag; [ -f $F ] || { $PY driver.py $C --out ../results/rev3_N_$DS.jsonl --seeds $S5 --methods fedavg,fltrust,probe2,probe2_fl --clean 1 --noniid 1 && touch $F; };;
 A) for AR in tf mlp; do for A in hide exagg bias; do F=../results/rev3_A_${DS}_${AR}_$A.flag; [ -f $F ] || { PT_ARCH=$AR $PY driver.py $C --out ../results/rev3_A_${AR}_$DS.jsonl --seeds $S5 --methods fedavg,median,krum,fltrust,probe2,probe2_fl --shares 0.4,0.6 --attack $A && touch $F; }; done
     F=../results/rev3_A_${DS}_${AR}_clean.flag; [ -f $F ] || { PT_ARCH=$AR $PY driver.py $C --out ../results/rev3_A_${AR}_$DS.jsonl --seeds $S5 --methods fedavg,fltrust,probe2,probe2_fl --clean 1 && touch $F; }; done;;
 esac
done; done
echo DONE > ../results/rev3_done.flag
