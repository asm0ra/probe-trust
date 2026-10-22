#!/usr/bin/env bash
# Final grid for the paper (2026-10-20), 5 seeds, shares 0.2/0.4/0.6, normalised probes (delta, tau from results/norm_choice.json, chosen only on METR-LA seeds 100-101).
# Phase A: the main balanced grid (11 methods x 6 attacks x 3 shares x 5 seeds x 3 datasets + clean); phase B: adaptive attackers. The GPU guard is inside driver.py.
# Usage: bash run_final.sh   (results results/final_<dataset>.jsonl and results/adapt_<dataset>.jsonl)
cd /mnt/projects/phd-fl-gpu/src; PY=../venv/bin/python
read DK TK < <($PY -c "import json;c=json.load(open('../results/norm_choice.json'));print(c['delta'],c['tau'])")
M=fedavg,median,trimmed,krum,normbound,bulyan,foolsgold,fltrust,probe,probe2,probe2_fl; S=0,1,2,3,4
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24)
for DS in metr_la pems_bay pemsd7m; do
 O=../results/final_$DS.jsonl; C="--dataset $DS --norm 1 --delta $DK --tau $TK --chunk ${CH[$DS]} --out $O --seeds $S --methods $M"
 [ -f ../results/final_${DS}_clean.flag ] || { $PY driver.py $C --clean 1 && touch ../results/final_${DS}_clean.flag; }
 for A in hide exagg bias backdoor alie flip; do
  [ -f ../results/final_${DS}_$A.flag ] && continue
  $PY driver.py $C --attack $A && touch ../results/final_${DS}_$A.flag
 done
done
echo A_DONE > ../results/final_A_done.flag
MA=fedavg,fltrust,probe2,probe2_fl,bulyan
for DS in metr_la pems_bay pemsd7m; do
 O=../results/adapt_$DS.jsonl
 for A in adapt_known adapt_unk adapt_fl adapt_all; do
  [ -f ../results/adapt_${DS}_$A.flag ] && continue
  $PY driver.py --dataset $DS --norm 1 --delta $DK --tau $TK --chunk ${CH[$DS]} --out $O --seeds $S --methods $MA --shares 0.4,0.6 --attack $A && touch ../results/adapt_${DS}_$A.flag
 done
 [ -f ../results/adapt_${DS}_collude.flag ] || { $PY driver.py --dataset $DS --norm 1 --delta $DK --tau $TK --chunk ${CH[$DS]} --out $O --seeds $S --methods $MA --shares 0.4,0.6 --attack adapt_all --collude 1 && touch ../results/adapt_${DS}_collude.flag; }
done
echo B_DONE > ../results/final_B_done.flag
