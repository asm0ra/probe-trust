#!/usr/bin/env bash
# Weak FLTrust root (2026-10-20): root scenarios x attacks x shares, 5 seeds, normalised probes (delta=0.5 sigma, tau=0.1 sigma). Plan and criterion: results/rootweak_plan.md
# Usage: bash run_rootweak.sh <dataset> (metr_la | pems_bay | pemsd7m); results results/rootweak_<dataset>.jsonl; flags results/rootweak_<dataset>_<scenario>_<block>.flag
cd /mnt/projects/phd-fl-gpu/src; PY=../venv/bin/python; DS=${1:-metr_la}; O=../results/rootweak_$DS.jsonl
read DK TK < <($PY -c "import json;c=json.load(open('../results/norm_choice.json'));print(c['delta'],c['tau'])")
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24)
M=fedavg,fltrust,probe2,probe2_fl; S=0,1,2,3,4
declare -A SC=([ctrl]="" [small2]="--root_n 2" [small5]="--root_n 5" [bias10]="--root_mode cluster" [poison3]="--root_poison 3" [noniid]="--noniid 1")
for s in ctrl small2 small5 bias10 poison3 noniid; do
 C="--dataset $DS --norm 1 --delta $DK --tau $TK --chunk ${CH[$DS]} --out $O --seeds $S --methods $M ${SC[$s]}"
 [ -f ../results/rootweak_${DS}_${s}_clean.flag ] || { $PY driver.py $C --clean 1 && touch ../results/rootweak_${DS}_${s}_clean.flag; }
 for A in hide exagg bias flip; do
  [ -f ../results/rootweak_${DS}_${s}_$A.flag ] || { $PY driver.py $C --attack $A --shares 0.4,0.6 && touch ../results/rootweak_${DS}_${s}_$A.flag; }
 done
done
echo DONE > ../results/rootweak_${DS}_done.flag
