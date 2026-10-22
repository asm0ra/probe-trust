#!/usr/bin/env bash
# Probes from the public history of the same network (PT_PROBEGEN=same): main Probe-Trust grid, 5 repetitions. Results results/same_<dataset>.jsonl
cd "$(dirname "$(readlink -f "$0")")/src"; PY=../venv/bin/python; B="--norm 1 --delta 0.5 --tau 0.1"; S5=0,1,2,3,4; export PT_PROBEGEN=${PT_PROBEGEN:-same}; TAG=${TAG:-same}
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24)
for DS in metr_la pems_bay pemsd7m; do C="--dataset $DS $B --chunk ${CH[$DS]}"
 for A in hide exagg bias; do F=../results/${TAG}_${DS}_$A.flag; [ -f $F ] || { $PY driver.py $C --out ../results/${TAG}_$DS.jsonl --seeds $S5 --methods fedavg,probe2,probe2_fl --shares 0.4,0.6 --attack $A && touch $F; }; done
 for A in adapt_known adapt_gen; do F=../results/${TAG}_${DS}_$A.flag; [ -f $F ] || { $PY driver.py $C --out ../results/${TAG}_$DS.jsonl --seeds $S5 --methods fedavg,probe2,probe2_fl --shares 0.6 --attack $A --lam 0 --aware_k 8 && touch $F; }; done
 F=../results/${TAG}_${DS}_clean.flag; [ -f $F ] || { $PY driver.py $C --out ../results/${TAG}_$DS.jsonl --seeds $S5 --methods fedavg,fltrust,probe2,probe2_fl --clean 1 && touch $F; }
done; echo DONE > ../results/${TAG}_done.flag
