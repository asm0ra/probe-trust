#!/usr/bin/env bash
# Extension: Probe2 + clipping of the update norm (probe2_clip) against flip (the weak spot of Probe2) and clean runs; same thresholds (norm_choice.json).
cd /mnt/projects/phd-fl-gpu/src; PY=../venv/bin/python
read DK TK < <($PY -c "import json;c=json.load(open('../results/norm_choice.json'));print(c['delta'],c['tau'])")
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24)
for DS in metr_la pems_bay pemsd7m; do
 O=../results/ext_$DS.jsonl; C="--dataset $DS --norm 1 --delta $DK --tau $TK --chunk ${CH[$DS]} --out $O --seeds 0,1,2,3,4 --methods probe2_clip"
 $PY driver.py $C --clean 1; for A in flip hide exagg; do $PY driver.py $C --attack $A; done
done
echo DONE > ../results/ext_done.flag
