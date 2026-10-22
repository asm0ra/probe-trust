#!/usr/bin/env bash
# Cost of the thresholds on clean data (overhead): the same delta, tau combinations as in run_sens.sh, without attacks. Results results/sensclean_<dataset>.jsonl.
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-../venv/bin/python}
declare -A CH=([metr_la]=40 [pemsd7m]=32 [pems_bay]=24)
for DS in metr_la pems_bay pemsd7m; do for DT in "0.25 0.1" "1.0 0.1" "0.5 0.05" "0.5 0.2"; do set -- $DT; D=$1; T=$2
 F=../results/sensclean_${DS}_${D}_${T}.flag; [ -f $F ] && continue
 $PY driver.py --dataset $DS --norm 1 --delta $D --tau $T --chunk ${CH[$DS]} --out ../results/sensclean_$DS.jsonl --seeds 0,1,2,3,4 --methods probe2 --clean 1 && touch $F
done; done
