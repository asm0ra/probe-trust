#!/usr/bin/env bash
# Choice of delta/tau of the two-sided probe on ANOTHER attack (exagg), seeds 100-101, METR-LA; for control the same parameters on hide (end-to-end sensitivity) and a clean run.
cd /mnt/projects/phd-fl-gpu/src; PY=../venv/bin/python; O=../results/tune.jsonl; rm -f $O
for D in 5 10 15; do for T in 0.5 1 2; do
 $PY bfl.py --methods probe2 --conditions clean,atk --seeds 100,101 --bad 0.4 --attack exagg --delta $D --tau $T --device cuda --max_batch 10 --out $O
 $PY bfl.py --methods probe2 --conditions atk --seeds 100,101 --bad 0.4 --attack hide --delta $D --tau $T --device cuda --max_batch 10 --out $O
done; done
$PY bfl.py --methods fedavg --conditions clean,atk --seeds 100,101 --bad 0.4 --attack exagg --device cuda --out $O
$PY bfl.py --methods fedavg --conditions atk --seeds 100,101 --bad 0.4 --attack hide --device cuda --out $O
echo DONE > ../results/tune_done.flag
