#!/usr/bin/env bash
# Parameters of the two-sided probe chosen on exagg (delta=5, tau=0.5; seeds 100-101), checked on the other attacks and datasets, seeds 0-4.
cd /mnt/projects/phd-fl-gpu/src; S=0,1,2,3,4; PY=../venv/bin/python
for DS in metr_la pems_bay pemsd7m; do
 O=../results/new3_$DS.jsonl; rm -f $O; C="--dataset $DS --device cuda --max_batch 15 --delta 5 --tau 0.5 --out $O"
 $PY bfl.py --methods probe2,probe2_fl --conditions clean --seeds $S --bad 0.2 $C
 for A in hide exagg bias backdoor; do MAG=25; [ $A = bias ] && MAG=8
  for B in 0.2 0.4 0.6; do $PY bfl.py --methods probe2,probe2_fl --conditions atk --seeds $S --bad $B --attack $A --hide_mag $MAG $C; done; done
done
echo DONE > ../results/new3_done.flag
