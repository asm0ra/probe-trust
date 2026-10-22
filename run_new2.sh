#!/usr/bin/env bash
# Background checks of 2026-10-19: two-sided probes (probe2), Probe+FLTrust (probe2_fl), backdoor, ALIE, flip with the probe applied to the submitted model. 3 datasets, seeds 0-4.
cd /mnt/projects/phd-fl-gpu/src; S=0,1,2,3,4; PY=../venv/bin/python
for DS in metr_la pems_bay pemsd7m; do
 O=../results/new2_$DS.jsonl; rm -f $O; C="--dataset $DS --device cuda --max_batch 15 --out $O"
 $PY bfl.py --methods probe2,probe2_fl --conditions clean --seeds $S --bad 0.2 $C
 $PY bfl.py --methods fedavg --conditions clean --seeds $S --bad 0.2 --attack backdoor $C
 for A in hide exagg bias; do MAG=25; [ $A = bias ] && MAG=8
  for B in 0.2 0.4 0.6; do $PY bfl.py --methods probe2,probe2_fl --conditions atk --seeds $S --bad $B --attack $A --hide_mag $MAG $C; done; done
 for B in 0.3 0.5; do $PY bfl.py --methods fedavg,median,fltrust,probe,probe2,probe2_fl --conditions atk --seeds $S --bad $B --attack flip $C; done
 for A in backdoor alie; do for B in 0.2 0.4 0.6; do $PY bfl.py --methods fedavg,median,krum40,fltrust,probe,probe2,probe2_fl --conditions atk --seeds $S --bad $B --attack $A $C; done; done
done
echo DONE > ../results/new2_done.flag
