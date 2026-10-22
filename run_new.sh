#!/usr/bin/env bash
# Independent check of Probe-Trust (2026-10-19): attacks with a different signature (exagg = false jam, bias = +8 shift on all windows), hide for comparison; the FLTrust competitor. 3 datasets, seeds 0-4.
cd /mnt/projects/phd-fl-gpu/src; S=0,1,2,3,4; M=fedavg,median,krum40,fltrust,probe,probe_clip
for DS in metr_la pems_bay pemsd7m; do
 O=../results/new_$DS.jsonl; rm -f $O
 ../venv/bin/python bfl.py --methods $M --conditions clean --seeds $S --bad 0.2 --dataset $DS --device cuda --max_batch 10 --out $O
 for A in hide exagg bias; do
  MAG=25; [ $A = bias ] && MAG=8
  for B in 0.2 0.4 0.6; do ../venv/bin/python bfl.py --methods $M --conditions atk --seeds $S --bad $B --attack $A --hide_mag $MAG --dataset $DS --device cuda --max_batch 10 --out $O; done
 done
done
echo DONE > ../results/new_done.flag
