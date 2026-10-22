#!/usr/bin/env bash
# Main test of Probe-Trust on PEMS-BAY and PeMS-D7(M): methods x share 20-60 % x 5 seeds + a clean run (the "hide the jam" attack).
cd /mnt/projects/phd-fl-gpu/src; M=fedavg,median,trimmed,krum40,gct,probe
for DS in pemsd7m pems_bay; do
  OUT=../results/main_$DS.jsonl; rm -f $OUT
  ../venv/bin/python bfl.py --dataset $DS --methods $M --conditions clean --seeds 0,1,2,3,4 --bad 0.2 --device cuda --max_batch 10 --out $OUT
  for B in 0.2 0.3 0.4 0.5 0.6; do ../venv/bin/python bfl.py --dataset $DS --methods $M --conditions atk --seeds 0,1,2,3,4 --bad $B --device cuda --max_batch 10 --out $OUT; done
done
