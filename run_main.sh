#!/usr/bin/env bash
# Main test of Probe-Trust (test seeds 0-4; calibration seeds 100-101 do not overlap): methods x attacker share 20-60 % x 5 seeds + a clean run.
cd /mnt/projects/phd-fl-gpu/src; OUT=../results/main_probe.jsonl; rm -f $OUT
M=fedavg,median,trimmed,krum40,gct,probe
../venv/bin/python bfl.py --methods $M --conditions clean --seeds 0,1,2,3,4 --bad 0.2 --device cuda --max_batch 10 --out $OUT
for B in 0.2 0.3 0.4 0.5 0.6; do ../venv/bin/python bfl.py --methods $M --conditions atk --seeds 0,1,2,3,4 --bad $B --device cuda --max_batch 10 --out $OUT; done
