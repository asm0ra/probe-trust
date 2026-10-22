#!/usr/bin/env bash
# Optimising attacker with a tuned penalty weight lam (the worst case for the defence), 40 and 60 %, seeds 0-4.
cd /mnt/projects/phd-fl-gpu/src; S=0,1,2,3,4; D=cuda; O=../results/lam.jsonl; rm -f $O
for L in 0.02 0.1 0.3 1 2; do for B in 0.4 0.6; do ../venv/bin/python bfl.py --methods fedavg,probe,probe_clip --conditions atk --seeds $S --bad $B --attack hide_opt --lam $L --device $D --max_batch 10 --out $O; done; done
