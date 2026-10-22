#!/usr/bin/env bash
# Robustness check of Probe-Trust: (A) the attacker knows the probe windows (hide_aware), fixed probes; (B) the same, but the probes change every round; (C) weak attacks (hide_mag 5/10/15 mph).
cd /mnt/projects/phd-fl-gpu/src; S=0,1,2,3,4; D=cuda
O=../results/adaptive.jsonl; rm -f $O
for B in 0.3 0.5; do
 ../venv/bin/python bfl.py --methods fedavg,krum40,probe --conditions atk --seeds $S --bad $B --attack hide_aware --device $D --max_batch 10 --out $O
 ../venv/bin/python bfl.py --methods probe --conditions atk --seeds $S --bad $B --attack hide_aware --rotate 1 --device $D --max_batch 10 --out $O
done
for MAG in 5 10 15; do ../venv/bin/python bfl.py --methods fedavg,probe --conditions atk --seeds $S --bad 0.4 --attack hide --hide_mag $MAG --device $D --max_batch 10 --out $O; done
