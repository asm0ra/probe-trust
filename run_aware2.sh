#!/usr/bin/env bash
# An attacker that knows the probe windows but honestly trains on them with only a small part of the batch (k of 64), so as not to "dilute" its own attack. 40 % attackers.
cd /mnt/projects/phd-fl-gpu/src; S=0,1,2,3,4; D=cuda; O=../results/aware2.jsonl; rm -f $O
for K in 1 2 4 8; do
 ../venv/bin/python bfl.py --methods fedavg,probe --conditions atk --seeds $S --bad 0.4 --attack hide_aware --aware_k $K --device $D --max_batch 10 --out $O
 ../venv/bin/python bfl.py --methods probe --conditions atk --seeds $S --bad 0.4 --attack hide_aware --aware_k $K --rotate 1 --device $D --max_batch 10 --out $O
done
