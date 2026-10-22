#!/usr/bin/env bash
cd /mnt/projects/phd-fl-gpu/src; S=0,1,2,3,4; D=cuda; O=../results/aware.jsonl; rm -f $O
for B in 0.3 0.5; do
 ../venv/bin/python bfl.py --methods fedavg,krum40,probe --conditions atk --seeds $S --bad $B --attack hide_aware --device $D --max_batch 10 --out $O
 ../venv/bin/python bfl.py --methods probe --conditions atk --seeds $S --bad $B --attack hide_aware --rotate 1 --device $D --max_batch 10 --out $O
done
