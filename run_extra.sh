#!/usr/bin/env bash
# Additional checks of Probe-Trust: heterogeneous data (a node has 30 % of the days) and other attacks (flip, noise), to make sure the method is not tailored to "hiding" only. Seeds 0-4.
cd /mnt/projects/phd-fl-gpu/src; S=0,1,2,3,4; D=cuda; O=../results/extra.jsonl; rm -f $O
for B in 0.3 0.5; do
 ../venv/bin/python bfl.py --methods fedavg,krum40,probe --conditions atk --seeds $S --bad $B --attack hide --noniid 1 --device $D --max_batch 10 --out $O
 for A in flip noise; do ../venv/bin/python bfl.py --methods fedavg,median,probe --conditions atk --seeds $S --bad $B --attack $A --device $D --max_batch 10 --out $O; done
done
../venv/bin/python bfl.py --methods fedavg,krum40,probe --conditions clean --seeds $S --bad 0.2 --noniid 1 --device $D --max_batch 10 --out $O
