#!/usr/bin/env bash
# Stronger checks of Probe-Trust, seeds 0-4: (A) an optimising adaptive attacker (penalty for violating the physics on its own probes, lam=5), 40 and 60 %;
# (B) an attacker "below the threshold" (hide_mag 6/8/9/12) at 40 and 60 %; (C) collusion (attackers form a neighbouring cluster) at 30 and 50 %.
cd /mnt/projects/phd-fl-gpu/src; S=0,1,2,3,4; D=cuda; O=../results/adv.jsonl; rm -f $O
for B in 0.4 0.6; do ../venv/bin/python bfl.py --methods fedavg,krum40,probe,probe_clip --conditions atk --seeds $S --bad $B --attack hide_opt --lam 5 --device $D --max_batch 10 --out $O; done
for MAG in 6 8 9 12; do for B in 0.4 0.6; do ../venv/bin/python bfl.py --methods fedavg,probe --conditions atk --seeds $S --bad $B --attack hide --hide_mag $MAG --device $D --max_batch 10 --out $O; done; done
for B in 0.3 0.5; do ../venv/bin/python bfl.py --methods fedavg,krum40,probe --conditions collude --seeds $S --bad $B --attack hide --device $D --max_batch 10 --out $O; done
