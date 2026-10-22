#!/usr/bin/env bash
cd /mnt/projects/phd-fl-gpu/agentic
for p in honest hide exagg hide_aware; do ../venv/bin/python -W ignore run_agents.py --pool $p; done
for a in hide exagg hide_aware; do ../venv/bin/python -W ignore coord.py $a; done
echo ALL_DONE
