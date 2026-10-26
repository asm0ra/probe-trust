#!/usr/bin/env bash
cd /mnt/projects/phd-fl-gpu/agentic_tools
export BUDGET=3.8
for p in honest hide exagg; do python3 run_tool_agents.py --pool $p --workers 8; done
OUT_DIR=out_tools TAG=_tools ATTACKS=hide,exagg ../venv/bin/python -W ignore analyze.py > analyze.log 2>&1
echo ALL_DONE
