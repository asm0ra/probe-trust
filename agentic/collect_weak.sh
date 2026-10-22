#!/usr/bin/env bash
cd "$(dirname "$(readlink -f "$0")")"
for p in hide10 hide5 exagg10; do ${PY:-../venv/bin/python} -W ignore run_agents.py --pool $p; done
echo WEAK_DONE > logs/weak_done.flag
