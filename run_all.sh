#!/usr/bin/env bash
# Reproduction of all tables and figures of the paper from scratch (about 2-4 h on one AMD RX 9070 XT GPU / ROCm; one CPU thread is enough for the analysis).
# Usage: bash run_all.sh [--analysis-only]   (--analysis-only: skip the computations and rebuild only the tables/figures from results/*.jsonl)
set -eu
cd /mnt/projects/phd-fl-gpu; PY=venv/bin/python
if [ "${1:-}" != "--analysis-only" ]; then
  # 0. data: data/metr_la.h5, adj_mx.pkl; PEMS-BAY.csv (Zenodo 5724362); pemsd7m/ (PeMSD7_V_228.csv, PeMSD7_W_228.csv); see README.md
  # 1. choice of the normalised thresholds delta, tau ONLY on METR-LA (seeds 100-101), then selection
  (cd src && rm -f ../results/norm_tune.jsonl && ../venv/bin/python tune_norm.py && ../venv/bin/python pick_norm.py)
  # 2. main grid (11 methods x 6 attacks x shares 0.2/0.4/0.6 x 5 seeds x 3 networks + clean) and adaptive attackers
  rm -f results/final_*.flag results/adapt_*.flag results/final_*.jsonl results/adapt_*.jsonl
  bash run_final.sh
  bash run_clip.sh      # extension: Probe2 + norm clipping (against flip)
fi
# 3. analysis: tables/figures, causal analysis + SHAP/LIME, defence-selection agent
$PY analysis/tables.py --tag final
$PY analysis/causal.py --glob "final_*.jsonl" --tag final
$PY analysis/agent.py --glob "final_*.jsonl" --tag final
echo "done: results/analysis/*, figures_final/*"
