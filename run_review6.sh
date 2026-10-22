#!/usr/bin/env bash
# Review response, part 6: second domain, solar power forecasting (NREL Solar-Energy, 137 plants, 10 min). Same thresholds (delta=0.5 sigma, tau=0.1 sigma), no tuning. Results results/rev6_S_solar.jsonl
ROOTDIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"; cd $ROOTDIR/src; PY=${PY:-../venv/bin/python}; mkdir -p ../results ../logs
export PT_DOMAIN=solar PT_HIDE_THR=${PT_HIDE_THR:-3.2} PT_EXAGG_THR=${PT_EXAGG_THR:-12.9}; B="--norm 1 --delta ${DELTA:-0.5} --tau ${TAU:-0.1} --hide_mag 20 --chunk 40"; S5=0,1,2,3,4; TAG=${TAG:-rev6_S}
for A in hide exagg bias; do F=../results/${TAG}_$A.flag; [ -f $F ] || { $PY driver.py --dataset solar $B --out ../results/${TAG}_solar.jsonl --seeds $S5 --methods fedavg,median,krum,fltrust,probe2,probe2_fl --shares 0.4,0.6 --attack $A && touch $F; }; done
F=../results/${TAG}_clean.flag; [ -f $F ] || { $PY driver.py --dataset solar $B --out ../results/${TAG}_solar.jsonl --seeds $S5 --methods fedavg,fltrust,probe2,probe2_fl --clean 1 && touch $F; }
echo DONE > ../results/${TAG}_done.flag
