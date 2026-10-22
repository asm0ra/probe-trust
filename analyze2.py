#!/usr/bin/env python3
"""Summary tables of the checks of 2026-10-19 (RMSE growth of honest nodes, % to clean FedAvg; backdoor: forecast shift, mph). Usage: venv/bin/python analyze2.py [ds ...]"""
import json, sys, os, statistics as st
R = os.path.dirname(os.path.abspath(__file__)) + '/results/'
def L(f): return [json.loads(l) for l in open(R + f)] if os.path.exists(R + f) else []
def mean(v): return st.mean(v) if v else float('nan')
for ds in (sys.argv[1:] or ['metr_la', 'pems_bay', 'pemsd7m']):
    A = L(f'new_{ds}.jsonl') + L(f'new2_{ds}.jsonl'); B = L(f'new3_{ds}.jsonl')
    b = {k: mean([x[k] for x in A if x['condition'] == 'clean' and x['method'] == 'fedavg' and x['attack'] == 'hide']) for k in ('rmse_event_honest', 'rmse_all_honest')}
    print('==', ds, ' clean FedAvg: events', round(b['rmse_event_honest'], 2), 'all', round(b['rmse_all_honest'], 2))
    print(' clean run (method weight without attack), RMSE all, % to FedAvg:', {m: round((mean([x['rmse_all_honest'] for x in A if x['condition'] == 'clean' and x['method'] == m]) / b['rmse_all_honest'] - 1) * 100, 1) for m in ('fltrust', 'probe', 'probe2', 'probe2_fl') if any(x['method'] == m and x['condition'] == 'clean' for x in A)})
    for atk, key in (('hide', 'rmse_event_honest'), ('exagg', 'rmse_all_honest'), ('bias', 'rmse_all_honest'), ('alie', 'rmse_all_honest'), ('flip', 'rmse_all_honest')):
        for bad in (0.2, 0.3, 0.4, 0.5, 0.6):
            sel = lambda m, D=A: [x[key] for x in D if x['condition'] == 'atk' and x['attack'] == atk and abs(x['bad'] - bad) < 1e-9 and x['method'] == m]
            row = {m: round((mean(sel(m)) / b[key] - 1) * 100) for m in ('fedavg', 'median', 'krum40', 'fltrust', 'probe', 'probe2', 'probe2_fl') if sel(m)}
            r3 = {m + '*': round((mean(sel(m, B)) / b[key] - 1) * 100) for m in ('probe2', 'probe2_fl') if sel(m, B)}
            if row or r3: print(' ', atk, bad, row, r3 if r3 else '')
    for bad in (0.2, 0.4, 0.6):
        sh = lambda m, D=A: [x['bd_shift'] for x in D if x['condition'] == 'atk' and x['attack'] == 'backdoor' and abs(x['bad'] - bad) < 1e-9 and x['method'] == m and x.get('bd_shift') is not None]
        rw = {m: round(mean(sh(m)), 1) for m in ('fedavg', 'median', 'krum40', 'fltrust', 'probe', 'probe2', 'probe2_fl') if sh(m)}; r3 = {m + '*': round(mean(sh(m, B)), 1) for m in ('probe2', 'probe2_fl') if sh(m, B)}
        if rw: print('  backdoor shift, mph', bad, rw, r3 if r3 else '')
    c = [x['bd_shift'] for x in A if x['condition'] == 'clean' and x.get('bd_shift') is not None]
    if c: print('  backdoor clean model, shift from the trigger:', round(mean(c), 2))
print('* = parameters delta=5, tau=0.5 (tuned on exagg); the others delta=10, tau=1')
