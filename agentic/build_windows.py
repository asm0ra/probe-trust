"""Windows for the experiment "LLM agents + Probe-Trust" (METR-LA, test = the last 20% in time). Plan: ../results/agentic_plan.md. Usage: ../venv/bin/python build_windows.py"""
import json, numpy as np, pandas as pd
df = pd.read_hdf("../data/metr_la.h5"); raw = np.nan_to_num(df.values.astype(np.float32), nan=0.0); T, N = raw.shape
t1, t2 = int(T * 0.7), int(T * 0.8)
tr = raw[:t1][raw[:t1] > 0]; sigma = float(tr.std()); mean = float(tr.mean())
rng = np.random.default_rng(2026); ev, nm = [], []
for t in range(t2 + 12, T - 3):
    pass
idx = np.arange(t2 + 12, T - 3); cand = []
for _ in range(200000):
    t = int(rng.integers(t2 + 12, T - 3)); s = int(rng.integers(0, N)); h = raw[t - 12:t, s]; y = raw[t:t + 3, s]
    if (h > 0).all() and (y > 0).all(): cand.append((t, s, h.tolist(), y.tolist()))
    if len(cand) >= 20000: break
ev = [c for c in cand if np.mean(c[3]) < 40]; nm = [c for c in cand if np.mean(c[3]) >= 55]
rng.shuffle(ev); rng.shuffle(nm)
W = [dict(id=f"e{i}", kind="event", t=c[0], sensor=c[1], hist=[round(x, 1) for x in c[2]], truth=[round(x, 1) for x in c[3]]) for i, c in enumerate(ev[:40])] + \
    [dict(id=f"n{i}", kind="normal", t=c[0], sensor=c[1], hist=[round(x, 1) for x in c[2]], truth=[round(x, 1) for x in c[3]]) for i, c in enumerate(nm[:20])]
json.dump(dict(sigma=sigma, mean=mean, windows=W), open("windows.json", "w"), ensure_ascii=False)
print("sigma", round(sigma, 2), "events", len(ev), "normal", len(nm), "windows", len(W), "event last<45:", sum(w["hist"][-1] < 45 for w in W if w["kind"] == "event"))
