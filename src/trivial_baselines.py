"""Trivial predictors on the same evaluation windows as the federated models: persistence (repeat the last observation) and historical average (mean by time of day over the training part). Result: RMSE in mph on event windows and all windows, together with the RMSE of clean FedAvg from the logs. Usage: venv/bin/python src/trivial_baselines.py -> results/analysis/trivial_baselines.json"""
import sys, json, glob, numpy as np, torch
sys.path.insert(0, "src")
import bfl
from bfl import Ctx, IN_LEN, OUT_LEN
out = {}
for ds in ("metr_la", "pems_bay", "pemsd7m"):
    ctx = Ctx(torch.device("cpu"), ds); raw = ctx.raw.numpy(); a, b = ctx.data.bounds["train"]; T = raw.shape[0]
    tod = np.arange(T) % 288; ha = np.zeros((288, raw.shape[1]))
    for k in range(288):
        sel = np.arange(a, b)[tod[a:b] == k]; v = raw[sel]; m = v > 0; ha[k] = np.where(m.sum(0) > 0, (v * m).sum(0) / np.maximum(m.sum(0), 1), np.nan)
    ha = np.where(np.isnan(ha), np.nanmean(ha), ha); r = {}
    for name, nodes, starts in (("event", ctx.ev_nodes, ctx.ev_starts), ("all", ctx.all_nodes, ctx.all_starts)):
        se_p = se_h = cnt = 0.0
        for h in range(OUT_LEN):
            idx = starts + IN_LEN + h; y = raw[idx, nodes]; m = y > 0
            p = raw[starts + IN_LEN - 1, nodes]; hh = ha[idx % 288, nodes]
            se_p += (((p - y) ** 2) * m).sum(); se_h += (((hh - y) ** 2) * m).sum(); cnt += m.sum()
        r[name] = {"persistence": float((se_p / cnt) ** 0.5), "hist_avg": float((se_h / cnt) ** 0.5)}
    fa = [json.loads(l) for f in glob.glob(f"results/final_{ds}.jsonl") for l in open(f)]
    cl = [x for x in fa if x["condition"] == "clean" and x["method"] == "fedavg"]
    r["fedavg_clean"] = {"event": float(np.median([x["rmse_event_honest"] for x in cl])), "all": float(np.median([x["rmse_all_honest"] for x in cl])), "n": len(cl)}
    out[ds] = r; print(ds, r)
json.dump(out, open("results/analysis/trivial_baselines.json", "w"), indent=1)
