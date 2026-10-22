"""Loading of the results final_*.jsonl / adapt_*.jsonl (and the old new*.jsonl for debugging): harm = max(RMSE_event, RMSE_all) / the same for clean FedAvg of the same network."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
RES = Path("/mnt/projects/phd-fl-gpu/results"); FIG = Path("/mnt/projects/phd-fl-gpu/figures_final"); FIG.mkdir(exist_ok=True)
OUT = Path("/mnt/projects/phd-fl-gpu/results/analysis"); OUT.mkdir(exist_ok=True)


def load(patterns=("final_*.jsonl",), baseline=None):
    fr = []
    for p in patterns:
        for f in sorted(glob.glob(str(RES / p))):
            d = pd.read_json(f, lines=True); d["file"] = Path(f).name; fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    if "dataset" not in d: d["dataset"] = "metr_la"
    d["bad"] = d["bad"].round(2); d["attack"] = np.where(d.condition == "clean", "none", d.attack.fillna("hide"))
    d["method"] = d.method.replace({"krum40": "krum"})
    key = ["dataset", "method", "condition", "seed", "bad", "attack"] + [c for c in ("delta", "tau", "adapt_c", "collude", "norm") if c in d]
    d = d.drop_duplicates(subset=key, keep="first").copy()
    clean = (baseline if baseline is not None else d)
    base = clean[(clean.condition == "clean") & (clean.method == "fedavg")].groupby("dataset")[["rmse_event_honest", "rmse_all_honest"]].mean()
    d["ratio_event"] = d.rmse_event_honest / d.dataset.map(base.rmse_event_honest); d["ratio_all"] = d.rmse_all_honest / d.dataset.map(base.rmse_all_honest)
    d["ratio_worst"] = d[["ratio_event", "ratio_all"]].max(axis=1); d["log_ratio"] = np.log(d.ratio_worst); d["harm_pct"] = (d.ratio_worst - 1) * 100
    if "sig" in d:
        sg = pd.json_normalize(d.sig.where(d.sig.notna(), None).apply(lambda x: x if isinstance(x, dict) else {})); sg.columns = ["sig_" + c for c in sg.columns]; d = pd.concat([d.drop(columns="sig").reset_index(drop=True), sg.reset_index(drop=True)], axis=1)
    d["cluster"] = d.dataset + "_s" + d.seed.astype(str)
    return d


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o): run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj
