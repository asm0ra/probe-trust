"""Numbers of the ablation and additional methods for the paper: probe2_rel, fedmid_like, flame_like, cost of thresholds, sensitivity. Output paper_numbers_abl.json."""
import json, glob, numpy as np, pandas as pd
R = "../results/"; DS = ("metr_la", "pems_bay", "pemsd7m")
f = pd.concat([pd.read_json(R + f"final_{s}.jsonl", lines=True) for s in DS], ignore_index=True)
base = f[(f.condition == "clean") & (f.method == "fedavg")].groupby("dataset")[["rmse_event_honest", "rmse_all_honest"]].mean()
def H(d): return (np.maximum(d.rmse_event_honest / d.dataset.map(base.rmse_event_honest), d.rmse_all_honest / d.dataset.map(base.rmse_all_honest)) - 1) * 100
def load(pref):
    d = pd.concat([pd.read_json(x, lines=True) for x in sorted(glob.glob(R + f"{pref}_*.jsonl"))], ignore_index=True)
    d = d.drop_duplicates(subset=["dataset", "method", "condition", "seed", "bad", "attack", "delta", "tau"]); d["harm"] = H(d); d["bad"] = d.bad.round(2); return d
ab, fl, sn, sc = load("abl"), load("flame"), load("sens"), load("sensclean")
ref = f[f.method.isin(["probe2", "krum", "fltrust", "median", "fedavg"])].copy(); ref["harm"] = H(ref); ref["bad"] = ref.bad.round(2); ref["delta"] = 0.5; ref["tau"] = 0.1
cols = ["dataset", "method", "condition", "seed", "bad", "attack", "harm"]
al = pd.concat([ab[cols], fl[cols], ref[cols]], ignore_index=True)
a = al[(al.condition == "atk") & al.attack.isin(["hide", "exagg", "bias", "flip"])]
per = a.groupby(["method", "attack", "bad", "dataset"]).harm.median().reset_index()
N = {"attack": {}}
for (m, at, b), g in per.groupby(["method", "attack", "bad"]): N["attack"].setdefault(m, {}).setdefault(at, {})[str(b)] = {"median": round(float(g.harm.median()), 1), "max": round(float(g.harm.max()), 1)}
N["clean_overhead"] = al[al.condition == "clean"].groupby("method").harm.mean().round(2).to_dict()
N["sensclean"] = {f"{d}_{t}": round(float(x.harm.mean()), 2) for (d, t), x in sc.groupby(["delta", "tau"])}
s2 = pd.concat([sn[["dataset", "attack", "bad", "delta", "tau", "seed", "harm"]], ref[(ref.method == "probe2") & (ref.condition == "atk") & ref.attack.isin(["hide", "exagg", "bias"]) & ref.bad.isin([0.4, 0.6])][["dataset", "attack", "bad", "delta", "tau", "seed", "harm"]]], ignore_index=True)
tt = s2.groupby(["delta", "tau", "bad", "attack", "dataset"]).harm.median().reset_index().groupby(["delta", "tau", "bad"]).harm.agg(["median", "max"]).round(1)
N["sens"] = {f"{d}_{t}_{b}": {"median": r["median"], "max": r["max"]} for (d, t, b), r in tt.iterrows()}
json.dump(N, open("paper_numbers_abl.json", "w"), ensure_ascii=False, indent=1); 
for m in ("probe2", "probe2_rel", "fedmid_like", "flame_like", "krum"):
    for at in ("hide", "exagg", "bias", "flip"):
        r = N["attack"].get(m, {}).get(at)
        if r: print(f"{m:12} {at:6}", "  ".join(f"{int(float(b)*100)}%: {v['median']:+7.1f} ({v['max']:+7.1f})" for b, v in sorted(r.items())))
print("clean:", N["clean_overhead"]); print("cost of thresholds:", N["sensclean"])
