"""Defence-selection agent (NOT an LLM agent): chooses the aggregation rule from server signals (without attacker labels). Offline evaluation on the grid logs: for each context (dataset, attack, share, seed) the harm of every method is known.
Signals are taken from the FedAvg run of the same context over rounds 1-3 (available to the server: probe violations, cosine with the root update, norms, pairwise similarity of updates) and do not depend on the method chosen by the agent.
Policy: regression of ln(harm) of each method on the signals (XGBoost), pick the method with the lowest prediction; compared with the best fixed method, the minimax fixed method and the oracle. Generalisation checks: leave-one-attack-out, leave-one-dataset-out, groups by seed.
Usage: python agent.py [--glob "final_*.jsonl"] [--tag final]"""
import argparse, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xgboost as xgb
from common import load, OUT, FIG
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--glob", default="final_*.jsonl"); ap.add_argument("--tag", default="final"); a = ap.parse_args()
d = load((a.glob,)); SIG = [c for c in d.columns if c.startswith("sig_")]
d["ctx"] = np.where(d.condition == "clean", d.dataset + "|none|0|" + d.seed.astype(str), d.dataset + "|" + d.attack + "|" + d.bad.astype(str) + "|" + d.seed.astype(str))
ACT = [m for m in ["fedavg", "median", "trimmed", "krum", "normbound", "bulyan", "foolsgold", "fltrust", "probe", "probe2", "probe2_fl"] if m in set(d.method)]
Y = d.pivot_table(index="ctx", columns="method", values="log_ratio")[ACT].dropna()
F = d[d.method == "fedavg"].drop_duplicates("ctx").set_index("ctx")[SIG].reindex(Y.index).dropna(); Y = Y.loc[F.index]
meta = pd.DataFrame([c.split("|") for c in Y.index], index=Y.index, columns=["dataset", "attack", "bad", "seed"])
R = {"n_contexts": int(len(Y)), "actions": ACT, "signals": SIG, "n_clean_contexts": int((meta.attack == "none").sum())}

def fit_predict(tr, te, acts):
    P = pd.DataFrame(index=te, columns=acts, dtype=float)
    for m in acts:
        mdl = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, subsample=0.9, tree_method="hist", n_jobs=8, random_state=0).fit(F.loc[tr], Y.loc[tr, m]); P[m] = mdl.predict(F.loc[te])
    return P

def evaluate(split, label, acts, safe=0.0):
    res = []
    for g in sorted(meta[split].unique()):
        te = meta.index[meta[split] == g]; tr = meta.index[meta[split] != g]
        if len(tr) < 30 or len(te) < 3: continue
        fixed_best = Y.loc[tr, acts].mean().idxmin(); fixed_minimax = Y.loc[tr, acts].max().idxmin(); ac = [m for m in acts if safe <= 0 or np.exp(Y.loc[tr, m].quantile(0.9)) - 1 < safe] or [fixed_minimax]
        P = fit_predict(tr, te, ac); pick = P.idxmin(axis=1)
        for c in te:
            res.append(dict(group=g, ctx=c, attack=meta.loc[c, "attack"], dataset=meta.loc[c, "dataset"], agent=Y.loc[c, pick[c]], fixed_best=Y.loc[c, fixed_best], fixed_minimax=Y.loc[c, fixed_minimax], oracle=Y.loc[c, acts].min(), fedavg=Y.loc[c, "fedavg"],
                            fl=Y.loc[c, "fltrust"] if "fltrust" in Y else np.nan, p2fl=Y.loc[c, "probe2_fl"] if "probe2_fl" in Y else np.nan, p2=Y.loc[c, "probe2"] if "probe2" in Y else np.nan, pick=pick[c], fb=fixed_best, fm=fixed_minimax))
    if not res: return None
    r = pd.DataFrame(res); r.to_csv(OUT / f"{a.tag}_agent_{label}.csv", index=False)
    pct = lambda x: float((np.exp(x) - 1).mean() * 100)
    atk_ = r[r.attack != "none"]; cl = r[r.attack == "none"]
    summ = {k: dict(mean_harm_pct=pct(r[k]), worst_pct=float((np.exp(r[k].max()) - 1) * 100), p90_pct=float((np.exp(r[k].quantile(0.9)) - 1) * 100), attacked_mean_pct=pct(atk_[k]), clean_mean_pct=pct(cl[k]) if len(cl) else None)
            for k in ("agent", "fixed_best", "fixed_minimax", "p2fl", "p2", "fl", "fedavg", "oracle")}
    summ["agent_regret_vs_oracle_log"] = float((r.agent - r.oracle).mean()); summ["agent_best_share"] = float((r.agent <= r.oracle + 1e-9).mean()); summ["agent_within_5pct_of_oracle"] = float(((np.exp(r.agent) - np.exp(r.oracle)) <= 0.05).mean())
    summ["pick_counts"] = r.pick.value_counts().to_dict(); summ["fixed_best_chosen"] = r.fb.value_counts().to_dict(); summ["by_group_mean_harm_pct"] = {g: dict(agent=pct(x.agent), fixed_best=pct(x.fixed_best), p2fl=pct(x.p2fl), p2=pct(x.p2), oracle=pct(x.oracle)) for g, x in r.groupby("group")}
    return summ

SCEN = {"all_actions": (ACT, 0.0), "no_clean_root": ([m for m in ACT if m not in ("fltrust", "probe2_fl")], 0.0), "all_actions_safe": (ACT, 25.0), "no_clean_root_safe": ([m for m in ACT if m not in ("fltrust", "probe2_fl")], 25.0)}
for sc, (acts, safe) in SCEN.items():
    R[sc] = {key: evaluate(sp, f"{sc}_{key}", acts, safe) for key, sp in (("leave_one_attack_out", "attack"), ("leave_one_dataset_out", "dataset"), ("seed_groups", "seed"))}
fig, ax = plt.subplots(2, 3, figsize=(9.5, 5.4)); keys = ["agent", "fixed_best", "fixed_minimax", "p2", "p2fl", "fedavg", "oracle"]
for row, sc in enumerate(("all_actions_safe", "no_clean_root_safe")):
    for k_, (nm, key) in enumerate((("unseen attack", "leave_one_attack_out"), ("unseen network", "leave_one_dataset_out"), ("unseen seeds", "seed_groups"))):
        s_ = R[sc][key]
        if s_ is None: continue
        ax[row, k_].bar(keys, [s_[x]["mean_harm_pct"] for x in keys], color=["#2b6cb0", "#718096", "#a0aec0", "#805ad5", "#c05621", "#cbd5e0", "#2f855a"]); ax[row, k_].set_title(f"{sc}: {nm}", fontsize=8); ax[row, k_].tick_params(axis="x", rotation=60, labelsize=7); ax[row, k_].set_ylabel("mean harm, %", fontsize=8); ax[row, k_].set_yscale("symlog", linthresh=5)
fig.tight_layout(); fig.savefig(FIG / f"{a.tag}_agent.png", dpi=150); plt.close(fig)
json.dump(R, open(OUT / f"{a.tag}_agent.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({sc: {key: {x: (R[sc][key][x]["mean_harm_pct"] if R[sc][key] else None) for x in ("agent", "fixed_best", "fixed_minimax", "p2", "p2fl", "fedavg", "oracle")} for key in R[sc]} for sc in SCEN}, ensure_ascii=False, indent=0, default=str)[:3500])
