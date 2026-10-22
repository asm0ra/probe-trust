"""Numbers and figures of the paper (English captions) from results/final_*.jsonl, rootweak_*.jsonl and agentic. Usage: ../venv/bin/python -W ignore paper_data.py -> paper_numbers.json, figs/*.png"""
import json, glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True); R = "../results/"; DS = ["metr_la", "pems_bay", "pemsd7m"]; DN = {"metr_la": "METR-LA", "pems_bay": "PEMS-BAY", "pemsd7m": "PeMS-D7(M)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = {"fedavg": "#6b7280", "median": "#2563a8", "trimmed": "#d9822b", "krum": "#2f8f5b", "bulyan": "#0f766e", "foolsgold": "#9d174d", "normbound": "#7c3aed", "fltrust": "#b45309", "probe": "#e8a09a", "probe2": "#c0392b", "probe2_fl": "#6d4aa8"}
LAB = {"fedavg": "FedAvg", "median": "Median", "trimmed": "Trimmed Mean", "krum": "Krum", "bulyan": "Bulyan", "foolsgold": "FoolsGold", "normbound": "Norm-bounding", "fltrust": "FLTrust", "probe": "Probe (one-sided)", "probe2": "Probe-Trust (two-sided)", "probe2_fl": "Probe-Trust + FLTrust"}
d = pd.concat([pd.read_json(R + f"final_{s}.jsonl", lines=True) for s in DS], ignore_index=True)
base = d[(d.condition == "clean") & (d.method == "fedavg")].groupby("dataset")[["rmse_event_honest", "rmse_all_honest"]].mean()
d["harm"] = (np.maximum(d.rmse_event_honest / d.dataset.map(base.rmse_event_honest), d.rmse_all_honest / d.dataset.map(base.rmse_all_honest)) - 1) * 100
d["bad"] = d.bad.round(2); a = d[d.condition == "atk"]
med = a.groupby(["dataset", "method", "attack", "bad"]).harm.median().reset_index()
N = {}
for share in (0.4, 0.6):
    t = med[med.bad == share].groupby(["method", "attack"]).harm.median().unstack("attack"); N[f"table_{share}"] = t.round(1).to_dict("index")
clean = d[d.condition == "clean"].groupby(["method"]).harm.mean().round(1); N["clean_overhead"] = clean.to_dict(); N["final_runs"] = int(len(d))
bd = a[a.attack == "backdoor"].groupby(["method", "bad"]).bd_shift.median().unstack("bad").round(1); N["backdoor_shift_mph"] = bd.to_dict("index")
au = a[(a.attack == "hide")].groupby(["method", "bad"]).auc_attackers.mean().unstack("bad").round(2); N["auc_hide"] = au.to_dict("index")
# Fig 2: harm vs share, hide, 3 datasets
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
for ax, s in zip(axs, DS):
    for m in ["fedavg", "median", "krum", "fltrust", "probe2"]:
        y = [med[(med.dataset == s) & (med.method == m) & (med.attack == "hide") & (med.bad == b)].harm.values[0] for b in (0.2, 0.4, 0.6)]
        ax.plot([20, 40, 60], np.clip(y, -5, 300), marker="o", ms=3, lw=2.2 if m == "probe2" else 1.2, color=COL[m], label=LAB[m])
    ax.set_title(DN[s], fontsize=8); ax.set_xlabel("attacker share, %"); ax.set_xticks([20, 40, 60]); ax.grid(alpha=.25)
axs[0].set_ylabel("harm in events, % (clipped at 300)"); axs[0].legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig("figs/fig_hide_share.png", dpi=250); plt.close(fig)
# Fig 3: one-sided vs two-sided across attacks at 40 %
atks = ["hide", "exagg", "bias", "flip", "alie"]; ms = ["fedavg", "fltrust", "probe", "probe2", "probe2_fl"]
fig, ax = plt.subplots(figsize=(7.2, 2.6)); w = 0.16
for j, m in enumerate(ms):
    y = [med[(med.method == m) & (med.attack == at) & (med.bad == 0.4)].harm.median() for at in atks]
    ax.bar(np.arange(len(atks)) + (j - 2) * w, np.clip(y, 0, 400), w, color=COL[m], label=LAB[m])
    for i, v in enumerate(y):
        if v > 400: ax.text(i + (j - 2) * w, 405, f"{v:.0f}", ha="center", fontsize=5.5, rotation=90, va="bottom")
ax.set_xticks(range(len(atks))); ax.set_xticklabels(["hide congestion", "false congestion", "bias +8 mph", "update flip", "ALIE"]); ax.set_ylabel("harm in events, %\n(median over 3 datasets, clipped at 400)"); ax.set_ylim(0, 520); ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper right"); ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig("figs/fig_attacks40.png", dpi=250); plt.close(fig)
# Fig 4: rootweak, paired differences Probe-Trust - FLTrust by root scenario (hide, bias; 40 %): intervals of EACH dataset separately (without averaging the bounds)
pr = pd.read_csv(R + "analysis/rootweak_pairs.csv"); N["rootweak_pairs_rows"] = int(len(pr))
SC = ["ctrl", "small2", "small5", "bias10", "poison3", "noniid"]; SL = {"ctrl": "clean root (10)", "small2": "root of 2", "small5": "root of 5", "bias10": "biased root", "poison3": "poisoned root (3/10)", "noniid": "non-IID clients"}
MK = {"metr_la": "o", "pems_bay": "s", "pemsd7m": "^"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, at in zip(axs, ["hide", "bias"]):
    for m, c, off0 in [("probe2", COL["probe2"], -0.16), ("probe2_fl", COL["probe2_fl"], 0.16)]:
        for k, ds in enumerate(DS):
            ys = []; lo = []; hi = []
            for s_ in SC:
                x = pr[(pr.scenario == s_) & (pr.attack == at) & (pr.bad.round(2) == 0.4) & (pr.method == m) & (pr.dataset == ds)]
                ys.append(x.med_diff_pp.values[0] if len(x) else np.nan); lo.append(x.lo.values[0] if len(x) else np.nan); hi.append(x.hi.values[0] if len(x) else np.nan)
            ys, lo, hi = map(np.array, (ys, lo, hi)); ax.errorbar(np.arange(len(SC)) + off0 + (k - 1) * 0.055, ys, yerr=[ys - lo, hi - ys], fmt=MK[ds], ms=3, color=c, capsize=1.2, lw=0.9, label=f"{LAB[m]}, {DN[ds]}" if at == "hide" and s_ == SC[-1] else None)
    ax.axhline(0, color="#9ca3af", ls=":"); ax.set_xticks(range(len(SC))); ax.set_xticklabels([SL[s_] for s_ in SC], rotation=30, ha="right", fontsize=6.5); ax.set_title(f"attack: {at}, 40 % attackers", fontsize=8); ax.grid(axis="y", alpha=.25)
axs[0].set_ylabel("harm difference vs FLTrust, p.p.\n(<0: better than FLTrust)"); h_, l_ = axs[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.legend(h_, l_, fontsize=5.8, frameon=False, ncol=3, loc="lower center"); fig.savefig("figs/fig_rootweak.png", dpi=250); plt.close(fig)
json.dump(N, open("paper_numbers.json", "w"), ensure_ascii=False, indent=1, default=float); print("ok"); print(json.dumps(N["table_0.4"], indent=0)[:1500])

# ---- rootweak: median harm by root scenario (as in analysis/rootweak.py) ----
fr = [pd.read_json(f, lines=True) for f in sorted(glob.glob(R + "rootweak_*.jsonl"))]; rw = pd.concat(fr, ignore_index=True)
for c, v in (("root_n", 10), ("root_mode", "random"), ("root_poison", 0), ("noniid", 0)): rw[c] = rw[c].fillna(v)
rw["bad"] = rw.bad.round(2); rw["attack"] = np.where(rw.condition == "clean", "none", rw.attack)
def _sc(r):
    if r.noniid: return "noniid"
    if r.root_poison: return f"poison{int(r.root_poison)}"
    if r.root_mode == "cluster": return "bias10"
    return "ctrl" if r.root_n == 10 else f"small{int(r.root_n)}"
rw["scenario"] = rw.apply(_sc, axis=1)
rw = rw.drop_duplicates(subset=["dataset", "scenario", "method", "condition", "seed", "bad", "attack"], keep="first")
_b = rw[(rw.condition == "clean") & (rw.method == "fedavg")].groupby(["dataset", "scenario"])[["rmse_event_honest", "rmse_all_honest"]].mean(); _k = list(zip(rw.dataset, rw.scenario))
rw["harm"] = (np.maximum(rw.rmse_event_honest.values / _b.rmse_event_honest.reindex(_k).values, rw.rmse_all_honest.values / _b.rmse_all_honest.reindex(_k).values) - 1) * 100
ra = rw[rw.condition == "atk"].groupby(["dataset", "scenario", "attack", "bad", "method"]).harm.median().reset_index()
N["rootweak_harm"] = {f"{sh}": ra[ra.bad == sh].groupby(["scenario", "attack", "method"]).harm.median().unstack("method").round(1).reset_index().to_dict("records") for sh in (0.4, 0.6)}
N["rootweak_datasets"] = sorted(rw.dataset.unique().tolist()); N["rootweak_runs"] = int(len(rw))
json.dump(N, open("paper_numbers.json", "w"), ensure_ascii=False, indent=1, default=float); print("rootweak ok", N["rootweak_runs"])

# ---- by dataset separately (for hypothesis checks) ----
pd_ = {}
for m in ("probe2", "probe2_fl", "fltrust", "krum", "median", "fedavg"):
    pd_[m] = {at: {DN[s]: round(float(med[(med.dataset == s) & (med.method == m) & (med.attack == at) & (med.bad == 0.6)].harm.values[0]), 1) for s in DS} for at in ("hide", "exagg", "bias", "flip", "alie")}
N["per_dataset_0.6"] = pd_
pd4 = {}
for m in ("probe2", "probe2_fl", "fltrust", "krum", "median", "fedavg", "probe"):
    pd4[m] = {at: {DN[s]: round(float(med[(med.dataset == s) & (med.method == m) & (med.attack == at) & (med.bad == 0.4)].harm.values[0]), 1) for s in DS} for at in ("hide", "exagg", "bias", "flip", "alie")}
N["per_dataset_0.4"] = pd4
json.dump(N, open("paper_numbers.json", "w"), ensure_ascii=False, indent=1, default=float); print("per-dataset ok")

# ---- adaptive attackers (adapt_*.jsonl), harm by dataset -> median and maximum over the 3 datasets ----
ad = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(R + "adapt_*.jsonl"))], ignore_index=True)
ad["harm"] = (np.maximum(ad.rmse_event_honest / ad.dataset.map(base.rmse_event_honest), ad.rmse_all_honest / ad.dataset.map(base.rmse_all_honest)) - 1) * 100; ad["bad"] = ad.bad.round(2)
ad = ad[ad.condition == "atk"].copy(); ad["name"] = np.where((ad.attack == "adapt_all") & (ad.collude == 1), "adapt_all_collude", ad.attack)
adm = ad.groupby(["name", "bad", "method", "dataset"]).harm.median().reset_index()
N["adaptive"] = {n: {str(b): {m: {"median": round(float(x.harm.median()), 1), "max": round(float(x.harm.max()), 1)} for m, x in g2.groupby("method")} for b, g2 in g.groupby("bad")} for n, g in adm.groupby("name")}
json.dump(N, open("paper_numbers.json", "w"), ensure_ascii=False, indent=1, default=float); print("adaptive ok")
