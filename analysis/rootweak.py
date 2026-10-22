"""Weak FLTrust root: harm of the methods by root scenario, paired differences with FLTrust (by seed), criterion from results/rootweak_plan.md.
Usage: ../venv/bin/python rootweak.py   -> results/analysis/rootweak_tables.md, rootweak_pairs.csv, figures_final/rootweak_*.png"""
import glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import RES, OUT, FIG
rng = np.random.default_rng(0)
fr = []
for f in sorted(glob.glob(str(RES / "rootweak_*.jsonl"))): fr.append(pd.read_json(f, lines=True))
d = pd.concat(fr, ignore_index=True)
for c, v in (("root_n", 10), ("root_mode", "random"), ("root_poison", 0), ("noniid", 0)): d[c] = d[c].fillna(v)
d["bad"] = d.bad.round(2); d["attack"] = np.where(d.condition == "clean", "none", d.attack)
def sc(r):
    if r.noniid: return "noniid"
    if r.root_poison: return f"poison{int(r.root_poison)}"
    if r.root_mode == "cluster": return "bias10"
    return "ctrl" if r.root_n == 10 else f"small{int(r.root_n)}"
d["scenario"] = d.apply(sc, axis=1)
d = d.drop_duplicates(subset=["dataset", "scenario", "method", "condition", "seed", "bad", "attack"], keep="first")
base = d[(d.condition == "clean") & (d.method == "fedavg")].groupby(["dataset", "scenario"])[["rmse_event_honest", "rmse_all_honest"]].mean()
key = list(zip(d.dataset, d.scenario))
d["ratio"] = np.maximum(d.rmse_event_honest.values / base.rmse_event_honest.reindex(key).values, d.rmse_all_honest.values / base.rmse_all_honest.reindex(key).values)
d["harm"] = (d.ratio - 1) * 100
SCEN = ["ctrl", "small2", "small5", "bias10", "poison3", "noniid"]; ATT = ["hide", "exagg", "bias", "flip"]; METH = ["fedavg", "fltrust", "probe2", "probe2_fl"]
a = d[d.condition == "atk"]
med = a.groupby(["dataset", "scenario", "attack", "bad", "method"]).harm.median().unstack("method")
cl = d[d.condition == "clean"].groupby(["dataset", "scenario", "method"]).harm.mean().unstack("method")
def boot(x, n=4000):
    x = np.asarray(x); m = [np.median(rng.choice(x, len(x))) for _ in range(n)]; return np.percentile(m, [2.5, 97.5])
rows = []
p = a.pivot_table(index=["dataset", "scenario", "attack", "bad", "seed"], columns="method", values="harm")
for (ds, s, at, b), g in p.groupby(level=[0, 1, 2, 3]):
    for m in ("probe2", "probe2_fl"):
        diff = (g[m] - g["fltrust"]).dropna().values
        if len(diff) < 3: continue
        lo, hi = boot(diff); rows.append(dict(dataset=ds, scenario=s, attack=at, bad=b, method=m, med_diff_pp=float(np.median(diff)), lo=lo, hi=hi, n=len(diff), fltrust_med=g["fltrust"].median(), method_med=g[m].median(), better=bool(hi < 0)))
pr = pd.DataFrame(rows); pr.to_csv(OUT / "rootweak_pairs.csv", index=False)
L = ["# Weak FLTrust root: summary (median harm, %, 5 seeds; harm = max(RMSE_event, RMSE_all)/clean FedAvg of the same dataset and scenario - 1)", ""]
for ds in sorted(d.dataset.unique()):
    L += [f"## {ds}", ""]
    for b in (0.4, 0.6):
        L += [f"### attacker share {b}", "", "| scenario | attack | FedAvg | FLTrust | Probe2 | Probe2+FL | Δ Probe2−FLT [95 % CI] | Δ Probe2+FL−FLT [95 % CI] |", "|---|---|---|---|---|---|---|---|"]
        for s in SCEN:
            for at in ATT:
                try: r = med.loc[(ds, s, at, b)]
                except KeyError: continue
                q = pr[(pr.dataset == ds) & (pr.scenario == s) & (pr.attack == at) & (pr.bad == b)].set_index("method")
                f = lambda m: (f"{q.loc[m].med_diff_pp:+.1f} [{q.loc[m].lo:+.1f}; {q.loc[m].hi:+.1f}]" + (" ✓" if q.loc[m].better else "")) if m in q.index else "—"
                L.append(f"| {s} | {at} | {r.get('fedavg', np.nan):.1f} | {r.get('fltrust', np.nan):.1f} | {r.get('probe2', np.nan):.1f} | {r.get('probe2_fl', np.nan):.1f} | {f('probe2')} | {f('probe2_fl')} |")
        L.append("")
    L += ["### clean runs (loss relative to clean FedAvg of the same scenario, %, mean over seeds)", "", "| scenario | FLTrust | Probe2 | Probe2+FL |", "|---|---|---|---|"]
    for s in SCEN:
        try: r = cl.loc[(ds, s)]
        except KeyError: continue
        L.append(f"| {s} | {r.get('fltrust', np.nan):+.1f} | {r.get('probe2', np.nan):+.1f} | {r.get('probe2_fl', np.nan):+.1f} |")
    L.append("")
# criterion: for each (scenario, attack, share) the method is better than FLTrust with a significant paired difference in at least 2 of 3 networks
crit = pr.groupby(["scenario", "attack", "bad", "method"]).apply(lambda g: pd.Series(dict(nets_better=int(g.better.sum()), nets=g.dataset.nunique()))).reset_index()
clo = d[d.condition == "clean"].groupby(["dataset", "scenario", "method"]).harm.mean().abs().unstack("method")
L += ["## Criterion (>=2 of 3 networks, CI of the difference < 0)", "", crit[crit.nets_better >= 1].to_markdown(index=False) if len(crit) else "no data", ""]
(OUT / "rootweak_tables.md").write_text("\n".join(L), encoding="utf-8")
# figure: harm by root scenario (hide, share 0.4, all networks)
for at in ATT:
    fig, axs = plt.subplots(1, d.dataset.nunique(), figsize=(4.2 * d.dataset.nunique(), 3.4), squeeze=False)
    for ax, ds in zip(axs[0], sorted(d.dataset.unique())):
        sub = a[(a.dataset == ds) & (a.attack == at) & (a.bad == 0.4)]
        for i, m in enumerate(METH[1:]):
            vals = [sub[(sub.scenario == s) & (sub.method == m)].harm.median() for s in SCEN]
            ax.bar(np.arange(len(SCEN)) + (i - 1) * 0.27, vals, 0.27, label={"fltrust": "FLTrust", "probe2": "Probe2", "probe2_fl": "Probe2+FLTrust"}[m])
        ax.set_xticks(range(len(SCEN))); ax.set_xticklabels(SCEN, rotation=40, ha="right", fontsize=8); ax.set_title(f"{ds}, {at}, 40 %"); ax.set_ylabel("harm, %"); ax.set_yscale("symlog", linthresh=5)
    axs[0][0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / f"rootweak_{at}.png", dpi=140); plt.close(fig)
print("done:", OUT / "rootweak_tables.md")
