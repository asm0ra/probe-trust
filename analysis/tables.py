"""Main tables and figures of the paper from final_*.jsonl and adapt_*.jsonl: harm by method and attack, transfer of thresholds between networks, adaptive attackers, the clean case.
Usage: python tables.py [--tag final]. Output: results/analysis/<tag>_tables.md, figures_final/<tag>_*.png"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load, OUT, FIG, RES
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="final"); a = ap.parse_args()
d = load(("final_*.jsonl",)); atk = d[d.condition == "atk"]; MD = []
ORD = [m for m in ["fedavg", "median", "trimmed", "krum", "normbound", "bulyan", "foolsgold", "fltrust", "probe", "probe2", "probe2_fl"] if m in set(atk.method)]; AT = [x for x in ["hide", "exagg", "bias", "backdoor", "alie", "flip"] if x in set(atk.attack)]
def md(df, title, fmt="{:.0f}"): MD.append(f"\n### {title}\n\n" + df.to_markdown(floatfmt=".0f" if fmt == "{:.0f}" else ".1f") + "\n")
for b in (0.2, 0.4, 0.6):
    t = atk[atk.bad == b].pivot_table(index="method", columns="attack", values="harm_pct", aggfunc="median").reindex(index=ORD, columns=AT); t["worst"] = t.max(axis=1); md(t, f"Median harm, % over clean (share {b}; 3 networks × 5 seeds)")
cl = d[d.condition == "clean"].assign(h=lambda x: x.harm_pct).groupby("method").h.agg(["median", "max"]).reindex(ORD); md(cl, "Clean runs: overhead vs clean FedAvg, % (median, max)", "{:.1f}")
for m in ("probe2", "probe2_fl", "fltrust"):
    t = atk[(atk.method == m) & (atk.bad == 0.4)].pivot_table(index="dataset", columns="attack", values="harm_pct", aggfunc="median").reindex(columns=AT); md(t, f"{m}: harm by network at share 0.4 (transfer of normalised thresholds)")
# adaptive
import glob
ad = [f for f in glob.glob(str(RES / "adapt_*.jsonl"))]
if ad:
    A = load(("adapt_*.jsonl",), baseline=d); A["variant"] = A.attack + np.where(A.get("collude", 0) == 1, "+collude", "")
    t = A.pivot_table(index="method", columns=["variant", "bad"], values="harm_pct", aggfunc="median"); md(t, "Adaptive attackers: median harm, % (shares 0.4 / 0.6; 3 networks × 5 seeds)")
    w = A.groupby(["method", "variant"]).harm_pct.agg(["median", "max"]).round(0); md(w.unstack("variant"), "Adaptive attackers: median and worst run harm, %")
    ref = atk[atk.attack == "hide"].groupby(["method", "bad"]).harm_pct.median().unstack("bad"); md(ref.reindex([m for m in A.method.unique()]), "Reference: non-adaptive 'hide', median harm, %")
open(OUT / f"{a.tag}_tables.md", "w").write("\n".join(MD)); print("\n".join(MD)[:6000])
# figure: heat map of the median harm (share 0.4) and the "harm vs attacker share" curve
t = atk[atk.bad == 0.4].pivot_table(index="method", columns="attack", values="harm_pct", aggfunc="median").reindex(index=ORD, columns=AT)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8)); im = ax[0].imshow(np.log1p(t.clip(lower=0).values), cmap="Reds"); ax[0].set_xticks(range(len(AT))); ax[0].set_xticklabels(AT, fontsize=7, rotation=30); ax[0].set_yticks(range(len(ORD))); ax[0].set_yticklabels(ORD, fontsize=7)
for i in range(len(ORD)):
    for j in range(len(AT)): ax[0].text(j, i, f"{t.values[i, j]:.0f}", ha="center", va="center", fontsize=6)
ax[0].set_title("Median harm, % (share 0.4)", fontsize=9)
for m, c in (("fedavg", "#718096"), ("fltrust", "#c05621"), ("bulyan", "#805ad5"), ("probe2", "#2b6cb0"), ("probe2_fl", "#2f855a")):
    if m in set(atk.method): s = atk[atk.method == m].groupby("bad").harm_pct.median(); ax[1].plot(s.index, s.values.clip(1e-1), marker="o", label=m, color=c, ms=3)
ax[1].set_yscale("symlog", linthresh=10); ax[1].set_xlabel("attacker share"); ax[1].set_ylabel("median harm over all attacks, %"); ax[1].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(FIG / f"{a.tag}_heat_share.png", dpi=150)
