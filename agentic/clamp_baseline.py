"""Trivial baselines for the LLM experiment (reviewer response): persistence (repeat the last observation) and persistence-clamp (mean/median of the answers clipped to +-C mph from the last observation). No probes, no root, an absolute rule. Usage: OUT_DIR=out ../venv/bin/python clamp_baseline.py -> ../results/analysis/clamp_<tag>.json"""
import json, os, sys, numpy as np
from lib import *
TAG = os.environ.get("TAG", ""); ATT = os.environ.get("ATTACKS", "hide,exagg,hide_aware").split(","); NDRAW = 200
items = [w["id"] for w in WIN]; kind = np.array([w["kind"] for w in WIN]); truth = np.array([w["truth"] for w in WIN]); EV, NM = np.where(kind == "event")[0], np.where(kind != "event")[0]
last = np.array([w["hist"][-1] for w in WIN])
H = load("honest"); FH = forecasts(H, items)
CS = [("clampmean_1sig", 1.0), ("clampmean_0.5sig", 0.5)]
METH = ["mean", "median", "persist"] + [c[0] for c in CS] + [c[0].replace("mean", "median") for c in CS]
def one(m, F, j):
    ok = ~np.isnan(F).any(1); F = F[ok]
    if m == "persist": return np.full(3, last[j])
    if m in ("mean", "median"): return F.mean(0) if m == "mean" else np.median(F, 0)
    c = float(dict(CS)[m.replace("median", "mean")]) * SIG; Fc = np.clip(F, last[j] - c, last[j] + c)
    return Fc.mean(0) if "mean" in m else np.median(Fc, 0)
def run(attack, share):
    A = load(attack); FA = forecasts(A, items); out = {m: np.zeros(len(WIN)) for m in METH}
    for j, w in enumerate(WIN):
        acc = {m: [] for m in METH}
        for r in range(NDRAW):
            h, a = comp(attack, share, w["id"], r); F = np.vstack([FH[h, j], FA[a, j]]) if len(a) else FH[h, j]
            for m in METH: acc[m].append(one(m, F, j))
        for m in METH: out[m][j] = np.nanmean(((np.array(acc[m]) - truth[j]) ** 2).mean(1))
    return out
def harm(e, b): return max(np.sqrt(np.mean(e[EV]) / np.mean(b[EV])), np.sqrt(np.mean(e[NM]) / np.mean(b[NM]))) * 100 - 100
base = run(ATT[0], 0.0)["mean"]; S = {"clean_rmse": {"event": float(np.sqrt(base[EV].mean())), "normal": float(np.sqrt(base[NM].mean()))}}
for at in ATT:
    if load(at) is None: continue
    for sh in (0.0, 0.4, 0.6):
        o = run(at, sh); S[f"{at}_{sh}"] = {m: float(harm(o[m], base)) for m in METH}; S[f"{at}_{sh}"]["abs_rmse_mean_event"] = float(np.sqrt(o["mean"][EV].mean()))
        print(at, sh, {m: round(v, 1) for m, v in S[f"{at}_{sh}"].items()})
json.dump(S, open(f"../results/analysis/clamp{TAG}.json", "w"), indent=1)
