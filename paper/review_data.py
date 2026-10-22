"""Numbers of the review response: sweep of attack strength (P1), misspecified probe ranges (P2), Min-Max (P3), 20 seeds at 60 % (P4). Output paper_numbers_rev.json. Harm as in ablation_data.py (maximum relative RMSE growth in events and on all windows against clean FedAvg)."""
import json, glob, numpy as np, pandas as pd
R = "../results/"; DS = ("metr_la", "pems_bay", "pemsd7m"); rng = np.random.default_rng(11)
f = pd.concat([pd.read_json(R + f"final_{s}.jsonl", lines=True) for s in DS], ignore_index=True)
base = f[(f.condition == "clean") & (f.method == "fedavg")].groupby("dataset")[["rmse_event_honest", "rmse_all_honest"]].mean()
def H(d): return (np.maximum(d.rmse_event_honest / d.dataset.map(base.rmse_event_honest), d.rmse_all_honest / d.dataset.map(base.rmse_all_honest)) - 1) * 100
def load(pat):
    fs = sorted(glob.glob(R + pat))
    if not fs: return pd.DataFrame()
    d = pd.concat([pd.read_json(x, lines=True) for x in fs], ignore_index=True); d["harm"] = H(d); d["bad"] = d.bad.round(2); return d
def agg(d, keys):
    per = d.groupby(keys + ["dataset"]).harm.median().reset_index(); g = per.groupby(keys).harm.agg(["median", "max"]).round(1)
    return {"|".join(map(str, k)) if isinstance(k, tuple) else str(k): {"median": r["median"], "max": r["max"]} for k, r in g.iterrows()}
N = {}
p1 = load("rev_P1_*.jsonl")
if len(p1): N["P1"] = agg(p1[p1.condition == "atk"], ["attack", "hide_mag", "bad", "method"])
p3 = load("rev_P3_*.jsonl")
if len(p3): N["P3"] = agg(p3[p3.condition == "atk"], ["bad", "method"])
p2 = []
for fn in sorted(glob.glob(R + "rev_P2_*.jsonl")):
    d = pd.read_json(fn, lines=True); d["harm"] = H(d); d["bad"] = d.bad.round(2); d["cfg"] = "_".join(fn.split("rev_P2_")[1].replace(".jsonl", "").split("_")[-4:]); p2.append(d)
if p2: N["P2"] = agg(pd.concat(p2, ignore_index=True), ["cfg", "attack", "bad"])
p4 = load("rev_P4_*.jsonl")
if len(p4):
    ab = load("abl_*.jsonl"); ref = f[f.method.isin(["fedavg", "median", "fltrust", "probe2"])].copy(); ref["harm"] = H(ref); ref["bad"] = ref.bad.round(2)
    allr = pd.concat([x[(x.condition == "atk") & (x.bad == 0.6) & x.attack.isin(["hide", "exagg", "bias"]) & x.method.isin(["fedavg", "median", "fltrust", "probe2", "probe2_rel"])][["dataset", "method", "attack", "seed", "harm"]] for x in (p4, ab, ref)], ignore_index=True).drop_duplicates(["dataset", "method", "attack", "seed"])
    N["P4_n_seeds"] = int(allr.seed.nunique()); N["P4"] = {}; N["P4_diff"] = {}
    for at in ("hide", "exagg", "bias"):
        a = allr[allr.attack == at]; per = a.groupby(["method", "dataset"]).harm.median().reset_index()
        N["P4"][at] = {m: {"median": round(float(g.harm.median()), 1), "max": round(float(g.harm.max()), 1)} for m, g in per.groupby("method")}
        piv = a.pivot_table(index=["dataset", "seed"], columns="method", values="harm")
        for other in ("fltrust", "probe2_rel", "median"):
            dd = (piv["probe2"] - piv[other]).dropna(); v = [np.median(rng.choice(dd.values, len(dd))) for _ in range(4000)]
            N["P4_diff"][f"{at}|probe2-{other}"] = {"median": round(float(dd.median()), 1), "ci": [round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)], "n": int(len(dd))}
json.dump(N, open("paper_numbers_rev.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(N, ensure_ascii=False)[:3000])

# ---- part 2: an adversary that distinguishes probes (Q1), attacks not bound by the rule (Q2)
N2 = {}
q1 = load("rev2_Q1_*.jsonl")
if len(q1): q1["lam"] = q1.lam; N2["Q1"] = agg(q1[q1.condition == "atk"], ["attack", "lam", "bad", "method"])
q2 = load("rev2_Q2_*.jsonl")
if len(q2): N2["Q2"] = agg(q2[q2.condition == "atk"], ["attack", "bad", "method"])
if N2:
    N.update(N2); json.dump(N, open("paper_numbers_rev.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(N2, ensure_ascii=False)[:2500])

# ---- part 3: sweep of the probe share in the attacker's training (Q3), strong non-IID (N), another architecture (A). For N and A the base is clean FedAvg of the same configuration (rev3_*, condition=clean).
N3 = {}
def load_own(pat):
    fs = sorted(glob.glob(R + pat))
    if not fs: return pd.DataFrame()
    d = pd.concat([pd.read_json(x, lines=True) for x in fs], ignore_index=True); d["bad"] = d.bad.round(2)
    b = d[(d.condition == "clean") & (d.method == "fedavg")].groupby("dataset")[["rmse_event_honest", "rmse_all_honest"]].mean()
    d["harm"] = (np.maximum(d.rmse_event_honest / d.dataset.map(b.rmse_event_honest), d.rmse_all_honest / d.dataset.map(b.rmse_all_honest)) - 1) * 100
    return d
def clean_cost(d):
    c = d[d.condition == "clean"].groupby(["dataset", "method"]).harm.median().reset_index(); return {m: {"median": round(float(g.harm.median()), 2), "max": round(float(g.harm.max()), 2)} for m, g in c.groupby("method")}
q3 = load("rev3_Q3_*.jsonl")
if len(q3): N3["Q3"] = agg(q3[q3.condition == "atk"], ["attack", "aware_k", "bad", "method"])
nn = load_own("rev3_N_*.jsonl")
if len(nn): N3["N"] = agg(nn[nn.condition == "atk"], ["attack", "bad", "method"]); N3["N_clean"] = clean_cost(nn)
for ar in ("tf", "mlp"):
    aa = load_own(f"rev3_A_{ar}_*.jsonl")
    if len(aa): N3["A_" + ar] = agg(aa[aa.condition == "atk"], ["attack", "bad", "method"]); N3["A_" + ar + "_clean"] = clean_cost(aa)
if N3:
    N.update(N3); json.dump(N, open("paper_numbers_rev.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(N3, ensure_ascii=False)[:1500])

# ---- part 4: Zeno++ and FLAME from the papers (rev4)
r4 = load("rev4_*.jsonl")
if len(r4):
    N["R4"] = agg(r4[r4.condition == "atk"], ["attack", "bad", "method"]); json.dump(N, open("paper_numbers_rev.json", "w"), ensure_ascii=False, indent=1)

# ---- part 5: probes from the public history of the same network (same), graph model (G), second domain: solar generation (S)
for key, pat in (("SAME", "same_[mpe]*.jsonl"), ("G", "rev5_G_*.jsonl"), ("S", "rev6_S_*.jsonl")):
    dd = load_own(pat)
    if len(dd):
        a = dd[dd.condition == "atk"]; N[key] = agg(a, ["attack", "bad", "method"]); N[key + "_clean"] = clean_cost(dd)
        N[key + "_n"] = int(len(dd))
json.dump(N, open("paper_numbers_rev.json", "w"), ensure_ascii=False, indent=1)

# solar generation: sensitivity to delta for hide (post hoc; the same dataset, so not an independent check)
_b = pd.read_json(R + "rev6_S_solar.jsonl", lines=True); _c = _b[(_b.condition == "clean") & (_b.method == "fedavg")]; _be, _ba = _c.rmse_event_honest.mean(), _c.rmse_all_honest.mean()
SD = {}
for _dl in ("0.25", "0.1"):
    _d = pd.read_json(R + f"rev6_Sd{_dl}.jsonl", lines=True); _d["h"] = (np.maximum(_d.rmse_event_honest / _be, _d.rmse_all_honest / _ba) - 1) * 100
    SD[_dl] = {f"{m}|{round(b,1)}": round(float(g.median()), 1) for (m, b), g in _d[_d.condition == "atk"].groupby(["method", "bad"]).h}
    if (_d.condition == "clean").any(): SD[_dl]["clean_probe2"] = round(float(_d[_d.condition == "clean"].h.median()), 2)
N["S_delta"] = SD; json.dump(N, open("paper_numbers_rev.json", "w"), ensure_ascii=False, indent=1)
N["S_auc"] = {f"{a}|{round(b,1)}": round(float(g.auc_attackers.mean()), 2) for (a, b), g in _b[(_b.condition == "atk") & (_b.method == "probe2")].groupby(["attack", "bad"])}
json.dump(N, open("paper_numbers_rev.json", "w"), ensure_ascii=False, indent=1)
