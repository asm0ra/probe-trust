"""Causal and explanatory analysis of the simulator results. The design is a randomised experiment: method, attack, share and network are assigned by us, so the effect of a method is identified by a paired comparison within ONE context (same dataset, attack, share, seed -> same attackers and same initialisation).
Usage: python causal.py [--glob "final_*.jsonl"] [--tag final]
Output: results/analysis/<tag>_*.csv|json, figures_final/<tag>_*.png"""
import argparse, json, warnings
import numpy as np, pandas as pd, scipy.stats as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from common import load, holm, OUT, FIG
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--glob", default="final_*.jsonl"); ap.add_argument("--tag", default="final"); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
d = load((a.glob,)); atk = d[d.condition == "atk"].copy(); rng = np.random.default_rng(0)
CTX = ["dataset", "attack", "bad", "seed"]; R = {"n_runs": int(len(d)), "n_attacked": int(len(atk)), "methods": sorted(atk.method.unique()), "attacks": sorted(atk.attack.unique())}
W = atk.pivot_table(index=CTX, columns="method", values="log_ratio")                        # paired contexts

def paired(m1, m0):
    """Effect of replacing m0 -> m1 within one context: difference of the logarithms of harm, averaged over contexts; clusters = dataset x seed (cluster bootstrap)."""
    rows = []
    for at in sorted(atk.attack.unique()):
        dd = (W[m1] - W[m0]).dropna() if m1 in W and m0 in W else pd.Series(dtype=float)
        if len(dd) == 0 or at not in dd.index.get_level_values("attack"): continue
        x = dd.xs(at, level="attack")
        if len(x) < 6: continue
        cl = x.groupby(level=["dataset", "seed"]).mean(); n = len(cl)
        bs = np.array([cl.values[rng.integers(0, n, n)].mean() for _ in range(a.boot)]); t = st.ttest_1samp(cl.values, 0.0)
        rows.append(dict(attack=at, m1=m1, m0=m0, n_ctx=int(len(x)), n_clusters=int(n), delta_log=float(cl.mean()), ratio=float(np.exp(cl.mean())), lo=float(np.exp(np.percentile(bs, 2.5))), hi=float(np.exp(np.percentile(bs, 97.5))),
                         p=float(t.pvalue), share_better=float((x < 0).mean())))
    return rows

pairs = [(m, "fedavg") for m in R["methods"] if m != "fedavg"] + [("probe2", "probe"), ("probe2_fl", "fltrust"), ("probe2_fl", "probe2"), ("probe2", "fltrust"), ("bulyan", "fltrust")]
ate = pd.DataFrame([r for m1, m0 in pairs for r in paired(m1, m0)])
if len(ate): ate["p_holm"] = holm(ate.p.values); ate.to_csv(OUT / f"{a.tag}_ate_paired.csv", index=False); R["ate_head"] = ate.round(3).head(40).to_dict("records")

# --- regression with cluster-robust errors (dataset x seed), method x attack interaction
mod = smf.ols("log_ratio ~ C(method, Treatment('fedavg')) * C(attack, Treatment('hide')) + bad + C(dataset)", data=atk).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(atk.cluster)[0]})
mod0 = smf.ols("log_ratio ~ C(method, Treatment('fedavg')) + C(attack, Treatment('hide')) + bad + C(dataset)", data=atk).fit()
R["reg"] = dict(r2=float(mod.rsquared), r2_adj=float(mod.rsquared_adj), r2_no_interaction=float(mod0.rsquared), n=int(mod.nobs), n_params=int(len(mod.params)),
                f_interaction=float(((mod.ssr if False else (mod0.ssr - mod.ssr)) / (mod.df_model - mod0.df_model)) / (mod.ssr / mod.df_resid)), df_inter=int(mod.df_model - mod0.df_model))
R["reg"]["bad_per10pp"] = dict(b=float(mod.params["bad"] * 0.1), p=float(mod.pvalues["bad"]), ci=[float(x * 0.1) for x in mod.conf_int().loc["bad"]])

# --- mediation: method -> weight separation (AUC_att) -> harm; product of coefficients, cluster bootstrap
def mediation(treat, ctrl, attack, M="auc_attackers", B=1000):
    s = atk[(atk.attack == attack) & atk.method.isin([treat, ctrl])].dropna(subset=[M]).copy()
    if len(s) < 40 or s.method.nunique() < 2: return None
    s["T"] = (s.method == treat).astype(float); cl = s.cluster.unique(); out = []
    def est(x):
        a_ = smf.ols(f"{M} ~ T + bad + C(dataset)", data=x).fit().params["T"]; f = smf.ols(f"log_ratio ~ T + {M} + bad + C(dataset)", data=x).fit(); c_ = smf.ols("log_ratio ~ T + bad + C(dataset)", data=x).fit().params["T"]
        return a_, f.params[M], f.params["T"], c_, a_ * f.params[M]
    e = est(s); grp = {c: g for c, g in s.groupby("cluster")}
    for _ in range(B):
        pick = rng.choice(cl, len(cl)); x = pd.concat([grp[c].assign(cluster=f"{c}_{i}") for i, c in enumerate(pick)], ignore_index=True)
        try: out.append(est(x))
        except Exception: pass
    o = np.array(out); ci = np.percentile(o[:, 4], [2.5, 97.5]) if len(o) else [np.nan] * 2
    return dict(attack=attack, treat=treat, ctrl=ctrl, a_T_to_M=float(e[0]), b_M_to_Y=float(e[1]), direct=float(e[2]), total=float(e[3]), indirect=float(e[4]), indirect_lo=float(ci[0]), indirect_hi=float(ci[1]), share_mediated=float(e[4] / e[3]) if abs(e[3]) > 1e-9 else None, n=int(len(s)))
med = [r for at in ("hide", "exagg", "bias", "backdoor") for t_, c_ in (("probe", "fedavg"), ("probe2", "fedavg"), ("probe2", "probe"), ("fltrust", "fedavg")) if (r := mediation(t_, c_, at, B=300))]
pd.DataFrame(med).to_csv(OUT / f"{a.tag}_mediation.csv", index=False); R["mediation"] = med

# --- SHAP: XGBoost surrogate on the design factors (explains the simulator, not a cause); method x attack interactivity
try:
    import xgboost as xgb, shap
    from sklearn.model_selection import GroupKFold, cross_val_predict
    X = pd.get_dummies(atk[["method", "attack", "dataset"]]).astype(float); X["bad"] = atk.bad.values; y = atk.log_ratio.values
    mdl = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9, tree_method="hist", n_jobs=8, random_state=0)
    pred = cross_val_predict(mdl, X, y, cv=GroupKFold(5), groups=atk.seed.values); R["surrogate_cv_r2_by_seed"] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    mdl.fit(X, y); ex = shap.TreeExplainer(mdl); idx = rng.choice(len(X), min(len(X), 800), replace=False); sv = ex.shap_values(X.iloc[idx])
    grp = {g: [c for c in X.columns if c.startswith(g + "_") or c == g] for g in ("method", "attack", "dataset", "bad")}
    R["shap_mean_abs_by_factor"] = {g: float(np.abs(sv[:, [X.columns.get_loc(c) for c in cs]]).sum(1).mean()) for g, cs in grp.items()}
    si = ex.shap_interaction_values(X.iloc[idx[:300]]); mc = [X.columns.get_loc(c) for c in grp["method"]]; ac = [X.columns.get_loc(c) for c in grp["attack"]]
    inter = np.abs(si[:, mc][:, :, ac]).mean(0); R["shap_inter_method_x_attack"] = pd.DataFrame(inter, index=[X.columns[i][7:] for i in mc], columns=[X.columns[i][7:] for i in ac]).round(4).to_dict("index")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6)); order = np.argsort(-np.abs(sv).mean(0))[:12]
    ax[0].barh([X.columns[i] for i in order][::-1], np.abs(sv).mean(0)[order][::-1], color="#2b6cb0"); ax[0].set_title("Mean |SHAP| (ln harm)", fontsize=9)
    im = ax[1].imshow(inter, cmap="Blues"); ax[1].set_xticks(range(len(ac))); ax[1].set_xticklabels([X.columns[i][7:] for i in ac], rotation=45, ha="right", fontsize=7); ax[1].set_yticks(range(len(mc))); ax[1].set_yticklabels([X.columns[i][7:] for i in mc], fontsize=7)
    ax[1].set_title("SHAP interaction method × attack", fontsize=9); fig.colorbar(im, ax=ax[1], fraction=0.04); fig.tight_layout(); fig.savefig(FIG / f"{a.tag}_shap.png", dpi=150); plt.close(fig)
    from lime.lime_tabular import LimeTabularExplainer
    lim = LimeTabularExplainer(X.values, feature_names=list(X.columns), mode="regression", random_state=0)
    ref = atk[(atk.method == "probe") & (atk.attack == "exagg") & (atk.bad == 0.4)]
    if len(ref):
        i0 = X.index.get_loc(ref.index[0]); le = lim.explain_instance(X.iloc[i0].values, mdl.predict, num_features=6); sv0 = ex.shap_values(X.iloc[[i0]])[0]
        R["lime_vs_shap_probe_exagg"] = dict(lime=[(k, round(v, 3)) for k, v in le.as_list()], shap=[(X.columns[j], round(float(sv0[j]), 3)) for j in np.argsort(-np.abs(sv0))[:6]])
except Exception as e: R["shap_error"] = repr(e)

# --- figure: effects of methods relative to FedAvg by attack (forest plot, harm ratio, log scale)
if len(ate):
    sub = ate[ate.m0 == "fedavg"]; ats = sorted(sub.attack.unique()); ms = [m for m in R["methods"] if m != "fedavg"]; fig, ax = plt.subplots(1, len(ats), figsize=(2.2 * len(ats) + 1, 3.4), sharey=True)
    for k, at in enumerate(ats):
        s = sub[sub.attack == at].set_index("m1").reindex(ms); y = np.arange(len(ms)); ax[k].errorbar(s.ratio, y, xerr=[s.ratio - s.lo, s.hi - s.ratio], fmt="o", ms=3, color="#2b6cb0", lw=1); ax[k].axvline(1, color="gray", lw=.7); ax[k].set_xscale("log"); ax[k].set_title(at, fontsize=8)
        ax[k].set_yticks(y); ax[k].set_yticklabels(ms, fontsize=7); ax[k].tick_params(axis="x", labelsize=6)
    fig.suptitle("Paired effect of replacing FedAvg (ratio of harm; <1 = better; 95% cluster bootstrap)", fontsize=8); fig.tight_layout(); fig.savefig(FIG / f"{a.tag}_ate_forest.png", dpi=150); plt.close(fig)
json.dump(R, open(OUT / f"{a.tag}_causal.json", "w"), ensure_ascii=False, indent=1, default=str); print(json.dumps({k: R[k] for k in R if k not in ("ate_head", "mediation", "shap_inter_method_x_attack")}, ensure_ascii=False, default=str)[:3000])
