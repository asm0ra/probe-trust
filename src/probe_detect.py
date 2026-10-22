"""Reviewer test: can a classifier tell the server's probes from real windows? (If yes, an attacker can behave honestly on probes and harm on real data.)
Real windows: from the training part (to train the detector) and the test part (to evaluate), separately "matched range" (last speed in the probe range) and "all windows".
Probes: make_probes / make_probes_free with norm=True (as in the paper), or PT_PROBEGEN=same / borrow (probes from real windows)."""
import sys, json, numpy as np, torch
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from data import TrafficData, IN_LEN
import bfl


def feats(x):                       # x: (n,12) normalised speed -> engineered features
    d = np.diff(x, axis=1); return np.c_[x, d, np.abs(d).mean(1), d.std(1), (d[:, 1:] * d[:, :-1]).mean(1) / (d.var(1) + 1e-6), x.std(1)]


def real_windows(data, split, n, rng, lo=None, hi=None):
    a, b = data.bounds[split]; out = []
    while len(out) < n:
        s = int(rng.integers(a, b - IN_LEN - 3)); k = int(rng.integers(0, data.N)); w = data.raw[s:s + IN_LEN, k].numpy()
        if (w <= 0).any(): continue
        if lo is not None and not (lo <= w[-1] <= hi): continue
        out.append((w - data.mean) / data.std)
    return np.array(out, dtype=np.float32)


def probe_set(data, kind, n, seed, gen):
    f = {"c": bfl.make_probes, "f": bfl.make_probes_free}[kind]
    x = f(data, "cpu", n=n, seed=seed, norm=True)[0][..., 0].numpy()
    return x


def one(ds, kind, matched, gen, seed=0):
    data = TrafficData(ds); rng = np.random.default_rng(seed)
    lo, hi = (bfl._q(data, 2), bfl._q(data, 10)) if kind == "c" else (bfl._q(data, 60), bfl._q(data, 95))
    rl, rh = (lo, hi) if matched else (None, None)
    n = 2000
    Xp_tr, Xp_te = probe_set(data, kind, n, 11, gen), probe_set(data, kind, n, 12, gen)
    Xr_tr, Xr_te = real_windows(data, "train", n, rng, rl, rh), real_windows(data, "test", n, rng, rl, rh)
    Xtr = np.r_[feats(Xp_tr), feats(Xr_tr)]; ytr = np.r_[np.ones(n), np.zeros(n)]; Xte = np.r_[feats(Xp_te), feats(Xr_te)]; yte = np.r_[np.ones(n), np.zeros(n)]
    res = {}
    for name, m in (("logreg", LogisticRegression(max_iter=2000)), ("boosting", HistGradientBoostingClassifier(max_iter=200))):
        m.fit(Xtr, ytr); res[name] = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    return res, (Xp_tr, Xr_tr, data)


if __name__ == "__main__":
    out = {}
    for ds in ("metr_la", "pems_bay", "pemsd7m"):
        for kind in ("c", "f"):
            for matched in (False, True):
                r, _ = one(ds, kind, matched, "server"); out[f"{ds}|{kind}|{'matched' if matched else 'all'}"] = r; print(ds, kind, matched, r, flush=True)
    json.dump(out, open(Path(__file__).parents[1] / "results/analysis/probe_detect.json", "w"), indent=1)
