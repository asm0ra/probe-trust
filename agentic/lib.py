"""Shared functions: loading of answers, ensemble draws, Probe-Trust weights, aggregation methods."""
import json, os, zlib, numpy as np
OUT_DIR = os.environ.get("OUT_DIR", "out")
W = json.load(open("windows.json")); SIG = W["sigma"]; WIN = W["windows"]; DELTA, TAU = 0.5 * SIG, 0.1 * SIG
PROBES = ["pc1", "pc2", "pf1", "pf2"]
PH = {"pc1": ("cong", 22.0), "pc2": ("cong", 24.0), "pf1": ("free", 64.0), "pf2": ("free", 68.0)}   # last value of the probe
N_INST, N_ENS = 12, 10
def load(pool):
    raw = {}
    fn = f"{OUT_DIR}/raw_{pool}.jsonl"
    if not os.path.exists(fn): return None
    for l in open(fn): d = json.loads(l); raw[(d["inst"], d["item"])] = d["forecast"]
    return raw
def forecasts(raw, items):  # (N_INST, len(items), 3) with NaN on a parse error
    F = np.full((N_INST, len(items), 3), np.nan)
    for i in range(N_INST):
        for j, it in enumerate(items):
            f = raw.get((i, it))
            if f is not None: F[i, j] = f
    return F
def probe_violation(raw, kinds=None):  # v per instance; parse errors are skipped; kinds=None -> all probes
    v = np.zeros(N_INST)
    for i in range(N_INST):
        xs = []
        for p in PROBES:
            f = raw.get((i, p)); kind, last = PH[p]
            if f is None or (kinds and PH[p][0] not in kinds): continue
            f = np.array(f); ex = np.maximum(0, f - (last + DELTA)) if kind == "cong" else np.maximum(0, (last - DELTA) - f); xs.append(ex.mean())
        v[i] = np.mean(xs) if xs else 0.0
    return v
def weights(raw, kinds=None): return np.exp(-probe_violation(raw, kinds) / TAU)
def comp(attack, share, wid, r):
    """Ensemble draw: indices of honest and compromised instances (deterministic)."""
    rng = np.random.default_rng(zlib.crc32(f"{attack}|{share}|{wid}|{r}".encode()))
    k = int(round(share * N_ENS)); return rng.permutation(N_INST)[:N_ENS - k], rng.permutation(N_INST)[:k]
def agg(method, F, w):  # F (n,3), w (n,): NaN rows are dropped
    ok = ~np.isnan(F).any(1); F, w = F[ok], w[ok]
    if len(F) == 0: return np.full(3, np.nan)
    if method == "mean": return F.mean(0)
    if method == "median": return np.median(F, 0)
    if method == "trimmed":
        k = max(1, int(0.2 * len(F))); S = np.sort(F, 0); return S[k:len(F) - k].mean(0) if len(F) > 2 * k else S.mean(0)
    if method in ("probe", "probe1"): s = w.sum(); return (w[:, None] * F).sum(0) / s if s > 1e-9 else F.mean(0)
    raise ValueError(method)
def auc(w_h, w_a):
    c = 0.0
    for a in w_h:
        for b in w_a: c += 1.0 if a > b else 0.5 if a == b else 0.0
    return c / (len(w_h) * len(w_a))
