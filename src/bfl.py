"""Vectorised simulator of federated traffic forecasting (same as h1_v6.py, but all clients and several runs are trained in ONE batch).
Speed-up: instead of a client-by-step loop, a batched GRU (R runs x N clients) on CPU or GPU. Semantics of attacks/defences/metrics as in h1_v6.py (hide/flip/noise; fedavg/median/trimmed/krum/krum40/gct).
Usage: python bfl.py --methods fedavg,median --conditions clean,atk --seeds 0,1,2 --bad 0.2 --rounds 25 --steps 20 [--device cuda|cpu]"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch
from torch.nn.utils import parameters_to_vector as p2v
import h1_v6 as ref
from data import TrafficData, IN_LEN, OUT_LEN
from model import GRUForecaster

ROOT = Path(__file__).resolve().parents[1]; H = 32; PD = 3555
ARCH = os.environ.get('PT_ARCH', 'gru')      # gru (default, the paper's main model) | tf (small one-layer transformer) | mlp (two-layer perceptron) | gcn (GRU with a graph-neighbour channel); used to check other architectures
CACHE = ROOT / "results" / "ev_cache.npz"      # for METR-LA; other datasets use ev_cache_<dataset>.npz


def unpack(P):
    n = P.shape[0]
    return (P[:, :192].view(n, 96, 2), P[:, 192:3264].view(n, 96, 32), P[:, 3264:3360], P[:, 3360:3456], P[:, 3456:3552].view(n, 3, 32), P[:, 3552:3555])


def fwd_gru(P, x):
    """P (M,PD): weights of M independent models, x (M,B,12,2) -> (M,B,3). Equivalent to GRUForecaster."""
    Wih, Whh, bih, bhh, Wh, bh = unpack(P); M, B = x.shape[0], x.shape[1]
    h = torch.zeros(M, B, H, device=x.device, dtype=x.dtype); WihT, WhhT = Wih.transpose(1, 2), Whh.transpose(1, 2)
    for t in range(IN_LEN):
        gi = torch.baddbmm(bih[:, None, :], x[:, :, t, :], WihT); gh = torch.baddbmm(bhh[:, None, :], h, WhhT)
        ir, iz, inn = gi.chunk(3, -1); hr, hz, hn = gh.chunk(3, -1)
        r = torch.sigmoid(ir + hr); z = torch.sigmoid(iz + hz); n = torch.tanh(inn + r * hn); h = (1 - z) * n + z * h
    return torch.baddbmm(bh[:, None, :], h, Wh.transpose(1, 2))


TFD, TFH, TFF = 16, 2, 32
TF_SHAPES = [("Win", (TFD, 2)), ("bin", (TFD,)), ("pos", (IN_LEN, TFD)), ("Wqkv", (3 * TFD, TFD)), ("bqkv", (3 * TFD,)), ("Wo", (TFD, TFD)), ("bo", (TFD,)), ("g1", (TFD,)), ("b1", (TFD,)), ("W1", (TFF, TFD)), ("c1", (TFF,)), ("W2", (TFD, TFF)), ("c2", (TFD,)), ("g2", (TFD,)), ("b2", (TFD,)), ("Wh", (OUT_LEN, TFD)), ("bh", (OUT_LEN,))]
MLP_SHAPES = [("W1", (48, IN_LEN * 2)), ("b1", (48,)), ("W2", (48, 48)), ("b2", (48,)), ("W3", (OUT_LEN, 48)), ("b3", (OUT_LEN,))]
GCN_SHAPES = [("Wih", (3 * H, 3)), ("Whh", (3 * H, H)), ("bih", (3 * H,)), ("bhh", (3 * H,)), ("Wh", (OUT_LEN, H)), ("bh", (OUT_LEN,))]      # gcn: GRU with a third input channel, the graph-weighted mean speed of neighbouring sensors (one graph-convolution step)
SHAPES = {"tf": TF_SHAPES, "mlp": MLP_SHAPES, "gcn": GCN_SHAPES}.get(ARCH)
if SHAPES: PD = sum(int(np.prod(s)) for _, s in SHAPES)


def unpack_g(P, shapes):
    n = P.shape[0]; o = 0; out = {}
    for k, s in shapes:
        m = int(np.prod(s)); out[k] = P[:, o:o + m].reshape(n, *s); o += m
    return out


def init_vec(seed):
    """Initial weights for tf/mlp/gcn: normal with scale 1/sqrt(fan_in); LayerNorm: g=1, b=0; positions small."""
    g = torch.Generator().manual_seed(seed); parts = []
    for k, s in SHAPES:
        if k in ("g1", "g2"): t = torch.ones(s)
        elif k.startswith("b") or k.startswith("c"): t = torch.zeros(s)
        elif k == "pos": t = torch.randn(s, generator=g) * 0.1
        else: t = torch.randn(s, generator=g) / (s[-1] ** 0.5)
        parts.append(t.reshape(-1))
    return torch.cat(parts)


def _ln(h, g, b): return (h - h.mean(-1, keepdim=True)) / (h.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt() * g[:, None, None, :] + b[:, None, None, :]


def fwd_tf(P, x):
    w = unpack_g(P, TF_SHAPES); M, B, T, _ = x.shape; d = TFD; dh = d // TFH
    h = torch.einsum("mbti,mdi->mbtd", x, w["Win"]) + w["bin"][:, None, None, :] + w["pos"][:, None]
    qkv = torch.einsum("mbtd,mkd->mbtk", h, w["Wqkv"]) + w["bqkv"][:, None, None, :]; q, k, v = [z.reshape(M, B, T, TFH, dh) for z in qkv.chunk(3, -1)]
    a = torch.softmax(torch.einsum("mbthe,mbshe->mbhts", q, k) / dh ** 0.5, -1); o = torch.einsum("mbhts,mbshe->mbthe", a, v).reshape(M, B, T, d)
    h = _ln(h + torch.einsum("mbtd,mkd->mbtk", o, w["Wo"]) + w["bo"][:, None, None, :], w["g1"], w["b1"])
    f = torch.relu(torch.einsum("mbtd,mkd->mbtk", h, w["W1"]) + w["c1"][:, None, None, :]); f = torch.einsum("mbtk,mdk->mbtd", f, w["W2"]) + w["c2"][:, None, None, :]
    h = _ln(h + f, w["g2"], w["b2"]).mean(2)
    return torch.einsum("mbd,mod->mbo", h, w["Wh"]) + w["bh"][:, None, :]


def fwd_mlp(P, x):
    w = unpack_g(P, MLP_SHAPES); M, B = x.shape[0], x.shape[1]; h = x.reshape(M, B, -1)
    h = torch.relu(torch.baddbmm(w["b1"][:, None, :], h, w["W1"].transpose(1, 2))); h = torch.relu(torch.baddbmm(w["b2"][:, None, :], h, w["W2"].transpose(1, 2)))
    return torch.baddbmm(w["b3"][:, None, :], h, w["W3"].transpose(1, 2))


def fwd_gcn(P, x):
    if x.shape[-1] == 2: x = torch.cat([x, x[..., :1]], -1)      # probes: "neighbours behave like the node itself" (a congested / free-flow area is homogeneous)
    w = unpack_g(P, GCN_SHAPES); M, B = x.shape[0], x.shape[1]; WihT, WhhT = w["Wih"].transpose(1, 2), w["Whh"].transpose(1, 2); h = torch.zeros(M, B, H, device=x.device, dtype=x.dtype)
    for t in range(IN_LEN):
        gi = torch.baddbmm(w["bih"][:, None, :], x[:, :, t, :], WihT); gh = torch.baddbmm(w["bhh"][:, None, :], h, WhhT)
        ir, iz, inn = gi.chunk(3, -1); hr, hz, hn = gh.chunk(3, -1)
        r = torch.sigmoid(ir + hr); z = torch.sigmoid(iz + hz); n = torch.tanh(inn + r * hn); h = (1 - z) * n + z * h
    return torch.baddbmm(w["bh"][:, None, :], h, w["Wh"].transpose(1, 2))


fwd = {"tf": fwd_tf, "mlp": fwd_mlp, "gcn": fwd_gcn}.get(ARCH, fwd_gru)


FWDT = torch.compile(fwd, dynamic=False) if os.environ.get('PT_COMPILE') == '1' else fwd      # faster forward pass for local training (needs CPATH to the Python headers); numbers match eager mode to 1e-7
ADAM_FUSED = os.environ.get('PT_COMPILE') == '1'


def aggregate(method, D, neigh_idx, f_frac):
    if method == 'krum40': method, f_frac = 'krum', 0.4
    n = D.shape[0]; dev = D.device
    if method == "fedavg": return D.mean(0), torch.ones(n, device=dev)
    if method == "median": return D.median(0).values, torch.ones(n, device=dev)
    if method == "trimmed":
        k = int(0.2 * n); S, _ = D.sort(0); rank = D.argsort(0).argsort(0)
        return S[k:n - k].mean(0), ((rank >= k) & (rank < n - k)).float().mean(1)
    if method == "krum":
        f = int(f_frac * n); dist = torch.cdist(D, D) ** 2; dist.fill_diagonal_(float("inf"))
        sc = dist.sort(1).values[:, : n - f - 2].sum(1); sel = sc.argsort()[: n - f]; w = torch.zeros(n, device=dev); w[sel] = 1
        return D[sel].mean(0), w
    if method in ("gct", "gct_glob"):
        nm = D.norm(dim=1); Dn = D / nm[:, None].clamp(min=1e-12)
        ref_ = Dn[neigh_idx].mean(1) if method == "gct" else Dn.median(0).values[None].expand(n, -1)
        c = (Dn * ref_).sum(1) / ref_.norm(dim=1).clamp(min=1e-12); w = torch.sigmoid((c - c.median()) / 0.05)
        D2 = D * (nm.median() / nm.clamp(min=1e-12)).clamp(max=1)[:, None]
        return (w[:, None] * D2).sum(0) / w.sum().clamp(min=1e-9), w
    if method == "normbound":      # Sun et al. 2019 (norm bounding): updates are clipped to the median norm, then averaged
        nm = D.norm(dim=1).clamp(min=1e-12); sc = (nm.median() / nm).clamp(max=1); return (D * sc[:, None]).mean(0), sc
    if method == "bulyan":         # Mhamdi et al. 2018: select theta=n-2f by multi-Krum, then coordinate-wise trimmed mean with beta=theta-2f; f is capped at (n-3)/4
        f = min(int(f_frac * n), (n - 3) // 4); theta = n - 2 * f; dist = torch.cdist(D, D) ** 2; dist.fill_diagonal_(float("inf")); alive = torch.ones(n, dtype=torch.bool, device=dev); chosen = []
        for _ in range(theta):
            m = int(alive.sum().item()); k = max(1, m - f - 2); dd = dist[alive][:, alive]; sc = dd.sort(1).values[:, :k].sum(1); j = alive.nonzero()[sc.argmin(), 0].item(); chosen.append(j); alive[j] = False
        S = D[chosen]; beta = max(1, theta - 2 * f); med = S.median(0).values; idx = (S - med).abs().argsort(0)[:beta]; agg = S.gather(0, idx).mean(0)
        w = torch.zeros(n, device=dev); w[chosen] = 1; return agg, w
    raise ValueError(method)


def foolsgold(H):
    """Fung et al. 2020: weights by the maximum cosine similarity of accumulated updates (collusion -> low weight). H (n,PD) is the history."""
    n = H.shape[0]; Hn = H / H.norm(dim=1, keepdim=True).clamp(min=1e-12); cs = Hn @ Hn.T; cs.fill_diagonal_(-float("inf")); mx = cs.max(1).values; cs.fill_diagonal_(0)
    pard = (mx[:, None] / mx[None, :]).clamp(max=1); cs = torch.where(mx[:, None] < mx[None, :], cs * pard, cs); cs.fill_diagonal_(-float("inf"))
    wv = (1 - cs.max(1).values).clamp(0, 1); wv = wv / wv.max().clamp(min=1e-12); wv = torch.where(wv >= 1, torch.full_like(wv, 0.99), wv)
    wv = (torch.log(wv / (1 - wv).clamp(min=1e-9)) + 0.5).clamp(0, 1); wv = torch.where(torch.isfinite(wv), wv, torch.zeros_like(wv))
    return wv


PROBE_DELTA, PROBE_TAU, PROBE_N = 10.0, 1.0, 96
HIDE_THR, EXAGG_THR = float(os.environ.get("PT_HIDE_THR", 40.0)), float(os.environ.get("PT_EXAGG_THR", 55.0))      # thresholds for a "low" and a "high" input in the hide/exagg attacks (default: mph of METR-LA; for another domain they are set by quantiles)
VALID_REQ = os.environ.get("PT_DOMAIN") == "solar"      # for solar generation the night values are "missing" (mask 0): the attacker selects only windows with a valid last value


_QCACHE = {}


def _q(data, p):
    """Quantile p (0-100) of the network speed over the training part (for normalised probes: profiles are set by network quantiles, not by METR-LA mph)."""
    if data.dataset not in _QCACHE:
        a, b = data.bounds["train"]; r = data.raw[a:b].numpy(); _QCACHE[data.dataset] = r[r > 0]
    return float(np.percentile(_QCACHE[data.dataset], p))


_BORROW = {}
_SRC = {"metr_la": "pems_bay", "pems_bay": "metr_la", "pemsd7m": "pems_bay"}


def _borrow_pool(data):
    """Probes from a FOREIGN public network (PT_PROBEGEN=borrow): windows from the training part of another dataset, values mapped to the target network speeds by quantile mapping
    (the server needs only the speed quantiles of the target network, as for the other normalised probes). Real noise and window shapes; no client data are used."""
    if data.dataset not in _BORROW:
        same = os.environ.get("PT_PROBEGEN") == "same"      # "same": windows of the VALIDATION part of the same network (public history, no labels needed), without mapping
        src = data if same else TrafficData(_SRC[data.dataset]); a, b = src.bounds["val" if same else "train"]; r = src.raw[a:b].numpy(); sv = np.sort(r[r > 0])
        at, bt = data.bounds["train"]; rt = data.raw[at:bt].numpy(); tv = np.sort(rt[rt > 0]); g = np.random.default_rng(31)
        W = []
        while len(W) < 40000:
            s_ = int(g.integers(a, b - IN_LEN)); k = int(g.integers(0, src.N)); w = src.raw[s_:s_ + IN_LEN, k].numpy()
            if (w > 0).all(): W.append(w)
        W = np.array(W); p = np.searchsorted(sv, W) / len(sv); M = W.astype(np.float32) if same else np.interp(p, np.linspace(0, 1, len(tv)), tv).astype(np.float32); _BORROW[data.dataset] = M
    return _BORROW[data.dataset]


def _borrow_probes(data, dev, n, seed, kind):
    M = _borrow_pool(data); rng = np.random.default_rng(seed)
    if kind == "c": sel = np.where((M[:, -1] >= _q(data, 2)) & (M[:, -1] <= _q(data, 10)))[0]
    else: sel = np.where((M[:, -1] >= _q(data, 60)) & (M[:, -1] <= _q(data, 95)) & (M.max(1) - M.min(1) <= 0.6 * data.std))[0]
    pick = rng.choice(sel, n, replace=len(sel) < n); v = M[pick]; X = np.zeros((n, IN_LEN, 2), dtype=np.float32); X[..., 0] = (v - data.mean) / data.std; X[..., 1] = 1.0
    return torch.from_numpy(X).to(dev), torch.from_numpy(v[:, -1].copy()).to(dev)


def make_probes(data, dev, n=PROBE_N, seed=777, norm=False):
    """Congestion probe windows: SYNTHETIC speed profiles (not client data). Input of 12 steps: free flow -> drop to 15-38 mph (onset of a jam) or a steady jam.
    norm=True: ranges come from network speed quantiles (congestion q02-q10, free flow q60-q95), noise 0.12 sigma.
    Returns x (n,12,2) in data normalisation and the last input speed s_last (n,) in mph."""
    if norm and os.environ.get("PT_PROBEGEN") in ("borrow", "same"): return _borrow_probes(data, dev, n, seed, "c")
    rng = np.random.default_rng(seed); X = np.zeros((n, IN_LEN, 2), dtype=np.float32); last = np.zeros(n, dtype=np.float32)
    lo_c, hi_c, lo_f, hi_f, sd = (_q(data, 2), _q(data, 10), _q(data, 60), _q(data, 95), 0.12 * data.std) if norm else (15, 38, 55, 68, 1.5)
    if norm and os.environ.get('PT_PROBERANGE'): pc = [float(x) for x in os.environ['PT_PROBERANGE'].split(',')]; lo_c, hi_c, lo_f, hi_f = _q(data, pc[0]), _q(data, pc[1]), _q(data, pc[2]), _q(data, pc[3])      # robustness check against misspecified probe ranges (network percentiles): congestion lo,hi; free flow lo,hi
    for k in range(n):
        s_last = rng.uniform(lo_c, hi_c); v0 = rng.uniform(lo_f, hi_f); onset = rng.integers(2, IN_LEN) if rng.random() < 0.6 else 0     # onset=0 means a steady jam
        v = np.empty(IN_LEN)
        for t in range(IN_LEN):
            v[t] = s_last if onset == 0 else (v0 if t < onset else s_last + (v0 - s_last) * max(0.0, 1 - (t - onset + 1) / 3.0))
        v = np.clip(v + rng.normal(0, sd, IN_LEN), 5, 75); v[-1] = s_last; last[k] = s_last
        X[k, :, 0] = (v - data.mean) / data.std; X[k, :, 1] = 1.0
    return torch.from_numpy(X).to(dev), torch.from_numpy(last).to(dev)


def make_probes_free(data, dev, n=PROBE_N, seed=778, norm=False):
    """FREE-FLOW probe windows (steady speed 55-68 mph, no drop): the forecast must not fall sharply in them (catches the "false congestion" attack). norm: q60-q95."""
    if norm and os.environ.get("PT_PROBEGEN") in ("borrow", "same"): return _borrow_probes(data, dev, n, seed, "f")
    rng = np.random.default_rng(seed); X = np.zeros((n, IN_LEN, 2), dtype=np.float32); last = np.zeros(n, dtype=np.float32)
    lo, hi, sd = (_q(data, 60), _q(data, 95), 0.12 * data.std) if norm else (55, 68, 1.5)
    if norm and os.environ.get('PT_PROBERANGE'): pc = [float(x) for x in os.environ['PT_PROBERANGE'].split(',')]; lo, hi = _q(data, pc[2]), _q(data, pc[3])
    for k in range(n):
        v = np.clip(rng.uniform(lo, hi) + rng.normal(0, sd, IN_LEN), 5, 75); last[k] = v[-1]; X[k, :, 0] = (v - data.mean) / data.std; X[k, :, 1] = 1.0
    return torch.from_numpy(X).to(dev), torch.from_numpy(last).to(dev)


def make_probes_alt(data, dev, n=1024, seed=997, norm=True):
    """The attacker's own probes (a different family from the server's): noise 0.05-0.35 sigma per sample, shapes: steady level / linear slope / step, wider ranges (congestion q1-q15, free flow q50-q99). Returns ((x_congestion, last), (x_free, last))."""
    rng = np.random.default_rng(seed); out = []
    for kind, (a, b) in (("c", (1, 15)), ("f", (50, 99))):
        lo, hi = _q(data, a), _q(data, b); X = np.zeros((n, IN_LEN, 2), dtype=np.float32); last = np.zeros(n, dtype=np.float32)
        for k in range(n):
            lvl = rng.uniform(lo, hi); sd = rng.uniform(0.05, 0.35) * data.std; f = rng.integers(0, 3); t = np.arange(IN_LEN)
            v = np.full(IN_LEN, lvl) if f == 0 else (lvl + rng.uniform(-6, 6) * (t - IN_LEN + 1) / IN_LEN if f == 1 else np.where(t < rng.integers(3, IN_LEN), lvl + rng.uniform(-8, 8), lvl))
            if kind == "c" and rng.random() < 0.4: v = np.where(t < rng.integers(2, IN_LEN), rng.uniform(_q(data, 60), _q(data, 95)), lvl)      # congestion with an onset (like the server's, but a different transition shape)
            v = np.clip(v + rng.normal(0, sd, IN_LEN), 5, 75); v[-1] = np.clip(lvl, 5, 75) if kind == "c" else v[-1]; last[k] = v[-1]; X[k, :, 0] = (v - data.mean) / data.std; X[k, :, 1] = 1.0
        out.append((torch.from_numpy(X).to(dev), torch.from_numpy(last).to(dev)))
    return out[0], out[1]


def probe_weights2(P, probes, probes_free, data, delta, tau, parts=False):
    """Two-sided probe: violation = forecast too high in congestion probes (> last + delta) + forecast too low in free-flow probes (< last - delta)."""
    N = P.shape[0]; px, last = probes; fx, flast = probes_free
    with torch.no_grad():
        up = (fwd(P, px[None].expand(N, -1, -1, -1)) * data.std + data.mean).mean(2); dn = (fwd(P, fx[None].expand(N, -1, -1, -1)) * data.std + data.mean).mean(2)
        vu = (up - (last[None] + delta)).clamp(min=0).mean(1); vd = ((flast[None] - delta) - dn).clamp(min=0).mean(1); viol = vu + vd
    if parts: return torch.exp(-viol / tau), viol, vu, vd
    return torch.exp(-viol / tau), viol


def flame_agg(Di):
    """FLAME-like aggregation (without noise): cosine distances between updates -> HDBSCAN (min_cluster_size = N/2+1) -> the largest cluster is taken as "honest" -> norm clipping to the median -> mean.
    Assumes an honest majority: with a malicious majority the largest cluster is the attackers. Re-implementation of the idea of FLAME (Nguyen et al., USENIX Security 2022), not the original code."""
    from sklearn.cluster import HDBSCAN
    N = Di.shape[0]; nm = Di.norm(dim=1).clamp(min=1e-12); Dn = Di / nm[:, None]
    C = (1 - (Dn @ Dn.T).clamp(-1, 1)).double().cpu().numpy(); np.fill_diagonal(C, 0.0); C = np.maximum((C + C.T) / 2, 0.0)
    lab = HDBSCAN(min_cluster_size=N // 2 + 1, min_samples=1, metric="precomputed", allow_single_cluster=True).fit_predict(C)
    good = np.ones(N, dtype=bool)
    if (lab >= 0).any(): good = lab == np.bincount(lab[lab >= 0]).argmax()
    w = torch.from_numpy(good).to(Di.device).float(); Dc = Di * (nm.median() / nm).clamp(max=1.0)[:, None]
    return (w[:, None] * Dc).sum(0) / w.sum().clamp(min=1.0), w


def rel_weights(viol, std, beta=1.0, t0=2.0):
    """Cohort-relative score (in the spirit of receiver-domain behavioural probing: median and MAD over clients): s_i = max(0, (v_i - med)/(MAD + 0.1 sigma)); w_i = exp(-beta*max(0, s_i - t0)). Used as an ablation "absolute rule vs relative" (needs an honest majority)."""
    med = viol.median(); mad = (viol - med).abs().median(); s = ((viol - med) / (mad + 0.1 * std)).clamp(min=0)
    return torch.exp(-beta * (s - t0).clamp(min=0))


def fedmid_weights(P, data, dev, n=64, seed=555, t0=2.0):
    """Simplified re-implementation of the idea of FedMID (random Gaussian inputs + relational distances of outputs + cohort outliers): the server feeds n random inputs to all submitted models,
    builds the matrix of pairwise distances between outputs (normalised by its mean), scores a client by the distance of its matrix to the median matrix of the cohort, then median/MAD over clients. Not the original FedMID code (which uses intermediate layers)."""
    g = torch.Generator().manual_seed(seed); x = torch.zeros(n, IN_LEN, 2); x[..., 0] = torch.randn(n, IN_LEN, generator=g); x[..., 1] = 1.0; x = x.to(dev); Nc = P.shape[0]
    with torch.no_grad():
        out = fwd(P, x[None].expand(Nc, -1, -1, -1)); D = torch.cdist(out, out); D = D / D.mean((1, 2), keepdim=True).clamp(min=1e-9)
        s = (D - D.median(0).values[None]).flatten(1).norm(dim=1); med = s.median(); mad = (s - med).abs().median(); z = ((s - med) / (mad + 0.1 * med + 1e-9)).clamp(min=0)
    return torch.exp(-(z - t0).clamp(min=0))


def probe_weights(P, probes, data, delta, tau, rng_pen=False):
    """P (N,PD): local models after training. Violation = by how much the mean 3-step forecast exceeds "last speed + delta" (in a jam the speed must not rise sharply).
    Weight = exp(-violation/tau): soft, no exclusion and no knowledge of the attacker share."""
    px, last = probes; N = P.shape[0]
    with torch.no_grad():
        pred = fwd(P, px[None].expand(N, -1, -1, -1)) * data.std + data.mean              # (N, n, 3)
        viol = (pred.mean(2) - (last[None] + delta)).clamp(min=0).mean(1)                    # (N,)
        if rng_pen: viol = viol + ((pred < 5) | (pred > 85)).float().mean((1, 2)) * 20.0     # physically impossible speeds
    return torch.exp(-viol / tau), viol


class Ctx:
    def __init__(self, dev, dataset="metr_la"):
        self.dev = dev; self.data = TrafficData(dataset); self.dataset = dataset; d = self.data; self.N = d.N; self.probes = None; self.att_probes = None; self.opt_pool = None; self.ap = None; self.root_nodes = None; self.probes_free = None
        self.neigh = ref.neighbors(d)
        cache = CACHE if dataset == "metr_la" else ROOT / "results" / f"ev_cache_{dataset}.npz"
        if cache.exists():
            z = np.load(cache); self.ev = z["ev"]
        else:
            ref.SPD = 144 if dataset == "solar" else 288; self.ev = ref.natural_events(d, self.neigh)
            if dataset == "solar": tod = np.arange(d.T) % 144; self.ev = self.ev & ((tod >= 54) & (tod < 90))[:, None]      # solar events are midday dips (cloudiness), 9:00-15:00
            np.savez_compressed(cache, ev=self.ev)
        self.neigh_idx = torch.tensor(np.stack(self.neigh), device=dev)
        self.series = d.series.to(dev); self.raw = d.raw.to(dev)
        if ARCH == "gcn":      # third channel: mean normalised speed of graph neighbours (adjacency weights, no self-loops, valid values only)
            A = torch.tensor(np.array(d.adj), dtype=torch.float32, device=dev); A.fill_diagonal_(0); sp, vl = self.series[..., 0], self.series[..., 1]
            nb = (sp * vl) @ A.T / (vl @ A.T).clamp(min=1e-6); nb = torch.where((vl @ A.T) > 1e-6, nb, sp); self.series = torch.cat([self.series, nb[..., None]], -1)
        a, b = d.bounds["train"]; self.starts_train = torch.from_numpy(d._starts["train"]).to(dev); self.nw = d.n_windows("train")
        self.ev_train = self.ev[a:b].mean(0)
        ta, tb = d.bounds["test"]; self.test_starts = np.arange(ta, tb - IN_LEN - OUT_LEN + 1, 2)
        ar_x, ar_y = torch.arange(IN_LEN, device=dev), torch.arange(OUT_LEN, device=dev)
        self.ar_x, self.ar_y = ar_x, ar_y
        # evaluation windows: all test windows and the event windows of each node
        nodes_all = np.repeat(np.arange(self.N), len(self.test_starts)); st_all = np.tile(self.test_starts, self.N)
        evsel = self.ev[st_all + IN_LEN, nodes_all]
        self.ev_nodes, self.ev_starts = nodes_all[evsel], st_all[evsel]; self.all_nodes, self.all_starts = nodes_all, st_all

    def node_windows(self, seed):
        nw = self.nw; ndays = nw // 288 + 1; out = []
        for c in range(self.N):
            ch = np.random.default_rng(9000 + 1000 * seed + c).choice(ndays, max(2, int(0.3 * ndays)), replace=False); out.append(np.where(np.isin(np.arange(nw) // 288, ch))[0])
        L = max(len(o) for o in out); pad = np.zeros((self.N, L), dtype=np.int64); ln = np.zeros(self.N, dtype=np.int64)
        for i, o in enumerate(out): pad[i, :len(o)] = o; ln[i] = len(o)
        return torch.from_numpy(pad), torch.from_numpy(ln)

    @torch.no_grad()
    def predict(self, g, nodes, starts, chunk=200000):
        out = []; d = self.data
        for i in range(0, len(nodes), chunk):
            n_ = torch.from_numpy(nodes[i:i + chunk]).to(self.dev); s_ = torch.from_numpy(starts[i:i + chunk]).to(self.dev)
            x = self.series[s_[:, None] + self.ar_x[None], n_[:, None]]; pr = fwd(g[None], x[None])[0] * d.std + d.mean
            raw = self.raw[s_[:, None] + IN_LEN + self.ar_y[None], n_[:, None]]; out.append((pr, raw, n_))
        return out

    def metrics(self, g, hon_mask):
        res = {}
        for name, nodes, starts in (("event", self.ev_nodes, self.ev_starts), ("all", self.all_nodes, self.all_starts)):
            sel = hon_mask[nodes]; nd, st = nodes[sel], starts[sel]; se = cnt = bs = 0.0
            for pr, raw, _ in self.predict(g, nd, st):
                mm = (raw > 0).float(); se += (((pr - raw) ** 2) * mm).sum().item(); bs += ((pr - raw) * mm).sum().item(); cnt += mm.sum().item()
            res[name] = ((se / cnt) ** 0.5 if cnt else float("nan"), bs / max(cnt, 1))
        return res


def _bd_shift(ctx, g, cap=20000):
    """Backdoor success: mean forecast shift (mph) when the trigger appears in the input (two oldest steps = 3 sigma), on the event windows of the test part."""
    rng = np.random.default_rng(5); sel = rng.choice(len(ctx.ev_nodes), min(cap, len(ctx.ev_nodes)), replace=False); d = ctx.data
    with torch.no_grad():
        n_ = torch.from_numpy(ctx.ev_nodes[sel]).to(ctx.dev); s_ = torch.from_numpy(ctx.ev_starts[sel]).to(ctx.dev)
        x = ctx.series[s_[:, None] + ctx.ar_x[None], n_[:, None]]; x2 = x.clone(); x2[:, :2, 0] = 3.0
        return ((fwd(g[None], x2[None])[0] - fwd(g[None], x[None])[0]) * d.std).mean().item()


def auc(w, bad, N):
    if not bad: return None
    h = [w[i] for i in range(N) if i not in set(bad)]; m = [w[i] for i in bad]
    return float(np.mean([(x > y) + 0.5 * (x == y) for x in h for y in m]))


def run_batch(ctx, specs, rounds, steps, batch, lr, attack, noniid, f_assumed, verbose=True, delta=PROBE_DELTA, tau=PROBE_TAU, hide_mag=25.0, rotate=False, aware_k=24, lam=5.0, sub_probe=True, norm=False, adapt_c=0.5, collude=False, root_n=10, root_mode='random', root_poison=0):
    """specs: list of dict(method, condition, seed, bad). All runs are trained in one batch."""
    R, N, dev, d = len(specs), ctx.N, ctx.dev, ctx.data; t0 = time.time()
    delta_in, tau_in = delta, tau
    if norm: delta, tau = delta * d.std, tau * d.std          # thresholds in fractions of the network speed sigma (sigma: METR-LA 12.8, BAY 9.4, D7 13.3 mph)
    ADAPT = ('adapt_known', 'adapt_unk', 'adapt_fl', 'adapt_all', 'adapt_gen'); sig = [dict() for _ in range(R)]; hist = torch.zeros(R, N, PD, device=dev) if any(sp["method"] == "foolsgold" for sp in specs) else None
    bad_sets, inits, nwin = [], [], []
    for sp in specs:
        rng = np.random.default_rng(sp["seed"]); torch.manual_seed(sp["seed"]); nbad = int(sp["bad"] * N)
        if sp["condition"] == "clean": bad = []
        elif sp["condition"] == "atk": bad = rng.choice(N, nbad, replace=False).tolist()
        else: bad = ref.cluster(d, np.random.default_rng(7000 + sp["seed"]), nbad)
        bad_sets.append(bad); inits.append(p2v(GRUForecaster().parameters()).detach().clone() if SHAPES is None else init_vec(sp["seed"])); nwin.append(ctx.node_windows(sp["seed"]) if noniid else None)
    gv = torch.stack(inits).to(dev)                                    # (R, PD)
    bad_mask = torch.zeros(R, N, dtype=torch.bool)
    for i, b in enumerate(bad_sets): bad_mask[i, b] = True
    bad_mask = bad_mask.to(dev).reshape(R * N)
    cl = torch.arange(N, device=dev).repeat(R)                         # client id for each of the R*N models
    if noniid:
        pad = torch.stack([w[0] for w in nwin]).reshape(R * N, -1).to(dev); ln = torch.stack([w[1] for w in nwin]).reshape(R * N).to(dev)
    g_t = torch.Generator(device=dev); g_t.manual_seed(1234 + specs[0]["seed"]); ws = [[] for _ in range(R)]
    for r in range(rounds):
        P = gv[:, None, :].expand(R, N, PD).reshape(R * N, PD).clone().requires_grad_(True); opt = torch.optim.Adam([P], lr=lr, fused=ADAM_FUSED)
        for _ in range(steps):
            u = torch.rand(R * N, batch, device=dev, generator=g_t)
            if noniid: idx = pad.gather(1, (u * ln[:, None]).long().clamp(max=pad.shape[1] - 1))
            else: idx = (u * ctx.nw).long().clamp(max=ctx.nw - 1)
            st = ctx.starts_train[idx]                                       # (RN, B)
            x = ctx.series[st[..., None] + ctx.ar_x, cl[:, None, None]]       # (RN,B,12,2)
            yy = ctx.series[st[..., None] + IN_LEN + ctx.ar_y, cl[:, None, None]]; y, m = yy[..., 0], yy[..., 1]
            if attack in ('hide', 'hide_aware', 'hide_opt', 'scaleup') or attack in ADAPT:
                sel = ((x[:, :, -1, 0] * d.std + d.mean) < HIDE_THR) & bad_mask[:, None] & ((x[:, :, -1, 1] > 0) if VALID_REQ else True)
                y = y + sel[..., None].float() * (hide_mag / d.std); m = torch.where(sel[..., None], torch.ones_like(m), m)
                if attack == 'hide_aware':      # the attacker knows the probe windows and honestly trains a part of the batch on them (answer = last speed)
                    if ctx.att_probes is None: ctx.att_probes = make_probes(d, dev)
                    px, pl = ctx.att_probes; k = min(aware_k, px.shape[0], batch); pick = torch.randint(0, px.shape[0], (k,), device=dev, generator=g_t)
                    bm = bad_mask[:, None, None]
                    x = torch.cat([torch.where(bm[..., None], px[pick][None].expand(R * N, -1, -1, -1), x[:, :k]), x[:, k:]], 1)
                    yp = ((pl[pick] - d.mean) / d.std)[None, :, None].expand(R * N, -1, OUT_LEN)
                    y = torch.cat([torch.where(bm, yp, y[:, :k]), y[:, k:]], 1); m = torch.cat([torch.where(bm, torch.ones_like(m[:, :k]), m[:, :k]), m[:, k:]], 1)
            if attack in ('adapt_known', 'adapt_unk', 'adapt_all', 'adapt_gen'):      # the attacker trains a part of the batch on probes of BOTH sides with an "honest" answer (last speed): the server's (known) or its own, different ones (unk/all)
                if ctx.ap is None:
                    if attack == 'adapt_known': ctx.ap = (make_probes(d, dev, norm=norm), make_probes_free(d, dev, norm=norm))
                    elif attack == 'adapt_gen': ctx.ap = make_probes_alt(d, dev, n=1024, seed=997, norm=norm)      # the attacker does not know the server's generator: own probes of another family (different noise, shapes, ranges)
                    else: ctx.ap = (make_probes(d, dev, n=1024, seed=999, norm=norm), make_probes_free(d, dev, n=1024, seed=998, norm=norm))
                (cx, cl_), (fx_, fl_) = ctx.ap; k2 = min(aware_k // 2, batch // 4); bm = bad_mask[:, None, None]
                pc = torch.randint(0, cx.shape[0], (k2,), device=dev, generator=g_t); pf = torch.randint(0, fx_.shape[0], (k2,), device=dev, generator=g_t)
                xa = torch.cat([cx[pc], fx_[pf]], 0); la = torch.cat([cl_[pc], fl_[pf]], 0); ya = ((la - d.mean) / d.std)[None, :, None].expand(R * N, -1, OUT_LEN); kk = 2 * k2
                x = torch.cat([torch.where(bm[..., None], xa[None].expand(R * N, -1, -1, -1), x[:, :kk]), x[:, kk:]], 1)
                y = torch.cat([torch.where(bm, ya, y[:, :kk]), y[:, kk:]], 1); m = torch.cat([torch.where(bm, torch.ones_like(m[:, :kk]), m[:, :kk]), m[:, kk:]], 1)
            if attack == 'exagg':      # "false congestion": in free flow (last speed >= 55) the attacker lowers the forecast by hide_mag; the probe penalises only an over-forecast
                sel = ((x[:, :, -1, 0] * d.std + d.mean) >= EXAGG_THR) & bad_mask[:, None] & ((x[:, :, -1, 1] > 0) if VALID_REQ else True)
                y = y - sel[..., None].float() * (hide_mag / d.std); m = torch.where(sel[..., None], torch.ones_like(m), m)
            elif attack == 'bias':       # slow bias: +hide_mag on all windows (below the probe threshold if hide_mag < delta)
                y = y + bad_mask[:, None, None].float() * (hide_mag / d.std)
            elif attack == 'backdoor':    # trigger (two oldest steps = 3 sigma) -> forecast +hide_mag; on clean inputs the model is trained honestly
                kb = int(0.3 * batch); ib = bad_mask.nonzero()[:, 0]
                if len(ib): x[ib, :kb, :2, 0] = 3.0; y[ib, :kb] += hide_mag / d.std; m[ib, :kb] = 1.0
            elif attack == 'noise':
                y = y + torch.randn(y.shape, device=dev, generator=g_t) * 1.5 * bad_mask[:, None, None].float()
            loss = (((FWDT(P, x) - y).abs() * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)).sum()
            if attack == 'hide_opt':     # optimising adaptive attacker: the poisoning loss gets a penalty for violating the physics on ITS OWN synthetic probes (a different sample from the server's)
                if ctx.opt_pool is None: ctx.opt_pool = make_probes(d, dev, n=1024, seed=999)
                px_, pl_ = ctx.opt_pool; ii = torch.randint(0, px_.shape[0], (16,), device=dev, generator=g_t)
                pr_ = fwd(P, px_[ii][None].expand(R * N, -1, -1, -1)) * d.std + d.mean
                pen = (pr_.mean(2) - (pl_[ii][None] + delta)).clamp(min=0).mean(1)
                loss = loss + lam * (pen * bad_mask.float()).sum()
            if attack in ('adapt_unk', 'adapt_all', 'adapt_gen'):     # + a penalty for violating the physics on both sides on its own probes (a different sample from the server's); knows delta
                (cx, cl_), (fx_, fl_) = ctx.ap; ii = torch.randint(0, cx.shape[0], (16,), device=dev, generator=g_t); jj = torch.randint(0, fx_.shape[0], (16,), device=dev, generator=g_t)
                pu = fwd(P, cx[ii][None].expand(R * N, -1, -1, -1)) * d.std + d.mean; pd_ = fwd(P, fx_[jj][None].expand(R * N, -1, -1, -1)) * d.std + d.mean
                pen = (pu.mean(2) - (cl_[ii][None] + delta)).clamp(min=0).mean(1) + ((fl_[jj][None] - delta) - pd_.mean(2)).clamp(min=0).mean(1)
                loss = loss + lam * (pen * bad_mask.float()).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        D = (P.detach().reshape(R, N, PD) - gv[:, None, :])
        if True:      # the root update is needed by all (FLTrust, server signals, adaptive attacker). FLTrust: the server trains itself on a small clean sample (10 nodes) and weights updates by cosine with its own update
            if getattr(ctx, '_rk', None) != (root_n, root_mode):      # weak root: root_n nodes (nested subsets of the same 10), root_mode='cluster' = graph neighbours (one district); by default the previous behaviour (10 random)
                rn = ref.cluster(d, np.random.default_rng(4242), root_n) if root_mode == 'cluster' else np.random.default_rng(4242).choice(N, 10, replace=False)[:root_n]
                ctx.root_nodes = torch.from_numpy(np.asarray(rn, dtype=np.int64)).to(dev); ctx._rk = (root_n, root_mode)
            P0 = gv.clone().requires_grad_(True); opt0 = torch.optim.Adam([P0], lr=lr)
            for _ in range(steps):
                idx0 = (torch.rand(R, batch, device=dev, generator=g_t) * ctx.nw).long().clamp(max=ctx.nw - 1); st0 = ctx.starts_train[idx0]
                sl0 = torch.randint(0, root_n, (R, batch), device=dev, generator=g_t); nd0 = ctx.root_nodes[sl0]
                x0 = ctx.series[st0[..., None] + ctx.ar_x, nd0[..., None]]; yy0 = ctx.series[st0[..., None] + IN_LEN + ctx.ar_y, nd0[..., None]]; y0, m0 = yy0[..., 0], yy0[..., 1]
                if root_poison:      # poisoned root: the first root_poison of the root_n nodes have corrupted labels (attack style: hide/exagg; otherwise a general shift of hide_mag)
                    pz = (sl0 < root_poison); lv = x0[:, :, -1, 0] * d.std + d.mean
                    if attack in ('hide', 'hide_aware', 'hide_opt') or attack in ADAPT: sel0 = (lv < HIDE_THR) & pz; y0 = y0 + sel0[..., None].float() * (hide_mag / d.std); m0 = torch.where(sel0[..., None], torch.ones_like(m0), m0)
                    elif attack == 'exagg': sel0 = (lv >= EXAGG_THR) & pz; y0 = y0 - sel0[..., None].float() * (hide_mag / d.std); m0 = torch.where(sel0[..., None], torch.ones_like(m0), m0)
                    else: y0 = y0 + pz[..., None].float() * (hide_mag / d.std)
                l0 = (((fwd(P0, x0) - y0).abs() * m0).sum((1, 2)) / m0.sum((1, 2)).clamp(min=1)).sum(); opt0.zero_grad(); l0.backward(); opt0.step()
            D0 = P0.detach() - gv
        for i, sp in enumerate(specs):
            Di = D[i]
            if attack == 'flip' and bad_sets[i]: Di = Di.clone(); Di[bad_sets[i]] = -3.0 * Di[bad_sets[i]]
            if attack == 'alie' and bad_sets[i]:       # "A Little Is Enough": the colluders submit mean - z*std of the honest updates
                from statistics import NormalDist
                nb = len(bad_sets[i]); s_ = N // 2 + 1 - nb; z = 3.0 if s_ <= 0 else min(3.0, NormalDist().inv_cdf(min(0.999, max(0.5, (N - s_) / N))))
                gm = torch.ones(N, dtype=torch.bool, device=dev); gm[bad_sets[i]] = False; Di = Di.clone(); Di[bad_sets[i]] = Di[gm].mean(0) - z * Di[gm].std(0)
            if attack == 'fang' and bad_sets[i]:      # Fang et al. (USENIX Sec. 2020), full knowledge, attack on the median/trimmed mean: per coordinate against the direction of the honest mean, beyond the honest range (b = 2)
                gm = torch.ones(N, dtype=torch.bool, device=dev); gm[bad_sets[i]] = False; Di = Di.clone(); Hh = Di[gm]; sgn = torch.sign(Hh.mean(0)); mn, mxx = Hh.min(0).values, Hh.max(0).values
                low = torch.where(mn > 0, mn / 2.0, mn * 2.0); high = torch.where(mxx > 0, mxx * 2.0, mxx / 2.0); Di[bad_sets[i]] = torch.where(sgn > 0, low, high)[None]
            if attack == 'scaleup' and bad_sets[i]: Di = Di.clone(); Di[bad_sets[i]] = Di[bad_sets[i]] * 10.0      # amplification of the poisoned update (model replacement, Bagdasaryan et al.)
            if attack == 'minmax' and bad_sets[i]:      # Min-Max (Shejwalkar & Houmansadr, NDSS 2021), full knowledge: m = mean - gamma*std of the honest updates, gamma is the maximum at which the distance from m to any honest update does not exceed the largest pairwise distance between honest updates
                gm = torch.ones(N, dtype=torch.bool, device=dev); gm[bad_sets[i]] = False; Di = Di.clone(); Hh = Di[gm]; mu = Hh.mean(0); dirn = -Hh.std(0); thr = torch.cdist(Hh, Hh).max()
                lo_, hi_ = 0.0, 50.0
                for _ in range(30):
                    mid_ = (lo_ + hi_) / 2; ok_ = torch.cdist((mu + mid_ * dirn)[None], Hh).max() <= thr
                    lo_, hi_ = (mid_, hi_) if ok_ else (lo_, mid_)
                Di[bad_sets[i]] = (mu + lo_ * dirn)[None]
            if attack in ('adapt_fl', 'adapt_all') and bad_sets[i]:      # the attacker knows the FLTrust root update: submits cos=adapt_c with it, the rest is poison (orthogonal component), norm = the root norm
                Di = Di.clone(); u0 = D0[i] / D0[i].norm().clamp(min=1e-12); pb = Di[bad_sets[i]]; po = pb - (pb @ u0)[:, None] * u0[None]; po = po / po.norm(dim=1, keepdim=True).clamp(min=1e-12)
                Di[bad_sets[i]] = D0[i].norm() * (adapt_c * u0[None] + (1 - adapt_c ** 2) ** 0.5 * po)
            if collude and bad_sets[i]: Di = Di.clone(); Di[bad_sets[i]] = Di[bad_sets[i]].mean(0, keepdim=True)      # collusion: all attackers submit the same update
            fa = f_assumed if f_assumed is not None else (sp["bad"] if bad_sets[i] else 0.2)
            if ctx.probes is None or rotate: ctx.probes = make_probes(d, dev, seed=(1000 + r) if rotate else 777, norm=norm)
            if ctx.probes_free is None: ctx.probes_free = make_probes_free(d, dev, norm=norm)
            if r < 3:      # server signals (no labels): probe violations, cosine with the root, norms, pairwise similarity; averaged over rounds 1-3
                with torch.no_grad():
                    _, vv, vu_, vd_ = probe_weights2(gv[i][None] + Di, ctx.probes, ctx.probes_free, d, delta, tau, parts=True); nm_ = Di.norm(dim=1).clamp(min=1e-12); n0_ = D0[i].norm().clamp(min=1e-12); cr_ = (Di @ D0[i]) / (nm_ * n0_)
                    ii_ = torch.randperm(N, device=dev, generator=g_t)[:64]; Dn_ = Di[ii_] / nm_[ii_, None]; cm_ = Dn_ @ Dn_.T; cm_.fill_diagonal_(-1)
                    sg = dict(vu=vu_.mean().item() / d.std, vd=vd_.mean().item() / d.std, v50=vv.median().item() / d.std, v90=vv.quantile(0.9).item() / d.std, fv=(vv > 0.5 * d.std).float().mean().item(),
                              cos_mean=cr_.mean().item(), cos_p10=cr_.quantile(0.1).item(), cos_neg=(cr_ < 0).float().mean().item(), nrat=(nm_.median() / n0_).item(), ncv=(nm_.std() / nm_.mean()).item(),
                              pair_mean=cm_.sum().item() / (63 * 64), pair_max=cm_.max(1).values.mean().item())
                for k_, v_ in sg.items(): sig[i][k_] = sig[i].get(k_, 0.0) + v_ / 3.0
            if sp["method"] in ("probe", "probe_clip", "probe2", "probe2_fl", "probe2_clip", "probe2_rel", "fedmid_like"):
                clip = sp["method"] in ("probe_clip", "probe2_clip"); Psub = gv[i][None] + Di if sub_probe else P.detach().reshape(R, N, PD)[i]       # sub_probe: the probe looks at the SUBMITTED model (important for flip/ALIE)
                if sp["method"] in ("probe2", "probe2_fl", "probe2_clip", "probe2_rel"): w, viol = probe_weights2(Psub, ctx.probes, ctx.probes_free, d, delta, tau)
                elif sp["method"] == "fedmid_like": w, viol = fedmid_weights(Psub, d, dev), None
                else: w, viol = probe_weights(Psub, ctx.probes, d, delta, tau, rng_pen=clip)
                if sp["method"] == "probe2_rel": w = rel_weights(viol, d.std)
                if sp["method"] == "probe2_fl":
                    nm = Di.norm(dim=1).clamp(min=1e-12); n0 = D0[i].norm().clamp(min=1e-12); w = w * torch.relu((Di @ D0[i]) / (nm * n0)); Di = Di * (n0 / nm)[:, None]
                if clip:      # + clipping of the update norm by the median (against "exploding" attacks) and into the weight
                    nm = Di.norm(dim=1); Di = Di * (nm.median() / nm.clamp(min=1e-12)).clamp(max=1)[:, None]
                agg = (w[:, None] * Di).sum(0) / w.sum().clamp(min=1e-9)
            elif sp["method"] == "fltrust":
                nm = Di.norm(dim=1).clamp(min=1e-12); n0 = D0[i].norm().clamp(min=1e-12); w = torch.relu((Di @ D0[i]) / (nm * n0))
                agg = (w[:, None] * Di * (n0 / nm)[:, None]).sum(0) / w.sum().clamp(min=1e-9)
            elif sp["method"] == "foolsgold":
                hist[i] += Di; w = foolsgold(hist[i]); agg = (w[:, None] * Di).sum(0) / w.sum() if w.sum() > 1e-9 else Di.mean(0)
            elif sp["method"] == "flame_like": agg, w = flame_agg(Di)
            elif sp["method"] == "flame_full": agg, w = flame_agg(Di); S_ = Di.norm(dim=1).median(); agg = agg + torch.randn_like(agg) * 0.001 * S_      # + noise lambda*S (lambda = 0.001, as in the original paper for training with noise at small lambda)
            elif sp["method"] == "zenopp":      # Zeno++ (Xie, Koyejo, Gupta, ICML 2020): the server has a root update v; a candidate u is clipped to ||v|| and accepted if <v,u> - rho*||u||^2 >= -eps; mean of the accepted. rho = 0.1*||v||^0 (not tuned), eps = 0. Re-implementation from the paper.
                v_ = D0[i]; nv = v_.norm().clamp(min=1e-12); nu = Di.norm(dim=1).clamp(min=1e-12); U = Di * (nv / nu).clamp(max=1.0)[:, None]
                sc = (U @ v_) - 0.1 * (U * U).sum(1); w = (sc >= 0).float(); agg = (w[:, None] * U).sum(0) / w.sum().clamp(min=1.0)
            else: agg, w = aggregate(sp["method"], Di, ctx.neigh_idx, fa)
            gv[i] = gv[i] + agg
            if r >= rounds // 2: ws[i].append(w.cpu().numpy())
        if verbose and (r % 5 == 4 or r == rounds - 1): print(f"  round {r+1}/{rounds}  {time.time()-t0:.1f} s", flush=True)
    out = []
    for i, sp in enumerate(specs):
        bad = bad_sets[i]; hon = np.array([k not in set(bad) for k in range(N)]); met = ctx.metrics(gv[i], hon)
        w = np.mean(ws[i], 0); wn = w / max(w.mean(), 1e-9); hon_l = [k for k in range(N) if k not in set(bad)]
        heavy = sorted(hon_l, key=lambda k: -ctx.ev_train[k])[: len(hon_l) // 5]
        out.append({"method": sp["method"], "condition": sp["condition"], "seed": sp["seed"], "bad": sp["bad"], "dataset": ctx.dataset, "attack": attack, "hide_mag": hide_mag, "rotate": rotate, "aware_k": aware_k, "lam": lam,
                    "rmse_event_honest": met["event"][0], "bias_event": met["event"][1], "rmse_all_honest": met["all"][0],
                    "delta": delta_in, "tau": tau_in, "norm": int(norm), "root_n": root_n, "root_mode": root_mode, "root_poison": root_poison, "noniid": int(noniid), "adapt_c": adapt_c, "collude": int(collude), "sig": sig[i], "bd_shift": (_bd_shift(ctx, gv[i]) if attack == "backdoor" else None), "w_event_heavy": float(wn[heavy].mean()), "w_attackers": float(wn[bad].mean()) if bad else None, "auc_attackers": auc(wn, bad, N)})
    return out, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="fedavg"); ap.add_argument("--conditions", default="clean,atk"); ap.add_argument("--seeds", default="0")
    ap.add_argument("--bad", type=float, default=0.2); ap.add_argument("--rounds", type=int, default=25); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=0.003); ap.add_argument("--attack", default="hide")
    ap.add_argument("--noniid", type=int, default=0); ap.add_argument("--f_assumed", type=float, default=None); ap.add_argument("--device", default="auto")
    ap.add_argument("--max_batch", type=int, default=8); ap.add_argument("--dataset", default="metr_la"); ap.add_argument("--hide_mag", type=float, default=25.0); ap.add_argument("--rotate", type=int, default=0); ap.add_argument("--aware_k", type=int, default=24); ap.add_argument("--lam", type=float, default=5.0); ap.add_argument("--delta", type=float, default=PROBE_DELTA); ap.add_argument("--tau", type=float, default=PROBE_TAU); ap.add_argument("--out", default=str(ROOT / "results/bfl.jsonl")); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--norm", type=int, default=0); ap.add_argument("--adapt_c", type=float, default=0.5); ap.add_argument("--collude", type=int, default=0)
    a = ap.parse_args(); dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)); torch.set_num_threads(a.threads)
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    ctx = Ctx(dev, a.dataset)
    specs = [dict(method=m, condition=c, seed=int(s), bad=a.bad) for m in a.methods.split(",") for c in a.conditions.split(",") for s in a.seeds.split(",")]
    t_all = time.time(); res = []
    for i in range(0, len(specs), a.max_batch):
        chunk = specs[i:i + a.max_batch]; out, t = run_batch(ctx, chunk, a.rounds, a.steps, a.batch, a.lr, a.attack, bool(a.noniid), a.f_assumed, True, a.delta, a.tau, a.hide_mag, bool(a.rotate), a.aware_k, a.lam, True, bool(a.norm), a.adapt_c, bool(a.collude))
        for o in out: o["time_batch_s"] = round(t, 1); print(json.dumps(o), flush=True); open(a.out, "a").write(json.dumps(o) + "\n")
        res += out; print(f"batch of {len(chunk)} runs: {t:.1f} s ({t/len(chunk):.1f} s per run)", flush=True)
    print(f"total {len(specs)} runs in {time.time()-t_all:.1f} s")
