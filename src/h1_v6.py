"""Test v6 = v5 + attack types (hide/flip/noise), heterogeneous data (--noniid: a node has only 30% of the days), method krum40 (f=0.4). Basis v5: the "hide the jam" attack. Attackers (20%) raise the speed in their training labels by +25 mph where the input shows a jam (<40 mph). Metrics: RMSE and bias of the forecast of honest nodes during real events. Basis v4 (backdoor): attackers (20%) train on data with a hidden trigger (feature "sensor status"=0.5 in the last 2 steps) -> forecast "jam" (about -2.5 sigma). Metrics: ASR (share of false jams under the trigger on normal flow), clean error. Basis v3: real events from METR-LA (no artificial scaling) + poisoning of LOCAL DATA with a "jam-like" forgery.
Conditions: clean | atk (single nodes, independent forgeries) | collude (a neighbouring cluster, identical forgeries).
Methods: fedavg, median, trimmed, krum, gct (trust by agreement with graph neighbours), gct_glob (the same without the graph, an ablation)."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p
from data import TrafficData, IN_LEN, OUT_LEN
from model import GRUForecaster
from fedavg import masked_loss

ROOT = Path(__file__).resolve().parents[1]
SPD = 288


def neighbors(data, k=4):
    adj = data.adj.copy(); np.fill_diagonal(adj, 0)
    return [np.argsort(-adj[i])[:k] for i in range(data.N)]


def natural_events(data, neigh, low=0.65, nb_low=0.75, need=3):
    """Real event: the node is slower than 65% of its usual speed for this time of day AND at least 3 of 4 nearest neighbours are also slower than 75%."""
    raw = data.raw.numpy().astype(np.float64); valid = raw > 0
    days = data.T // SPD; r = np.where(valid, raw, np.nan)[: days * SPD].reshape(days, SPD, data.N)
    med = np.nanmedian(r, axis=0)                                    # (288, N)
    ratio = np.full(raw.shape, np.nan); ratio[: days * SPD] = (r / med[None]).reshape(days * SPD, data.N)
    slow = np.nan_to_num(ratio, nan=1.0) < low; slow_nb = np.nan_to_num(ratio, nan=1.0) < nb_low
    cnt = np.stack([slow_nb[:, neigh[i]].sum(1) for i in range(data.N)], axis=1)
    return slow & (cnt >= need) & valid


def fake_episodes(rng, T0, T1, cover=0.35, factor=0.35, ramp=3):
    """Forgery "like a jam": a smooth drop over 3 steps, a hold of 12-36 steps, a smooth return. Returns a multiplier over time (T1-T0,)."""
    m = np.ones(T1 - T0); covered = 0
    while covered < cover * (T1 - T0):
        L = int(rng.integers(12, 37)); s = int(rng.integers(0, T1 - T0 - L - 6)); f = np.ones(L + 6)
        f[:ramp] = np.linspace(1, factor, ramp + 1)[1:]; f[ramp:ramp + L] = factor; f[ramp + L:ramp + L + ramp] = np.linspace(factor, 1, ramp + 1)[1:]; f = f[:L + 6]
        m[s:s + L + 6] = np.minimum(m[s:s + L + 6], f); covered = (m < 1).sum()
    return m


def cluster(data, rng, size):
    adj = data.adj.copy(); np.fill_diagonal(adj, 0); cl = [int(rng.integers(0, data.N))]
    while len(cl) < size:
        sc = adj[cl].sum(0); sc[cl] = -1; cl.append(int(sc.argmax()))
    return cl


def poison(data, bad, condition, seed, cover, factor, ramp=3):
    """Replacement of local training data at the malicious nodes (train only; test is untouched)."""
    a, b = data.bounds["train"]; rng = np.random.default_rng(5000 + seed)
    shared = fake_episodes(rng, a, b, cover, factor, ramp) if condition == "collude" else None
    for n in bad:
        f = shared if shared is not None else fake_episodes(rng, a, b, cover, factor, ramp)
        ft = torch.from_numpy(f).float(); v = data.raw[a:b, n] > 0
        data.raw[a:b, n] = torch.where(v, data.raw[a:b, n] * ft, data.raw[a:b, n])
        data.series[a:b, n, 0] = torch.where(v, (data.raw[a:b, n] - data.mean) / data.std, data.series[a:b, n, 0])


def evaluate(model, data, nodes, ev_mask, split="test", only_events=False):
    a, b = data.bounds[split]; starts = np.arange(a, b - IN_LEN - OUT_LEN + 1, 2); se = cnt = 0.0
    for n in nodes:
        st = starts[ev_mask[starts + IN_LEN, n]] if only_events else starts
        if len(st) == 0: continue
        s = torch.from_numpy(st); xi = s[:, None] + torch.arange(IN_LEN)[None]; yi = s[:, None] + IN_LEN + torch.arange(OUT_LEN)[None]
        with torch.no_grad(): pred = model(data.series[xi, n]) * data.std + data.mean
        raw = data.raw[yi, n]; mm = (raw > 0).float(); se += (((pred - raw) ** 2) * mm).sum().item(); cnt += mm.sum().item()
    return (se / cnt) ** 0.5 if cnt else float("nan")


def asr(model, data, nodes):
    """Share of "jam" forecasts (<40 mph on average) on normal flow (>55 mph): with and without the trigger."""
    a, b = data.bounds["test"]; starts = np.arange(a, b - IN_LEN - OUT_LEN + 1, 6); t = c = n_ = 0
    for n in nodes[::2]:
        s_ = torch.from_numpy(starts); xi = s_[:, None] + torch.arange(IN_LEN)[None]; x = data.series[xi, n]
        sel = ((x[:, -1, 0] * data.std + data.mean) > 55) & (x[:, -1, 1] > 0); x = x[sel]
        if len(x) == 0: continue
        xt = x.clone(); xt[:, -2:, 1] = 0.5
        with torch.no_grad(): pt = (model(xt) * data.std + data.mean).mean(1); pc = (model(x) * data.std + data.mean).mean(1)
        t += (pt < 40).sum().item(); c += (pc < 40).sum().item(); n_ += len(x)
    return t / max(n_, 1), c / max(n_, 1)


def bias_event(model, data, nodes, ev):
    """Mean (forecast - truth), mph, on honest nodes in real events: positive = the model "hides" the jam."""
    a, b = data.bounds["test"]; starts = np.arange(a, b - IN_LEN - OUT_LEN + 1, 2); s_ = c = 0.0
    for n in nodes:
        st = starts[ev[starts + IN_LEN, n]]
        if len(st) == 0: continue
        t = torch.from_numpy(st); xi = t[:, None] + torch.arange(IN_LEN)[None]; yi = t[:, None] + IN_LEN + torch.arange(OUT_LEN)[None]
        with torch.no_grad(): pred = model(data.series[xi, n]) * data.std + data.mean
        raw = data.raw[yi, n]; mm = (raw > 0).float(); s_ += ((pred - raw) * mm).sum().item(); c += mm.sum().item()
    return s_ / max(c, 1)


def auc(w, bad, N):
    """Probability that an honest node has a higher weight than an attacker (0.5 = no separation, 1.0 = perfect)."""
    if not bad: return None
    h = [w[i] for i in range(N) if i not in bad]; m = [w[i] for i in bad]
    return float(np.mean([(x > y) + 0.5 * (x == y) for x in h for y in m]))


def aggregate(method, D, neigh, f_frac):
    if method == 'krum40': method, f_frac = 'krum', 0.4
    n = D.shape[0]
    if method == "fedavg": return D.mean(0), torch.ones(n)
    if method == "median": return D.median(0).values, torch.ones(n)
    if method == "trimmed":
        k = int(0.2 * n); S, _ = D.sort(0); rank = D.argsort(0).argsort(0)
        return S[k:n - k].mean(0), ((rank >= k) & (rank < n - k)).float().mean(1)
    if method == "krum":
        f = int(f_frac * n); dist = torch.cdist(D, D) ** 2
        sc = torch.stack([dist[i][torch.arange(n) != i].sort().values[: n - f - 2].sum() for i in range(n)])
        sel = sc.argsort()[: n - f]; w = torch.zeros(n); w[sel] = 1; return D[sel].mean(0), w
    if method in ("gct", "gct_glob"):
        nm = D.norm(dim=1); Dn = D / nm[:, None].clamp(min=1e-12)
        ref = torch.stack([Dn[neigh[i]].mean(0) for i in range(n)]) if method == "gct" else Dn.median(0).values[None].expand(n, -1)
        c = (Dn * ref).sum(1) / ref.norm(dim=1).clamp(min=1e-12)
        w = torch.sigmoid((c - c.median()) / 0.05)          # relative to the median: honest about 0.5..1, atypical -> 0
        D2 = D * (nm.median() / nm.clamp(min=1e-12)).clamp(max=1)[:, None]
        return (w[:, None] * D2).sum(0) / w.sum().clamp(min=1e-9), w
    raise ValueError(method)


F_ASSUMED = None
ATTACK = 'hide'
NONIID = False


def run(method, condition, seed, rounds, steps, batch, lr, bad_frac, cover, factor, ramp=3):
    data = TrafficData(); N = data.N; rng = np.random.default_rng(seed); torch.manual_seed(seed)
    neigh = neighbors(data); ev = natural_events(data, neigh)
    nbad = int(bad_frac * N)
    if condition == "clean": bad = []
    elif condition == "atk": bad = rng.choice(N, nbad, replace=False).tolist()
    else: bad = cluster(data, np.random.default_rng(7000 + seed), nbad)
    ev_train = ev[data.bounds["train"][0]:data.bounds["train"][1]].mean(0)       # share of events in the node's training data
    bad_set = set(bad)
    hon = [i for i in range(N) if i not in set(bad)]
    heavy = sorted(hon, key=lambda i: -ev_train[i])[: len(hon) // 5]           # honest nodes with the largest number of real events
    nw = data.n_windows("train"); ndays = nw // 288 + 1; node_win = []
    for c_ in range(N):
        ch = np.random.default_rng(9000 + 1000 * seed + c_).choice(ndays, max(2, int(0.3 * ndays)), replace=False); node_win.append(np.where(np.isin(np.arange(nw) // 288, ch))[0])
    g = GRUForecaster(); loc = GRUForecaster(); gv = p2v(g.parameters()).detach().clone(); ws = []
    for r in range(rounds):
        ds = []
        for c in range(N):
            v2p(gv.clone(), loc.parameters()); opt = torch.optim.Adam(loc.parameters(), lr=lr)
            for _ in range(steps):
                idx = node_win[c][rng.integers(0, len(node_win[c]), batch)] if NONIID else rng.integers(0, data.n_windows("train"), batch); x, y, m, _ = data.batch("train", c, idx)
                if c in bad_set and ATTACK == 'hide':
                    y = y.clone(); m = m.clone(); sel = (x[:, -1, 0] * data.std + data.mean) < 40.0; y[sel] = y[sel] + 25.0 / data.std; m[sel] = 1.0
                elif c in bad_set and ATTACK == 'noise':
                    y = y + torch.randn_like(y) * 1.5
                opt.zero_grad(); masked_loss(loc(x), y, m).backward(); opt.step()
            ds.append(p2v(loc.parameters()).detach() - gv)
        for b_ in (bad if ATTACK == 'flip' else []): ds[b_] = -3.0 * ds[b_]
        agg, w = aggregate(method, torch.stack(ds), neigh, F_ASSUMED if F_ASSUMED is not None else (bad_frac if bad else 0.2)); gv = gv + agg
        if r >= rounds // 2: ws.append(w.numpy())
    v2p(gv.clone(), g.parameters()); w = np.mean(ws, 0); wn = w / max(w.mean(), 1e-9)
    return {"method": method, "condition": condition, "seed": seed,
            "bias_event": bias_event(g, data, hon, ev), "rmse_event_honest": evaluate(g, data, hon, ev, only_events=True), "rmse_all_honest": evaluate(g, data, hon, ev),
            "w_event_heavy": float(wn[heavy].mean()), "w_other_honest": float(np.mean([wn[i] for i in hon if i not in set(heavy)])),
            "w_attackers": float(wn[bad].mean()) if bad else None, "auc_attackers": auc(wn, bad, N), "event_share_test": float(ev[data.bounds["test"][0]:].mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--condition", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=25); ap.add_argument("--steps", type=int, default=20); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.003); ap.add_argument("--bad", type=float, default=0.2)
    ap.add_argument("--attack", default="hide"); ap.add_argument("--noniid", type=int, default=0); ap.add_argument("--f_assumed", type=float, default=None); ap.add_argument("--ramp", type=int, default=3); ap.add_argument("--cover", type=float, default=0.35); ap.add_argument("--factor", type=float, default=0.35)
    ap.add_argument("--out", default=str(ROOT / "results/h1_v6.jsonl")); a = ap.parse_args(); F_ASSUMED = a.f_assumed; ATTACK = a.attack; NONIID = bool(a.noniid); torch.set_num_threads(1)
    t0 = time.time(); r = run(a.method, a.condition, a.seed, a.rounds, a.steps, a.batch, a.lr, a.bad, a.cover, a.factor, a.ramp)
    r.update(time_s=round(time.time() - t0), cover=a.cover, factor=a.factor, ramp=a.ramp, bad=a.bad, f_assumed=a.f_assumed, attack=a.attack, noniid=a.noniid); print(json.dumps(r)); open(a.out, "a").write(json.dumps(r) + "\n")
