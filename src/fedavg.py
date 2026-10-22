"""Baseline federated learning (FedAvg) and a centralised control on METR-LA. No attacks and no trust: the starting point."""
import argparse, copy, json, time
import numpy as np
import torch
from data import ROOT, TrafficData
from model import GRUForecaster


def masked_loss(pred, y, m):
    return ((pred - y).abs() * m).sum() / m.sum().clamp(min=1)


def local_train(model, data, client, steps, batch, lr, rng):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = data.n_windows("train")
    for _ in range(steps):
        idx = rng.integers(0, n, batch)
        x, y, m, _ = data.batch("train", client, idx)
        opt.zero_grad(); masked_loss(model(x), y, m).backward(); opt.step()


@torch.no_grad()
def evaluate(model, data, split, clients, per_client=300, seed=0):
    """MAE and RMSE in original units (mph), over real values only."""
    rng = np.random.default_rng(seed); se = ae = cnt = 0.0
    for c in clients:
        idx = rng.integers(0, data.n_windows(split), per_client)
        x, _, m, raw = data.batch(split, c, idx)
        pred = model(x) * data.std + data.mean
        mm = (raw > 0).float()
        ae += ((pred - raw).abs() * mm).sum().item(); se += (((pred - raw) ** 2) * mm).sum().item(); cnt += mm.sum().item()
    return {"mae": ae / cnt, "rmse": (se / cnt) ** 0.5}


def average(states, weights):
    out = copy.deepcopy(states[0])
    w = torch.tensor(weights, dtype=torch.float32); w = w / w.sum()
    for k in out:
        out[k] = sum(w[i] * states[i][k] for i in range(len(states)))
    return out


def run_fedavg(data, rounds, frac, steps, batch, lr, seed, log=print):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    g = GRUForecaster(); local = GRUForecaster(); hist = []
    k = max(1, int(frac * data.N)); t0 = time.time()
    for r in range(1, rounds + 1):
        chosen = rng.choice(data.N, k, replace=False); states = []
        for c in chosen:
            local.load_state_dict(g.state_dict()); local_train(local, data, int(c), steps, batch, lr, rng)
            states.append(copy.deepcopy(local.state_dict()))
        g.load_state_dict(average(states, [1.0] * k))
        if r % 10 == 0 or r == rounds:
            v = evaluate(g, data, "val", range(0, data.N, 4)); hist.append({"round": r, **v}); log(f"round {r:3d}  val MAE {v['mae']:.3f}  RMSE {v['rmse']:.3f}  ({time.time()-t0:.0f} s)")
    return g, hist


def run_central(data, total_steps, batch, lr, seed, log=print):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); g = GRUForecaster(); opt = torch.optim.Adam(g.parameters(), lr=lr); n = data.n_windows("train"); t0 = time.time()
    for s in range(1, total_steps + 1):
        c = int(rng.integers(0, data.N)); idx = rng.integers(0, n, batch); x, y, m, _ = data.batch("train", c, idx)
        opt.zero_grad(); masked_loss(g(x), y, m).backward(); opt.step()
    log(f"centralized: {total_steps} steps in {time.time()-t0:.0f} s")
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60); ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=30); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.003); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--out", default=str(ROOT / "results/baseline.json"))
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    data = TrafficData(); print(f"METR-LA: {data.N} nodes, windows train/val/test = {data.n_windows('train')}/{data.n_windows('val')}/{data.n_windows('test')}")
    t0 = time.time(); g, hist = run_fedavg(data, a.rounds, a.frac, a.steps, a.batch, a.lr, a.seed); t_fl = time.time() - t0
    test_fl = evaluate(g, data, "test", range(data.N)); print(f"FedAvg    test MAE {test_fl['mae']:.3f}  RMSE {test_fl['rmse']:.3f} mph  (time {t_fl:.0f} s)")
    k = max(1, int(a.frac * data.N)); gc = run_central(data, a.rounds * k * a.steps, a.batch, a.lr, a.seed)
    test_c = evaluate(gc, data, "test", range(data.N)); print(f"Central   test MAE {test_c['mae']:.3f}  RMSE {test_c['rmse']:.3f} mph (same number of steps)")
    json.dump({"args": vars(a), "fedavg": test_fl, "central": test_c, "hist": hist, "time_fedavg_s": t_fl}, open(a.out, "w"), indent=1)
