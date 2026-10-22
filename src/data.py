"""METR-LA: loading, chronological split, normalisation, windows "12 steps back -> 3 steps ahead" (5 minutes each)."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

IN_LEN, OUT_LEN = 12, 3
ROOT = Path(__file__).resolve().parents[1]


def _gauss_adj(dist, sigma2=0.1, eps=0.5):
    """Sensor proximity matrix for PeMS-D7(M) from distances (as in STGCN): exp(-(d/10000)^2/sigma2) with threshold eps."""
    w = dist.astype(np.float64) / 10000.0; w2 = w * w; k = np.exp(-w2 / sigma2)
    return (k * (k >= eps)).astype(np.float32)


class TrafficData:
    def __init__(self, dataset="metr_la", path=None, adj_path=None):
        self.dataset = dataset
        if dataset == "metr_la":
            df = pd.read_hdf(path or ROOT / "data/metr_la.h5"); raw = df.values.astype(np.float32)        # (T, N) speed, 0 = missing
            adj_path = adj_path or ROOT / "data/adj_mx.pkl"
        elif dataset == "pems_bay":
            df = pd.read_csv(ROOT / "data/PEMS-BAY.csv", index_col=0); raw = df.values.astype(np.float32)
            adj_path = ROOT / "data/adj_mx_PEMS-BAY.pkl"
        elif dataset == "pemsd7m":
            raw = np.loadtxt(ROOT / "data/pemsd7m/PeMSD7_V_228.csv", delimiter=",", dtype=np.float32)      # (12672, 228), 44 working days of 288 steps
            adj_path = None
        elif dataset == "solar":      # NREL Solar-Energy (Alabama, 137 plants, 10-minute step, 2006; LSTNet): power, MW; at night 0 = "no value" (like a missing value in traffic)
            raw = np.loadtxt(ROOT / "data/solar/solar_AL.txt.gz", delimiter=",", dtype=np.float32)
            adj_path = None
        else:
            raise ValueError(dataset)
        raw = np.nan_to_num(raw, nan=0.0)
        self.T, self.N = raw.shape
        self.valid = (raw > 0)
        t1, t2 = int(self.T * 0.7), int(self.T * 0.8)
        self.bounds = {"train": (0, t1), "val": (t1, t2), "test": (t2, self.T)}
        tr = raw[:t1][self.valid[:t1]]
        self.mean, self.std = float(tr.mean()), float(tr.std())
        norm = np.where(self.valid, (raw - self.mean) / self.std, 0.0).astype(np.float32)
        # two channels: normalised speed and a "value present" flag
        self.series = torch.from_numpy(np.stack([norm, self.valid.astype(np.float32)], axis=-1))   # (T, N, 2)
        self.raw = torch.from_numpy(raw)
        if adj_path is None:
            self.sensor_ids = list(range(self.N)); self.sensor_index = {i: i for i in range(self.N)}
            if dataset == "solar":      # adjacency by power correlation (the file has no plant coordinates): weight = correlation, neighbors() takes the 4 nearest plants
                tr = raw[: int(self.T * 0.7)]; self.adj = np.nan_to_num(np.corrcoef(tr.T)).astype(np.float32)
            else: self.adj = _gauss_adj(np.loadtxt(ROOT / "data/pemsd7m/PeMSD7_W_228.csv", delimiter=","))
        else:
            with open(adj_path, "rb") as fh:
                self.sensor_ids, self.sensor_index, self.adj = pickle.load(fh, encoding="latin1")  # sensor proximity graph
        self._starts = {k: np.arange(a, b - IN_LEN - OUT_LEN + 1) for k, (a, b) in self.bounds.items()}

    def n_windows(self, split):
        return len(self._starts[split])

    def batch(self, split, client, idx):
        """idx: window numbers inside the split; returns X (B, 12, 2), Y (B, 3), mask (B, 3), Y in original units."""
        s = torch.from_numpy(self._starts[split][idx])
        xi = s[:, None] + torch.arange(IN_LEN)[None]
        yi = s[:, None] + IN_LEN + torch.arange(OUT_LEN)[None]
        x = self.series[xi, client]                              # (B, 12, 2)
        y = self.series[yi, client][..., 0]                      # (B, 3) normalised
        m = self.series[yi, client][..., 1]
        return x, y, m, self.raw[yi, client]
