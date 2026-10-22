import torch
from torch import nn


class GRUForecaster(nn.Module):
    """A small model shared by all nodes: a GRU over a window of 12 steps -> a forecast for 3 steps."""
    def __init__(self, hidden=32, out_len=3):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_len)

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])
