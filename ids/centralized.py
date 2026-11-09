"""Centralized baseline: pool every device's data and train one model."""

from __future__ import annotations

from ids.dataset import Dataset
from ids.model import LogisticIDS, Scaler


def train_centralized(train: Dataset, epochs: int = 400, lr: float = 0.3) -> LogisticIDS:
    scaler = Scaler.fit(train.X)
    model = LogisticIDS(train.X.shape[1])
    model.fit(train.X, train.y, epochs=epochs, lr=lr, scaler=scaler)
    return model
