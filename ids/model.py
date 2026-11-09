"""A small logistic-regression detector implemented in numpy.

Kept deliberately simple so that (a) it trains in milliseconds and (b) model
*parameters are a flat vector*, which makes federated averaging a one-liner.
Standardization statistics travel with the model so a device can score traffic
without re-fitting the scaler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean, std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def _linear(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # numpy 2.x on some macOS BLAS backends raises spurious FP-exception
    # warnings from matmul (e.g. "divide by zero" even when a weight is 0).
    # These are hardware-flag artifacts, not real numeric errors, so the
    # matmul is evaluated with FP warnings suppressed.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return X @ w + b


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticIDS:
    def __init__(self, n_features: int, l2: float = 1e-3):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.l2 = l2
        self.scaler: Optional[Scaler] = None

    # --- parameter vector helpers (used by FedAvg) ---
    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.w, [self.b]])

    def set_params(self, vec: np.ndarray) -> None:
        self.w = vec[:-1].copy()
        self.b = float(vec[-1])

    def attach_scaler(self, scaler: Scaler) -> None:
        self.scaler = scaler

    # --- training ---
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.2,
            scaler: Optional[Scaler] = None) -> "LogisticIDS":
        self.scaler = scaler or Scaler.fit(X)
        Xs = self.scaler.transform(X)
        n = len(y)
        for _ in range(epochs):
            p = _sigmoid(_linear(Xs, self.w, self.b))
            err = p - y
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= lr * grad_w
            self.b -= lr * grad_b
        return self

    def local_update(self, X: np.ndarray, y: np.ndarray, scaler: Scaler,
                     epochs: int = 30, lr: float = 0.2) -> np.ndarray:
        """One client's local training from the current params; returns new params."""
        self.scaler = scaler
        Xs = scaler.transform(X)
        n = len(y)
        for _ in range(epochs):
            p = _sigmoid(_linear(Xs, self.w, self.b))
            err = p - y
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                grad_w = Xs.T @ err / n + self.l2 * self.w
            self.w -= lr * grad_w
            self.b -= lr * err.mean()
        return self.params

    # --- inference ---
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler is not None, "model has no scaler"
        Xs = self.scaler.transform(X)
        return _sigmoid(_linear(Xs, self.w, self.b))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
