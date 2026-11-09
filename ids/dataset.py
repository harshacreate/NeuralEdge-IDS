"""Synthetic IoT network-traffic generator.

Each record is a short window of flow statistics for one device. Devices have
*different benign baselines* (a camera streams; a lock is mostly idle), which
makes the data naturally non-IID across devices -- the realistic hard case for
federated learning.

Attacks are injected as two families:
  * flood  - volumetric DDoS-style bursts (high packet rate, low payload variety)
  * scan   - port/host scanning (many unique destination ports, short flows)

`concept_drift` shifts the benign baseline over time (e.g. a firmware update
changes normal behaviour) so detectors can be tested under changing conditions.

Features (all per-window):
  0 packet_rate          packets/sec
  1 mean_payload         mean payload bytes
  2 unique_dst_ports     distinct destination ports
  3 failed_conn_ratio    fraction of failed connections
  4 inter_arrival_var    variance of packet inter-arrival time
  5 outbound_ratio       outbound/total byte ratio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

FEATURES = [
    "packet_rate",
    "mean_payload",
    "unique_dst_ports",
    "failed_conn_ratio",
    "inter_arrival_var",
    "outbound_ratio",
]
N_FEATURES = len(FEATURES)

# Pre-agreed feature normalization, published to every device IN ADVANCE from
# domain knowledge of plausible IoT flow ranges. Federated training uses this so
# that NO raw feature rows have to be pooled on a server just to fit a scaler --
# which keeps the reported communication cost (model updates only) honest.
# The values are round-number stand-ins for the real pooled statistics.
AGREED_SCALER_MEAN = np.array([90.0, 300.0, 6.5, 0.12, 1.2, 0.6])
AGREED_SCALER_STD = np.array([100.0, 310.0, 8.5, 0.17, 1.2, 0.2])

# Per-device benign baselines: (mean, std) per feature.
# Spreads are wide enough that benign traffic overlaps the attack region in some
# features, so no single threshold separates the classes -- the detector has to
# combine features, and the centralized/federated gap becomes visible.
DEVICE_PROFILES: Dict[str, np.ndarray] = {
    # packet_rate, mean_payload, uniq_ports, failed_ratio, ia_var, outbound
    "camera":     np.array([[120, 45], [900, 180], [4, 2.0], [0.03, 0.02], [0.5, 0.3], [0.80, 0.09]]),
    "thermostat": np.array([[8, 5], [120, 45], [2, 1.2], [0.02, 0.02], [2.0, 0.9], [0.50, 0.15]]),
    "lock":       np.array([[3, 2], [80, 30], [1, 0.9], [0.02, 0.02], [3.0, 1.1], [0.40, 0.15]]),
    "hub":        np.array([[55, 22], [300, 110], [7, 3.0], [0.06, 0.04], [1.0, 0.6], [0.60, 0.13]]),
}
DEVICES = list(DEVICE_PROFILES)


@dataclass
class Dataset:
    X: np.ndarray            # (n, N_FEATURES)
    y: np.ndarray            # (n,) 0=benign 1=attack
    device: np.ndarray       # (n,) device name per row

    def __len__(self) -> int:
        return len(self.y)


def _sample_benign(profile: np.ndarray, n: int, rng: np.random.Generator,
                   drift: float = 0.0) -> np.ndarray:
    means = profile[:, 0].copy()
    stds = profile[:, 1].copy()
    if drift:
        # Concept drift: normal behaviour shifts toward the attack region
        # (busier devices: more packets, more distinct ports, more retries).
        # A detector trained before the drift will raise more false alarms.
        means = means * (1.0 + drift * np.array([1.2, 0.2, 1.5, 2.0, 0.3, 0.1]))
        stds = stds * (1.0 + drift * 0.4)
    x = rng.normal(means, stds, size=(n, N_FEATURES))
    return np.clip(x, 0, None)


def _sample_attack(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    # Attack means sit close enough to benign traffic (esp. the busy camera/hub
    # devices) that the classes overlap; detection is a multi-feature decision,
    # not a single obvious threshold.
    if kind == "flood":
        base = np.array([260, 180, 3, 0.18, 0.15, 0.90])
        std = np.array([110, 70, 2.0, 0.08, 0.10, 0.07])
    elif kind == "scan":
        base = np.array([90, 70, 24, 0.45, 0.35, 0.52])
        std = np.array([45, 35, 12, 0.18, 0.18, 0.15])
    else:
        raise ValueError(kind)
    x = rng.normal(base, std, size=(n, N_FEATURES))
    return np.clip(x, 0, None)


def make_dataset(
    n_per_device: int = 400,
    attack_fraction: float = 0.3,
    drift: float = 0.0,
    seed: int = 0,
) -> Dataset:
    """Build a pooled dataset across all devices."""
    rng = np.random.default_rng(seed)
    X_parts, y_parts, dev_parts = [], [], []
    for dev, profile in DEVICE_PROFILES.items():
        n_attack = int(n_per_device * attack_fraction)
        n_benign = n_per_device - n_attack
        benign = _sample_benign(profile, n_benign, rng, drift=drift)
        # split attacks between flood and scan
        n_flood = n_attack // 2
        attack = np.vstack([
            _sample_attack("flood", n_flood, rng),
            _sample_attack("scan", n_attack - n_flood, rng),
        ])
        X = np.vstack([benign, attack])
        y = np.concatenate([np.zeros(n_benign), np.ones(n_attack)])
        order = rng.permutation(len(y))
        X_parts.append(X[order])
        y_parts.append(y[order])
        dev_parts.append(np.array([dev] * len(y))[order])
    return Dataset(np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(dev_parts))


def partition_by_device(ds: Dataset) -> Dict[str, Dataset]:
    """Split a pooled dataset into one dataset per device (non-IID shards)."""
    out: Dict[str, Dataset] = {}
    for dev in DEVICES:
        mask = ds.device == dev
        out[dev] = Dataset(ds.X[mask], ds.y[mask], ds.device[mask])
    return out


def train_test_split(ds: Dataset, test_frac: float = 0.25, seed: int = 1) -> Tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    cut = int(len(ds) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return (
        Dataset(ds.X[tr], ds.y[tr], ds.device[tr]),
        Dataset(ds.X[te], ds.y[te], ds.device[te]),
    )
