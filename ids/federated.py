"""Federated training (FedAvg) across devices, with communication accounting.

Each round:
  1. the server broadcasts the current global parameters to every device,
  2. each device runs a few local epochs on its own (non-IID) data,
  3. the server averages the returned parameters, weighted by device sample count.

Raw traffic never leaves a device - only parameter vectors are exchanged. We
account for the communication cost (floats transferred) so it can be compared
against the centralized baseline's cost of shipping raw data to one place.

NB: federated learning here is studied as an *experiment in robustness and
cost*, not treated as a privacy guarantee on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ids.dataset import AGREED_SCALER_MEAN, AGREED_SCALER_STD, Dataset
from ids.model import LogisticIDS, Scaler


@dataclass
class FedResult:
    model: LogisticIDS
    rounds: int
    floats_transferred: int      # communication cost proxy
    history: List[float]         # mean local training loss per round (optional signal)


def agreed_scaler() -> Scaler:
    """The pre-agreed normalization shared with every device in advance.

    No raw feature rows are pooled to build this -- it comes from published domain
    ranges (`AGREED_SCALER_MEAN/STD`). That is the honest choice for a federated
    setup: the whole point is that raw data never leaves a device, so the scaler
    cannot be fit by gathering everyone's rows on a server.
    """
    return Scaler(AGREED_SCALER_MEAN.copy(), AGREED_SCALER_STD.copy())


def train_federated(
    shards: Dict[str, Dataset],
    rounds: int = 25,
    local_epochs: int = 20,
    lr: float = 0.3,
) -> FedResult:
    n_features = next(iter(shards.values())).X.shape[1]
    scaler = agreed_scaler()

    global_params = np.zeros(n_features + 1)
    sizes = {dev: len(s) for dev, s in shards.items()}
    total = sum(sizes.values())
    floats = 0
    param_len = n_features + 1

    for _ in range(rounds):
        updates = []
        weights = []
        for dev, shard in shards.items():
            floats += param_len            # server -> client broadcast
            client = LogisticIDS(n_features)
            client.set_params(global_params)
            new_params = client.local_update(shard.X, shard.y, scaler,
                                              epochs=local_epochs, lr=lr)
            floats += param_len            # client -> server upload
            updates.append(new_params)
            weights.append(sizes[dev])
        w = np.array(weights, dtype=float) / total
        global_params = np.average(np.vstack(updates), axis=0, weights=w)

    model = LogisticIDS(n_features)
    model.set_params(global_params)
    model.attach_scaler(scaler)
    return FedResult(model=model, rounds=rounds, floats_transferred=floats, history=[])


def centralized_comm_cost(train: Dataset) -> int:
    """Floats a centralized approach must move: every raw sample to the server."""
    return int(train.X.size)
