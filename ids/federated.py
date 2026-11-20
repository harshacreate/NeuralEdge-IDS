"""Train a model across devices with FedAvg. The server sends the model values to each device.
Each device trains on its own traffic and sends its updated values back. The server averages
those values. Raw traffic stays on each device. This code counts how many numeric values are
sent so the cost can be compared with centralized training. Keeping raw traffic local does
not by itself guarantee privacy.
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
    """Use the same scaling values on every device. These values come from published ranges, not
    raw device traffic. This lets devices keep their traffic data on their own computers.
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
    """Count the raw traffic values a centralized setup sends to the server."""
    return int(train.X.size)
