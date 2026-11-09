"""NeuralEdge-IDS — interactive federated-learning dashboard (Flask).

Lets you adjust concept drift, attack rate, federated rounds and device dropout,
then trains a centralized and a federated detector and compares them:
detection/false-alarm/F1, communication cost, confusion matrices, per-device
results and a live chart. Everything runs on synthetic IoT traffic.
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from ids import evaluate  # noqa: E402
from ids.centralized import train_centralized  # noqa: E402
from ids.dataset import DEVICES, make_dataset, partition_by_device, train_test_split  # noqa: E402
from ids.federated import centralized_comm_cost, train_federated  # noqa: E402

STATIC = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=STATIC, static_url_path="/static")


def _confusion(y_true, y_pred):
    y_true = y_true.astype(int); y_pred = y_pred.astype(int)
    return {
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
    }


def _per_device(model, test):
    rows = []
    for dev in DEVICES:
        mask = test.device == dev
        if not mask.any():
            continue
        m = evaluate.score(test.y[mask], model.predict(test.X[mask]))
        rows.append({"device": dev, "n": int(mask.sum()),
                     "detection_rate": m.detection_rate, "false_alarm_rate": m.false_alarm_rate})
    return rows


def _chart(cen, fed):
    conds = ["in_distribution", "under_drift"]; labels = ["in-dist", "drift"]
    metrics = ["detection_rate", "false_alarm_rate", "f1"]; titles = ["Detection", "False alarm", "F1"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    x = range(len(conds)); w = 0.36
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar([i - w/2 for i in x], [cen[c][metric] for c in conds], w, label="central", color="#7c3aed")
        ax.bar([i + w/2 for i in x], [fed[c][metric] for c in conds], w, label="federated", color="#06b6d4")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, transparent=True); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.route("/api/run", methods=["POST"])
def run():
    d = request.get_json(force=True, silent=True) or {}
    drift = float(d.get("drift", 0.6))
    attack = float(d.get("attack", 0.3))
    rounds = int(d.get("rounds", 25))
    dropout = int(d.get("dropout", 0))
    seed = int(d.get("seed", 0))

    full = make_dataset(n_per_device=500, attack_fraction=attack, drift=0.0, seed=seed)
    train, test = train_test_split(full, test_frac=0.25, seed=seed + 1)
    drift_test = make_dataset(n_per_device=250, attack_fraction=attack, drift=drift, seed=seed + 99)

    cen = train_centralized(train)
    cen_metrics = {
        "in_distribution": evaluate.score(test.y, cen.predict(test.X)).as_dict(),
        "under_drift": evaluate.score(drift_test.y, cen.predict(drift_test.X)).as_dict(),
    }

    shards = partition_by_device(train)
    dropped = []
    if dropout > 0:
        keep = list(shards)[:max(1, len(shards) - dropout)]
        dropped = [k for k in shards if k not in keep]
        shards = {k: shards[k] for k in keep}
    fed = train_federated(shards, rounds=rounds, local_epochs=20)
    fed_metrics = {
        "in_distribution": evaluate.score(test.y, fed.model.predict(test.X)).as_dict(),
        "under_drift": evaluate.score(drift_test.y, fed.model.predict(drift_test.X)).as_dict(),
    }

    cen_cost = centralized_comm_cost(train)
    return jsonify({
        "centralized": cen_metrics,
        "federated": fed_metrics,
        "comm": {"centralized": cen_cost, "federated": fed.floats_transferred,
                 "ratio": round(fed.floats_transferred / cen_cost, 3)},
        "confusion": {
            "centralized": _confusion(test.y, cen.predict(test.X)),
            "federated": _confusion(test.y, fed.model.predict(test.X)),
        },
        "per_device": _per_device(fed.model, test),
        "dropped_devices": dropped,
        "rounds": rounds,
        "chart": _chart(cen_metrics, fed_metrics),
    })


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
