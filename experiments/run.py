"""Compare centralized vs federated IoT intrusion detection, in-distribution and
under concept drift, and report detection quality, false alarms, and comm cost.

Usage:
  python -m experiments.run                 # printed report
  python -m experiments.run --json          # JSON
  python -m experiments.run --plot          # also write plots to out/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from ids import evaluate
from ids.centralized import train_centralized
from ids.dataset import (
    make_dataset,
    partition_by_device,
    train_test_split,
)
from ids.federated import centralized_comm_cost, train_federated

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(seed: int = 0) -> Dict:
    # In-distribution data.
    full = make_dataset(n_per_device=500, attack_fraction=0.3, drift=0.0, seed=seed)
    train, test = train_test_split(full, test_frac=0.25, seed=seed + 1)

    # A drifted test set: benign baselines have shifted (firmware/behaviour change).
    drift_test = make_dataset(n_per_device=250, attack_fraction=0.3, drift=0.6, seed=seed + 99)

    # --- Centralized ---
    cen = train_centralized(train)
    cen_id = evaluate.score(test.y, cen.predict(test.X))
    cen_drift = evaluate.score(drift_test.y, cen.predict(drift_test.X))
    cen_cost = centralized_comm_cost(train)

    # --- Federated (non-IID: one shard per device) ---
    shards = partition_by_device(train)
    fed = train_federated(shards, rounds=25, local_epochs=20)
    fed_id = evaluate.score(test.y, fed.model.predict(test.X))
    fed_drift = evaluate.score(drift_test.y, fed.model.predict(drift_test.X))

    return {
        "centralized": {
            "in_distribution": cen_id.as_dict(),
            "under_drift": cen_drift.as_dict(),
            "comm_floats": cen_cost,
        },
        "federated": {
            "in_distribution": fed_id.as_dict(),
            "under_drift": fed_drift.as_dict(),
            "comm_floats": fed.floats_transferred,
            "rounds": fed.rounds,
        },
        "comm_ratio_fed_over_cen": round(fed.floats_transferred / cen_cost, 3),
    }


def _row(label: str, m: Dict) -> str:
    return "  %-26s DR=%.3f  FAR=%.3f  F1=%.3f" % (
        label, m["detection_rate"], m["false_alarm_rate"], m["f1"])


def _print(result: Dict) -> None:
    print("=" * 64)
    print("IoT Intrusion Detection - centralized vs federated")
    print("  DR = detection rate (recall), FAR = false-alarm rate")
    print("=" * 64)
    print("[centralized]")
    print(_row("in-distribution", result["centralized"]["in_distribution"]))
    print(_row("under concept drift", result["centralized"]["under_drift"]))
    print("  communication (floats): %d  (raw samples shipped to server)"
          % result["centralized"]["comm_floats"])
    print("[federated: %d rounds, non-IID per-device shards]" % result["federated"]["rounds"])
    print(_row("in-distribution", result["federated"]["in_distribution"]))
    print(_row("under concept drift", result["federated"]["under_drift"]))
    print("  communication (floats): %d  (only model params exchanged)"
          % result["federated"]["comm_floats"])
    print("-" * 64)
    print("federated / centralized comm cost: %.3fx"
          % result["comm_ratio_fed_over_cen"])
    print("=" * 64)
    print("Note: federated learning is evaluated here as a robustness/cost")
    print("experiment, not as a standalone privacy guarantee.")


def _plot(result: Dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(ROOT, "out")
    os.makedirs(out_dir, exist_ok=True)

    conditions = ["in_distribution", "under_drift"]
    labels = ["in-distribution", "under drift"]
    metrics = ["detection_rate", "false_alarm_rate", "f1"]
    metric_titles = ["Detection rate", "False-alarm rate", "F1"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = range(len(conditions))
    width = 0.35
    for ax, metric, title in zip(axes, metrics, metric_titles):
        cen = [result["centralized"][c][metric] for c in conditions]
        fed = [result["federated"][c][metric] for c in conditions]
        ax.bar([i - width / 2 for i in x], cen, width, label="centralized")
        ax.bar([i + width / 2 for i in x], fed, width, label="federated")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle("IoT IDS: centralized vs federated, by condition")
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compare centralized vs federated IoT IDS.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    result = run(seed=args.seed)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print(result)
    if args.plot:
        path = _plot(result)
        print("plot written to %s" % path)


if __name__ == "__main__":
    main()
