# NeuralEdge-IDS — Federated IoT Intrusion Detection Under Concept Drift

## Run with Docker

```bash
docker compose up --build -d      # then open http://localhost:8094
docker compose down               # stop
```

This repository is standalone; the commands below build and run this project on its own.

A study of how an IoT network intrusion detector behaves when the world doesn't
stay still: when device behaviour **drifts** over time, and when training data is
**split across devices** instead of pooled in one place. It trains a detector two
ways — a single **centralized** model and a **federated** model (FedAvg) — and
reports detection quality, false-alarm rate, and communication cost for each,
in-distribution and under concept drift.

Everything runs on a **synthetic** IoT traffic generator, so the experiment is
fully reproducible offline and needs no sensitive capture data.

## The question it answers

IoT fleets are the hard case for intrusion detection:

- **Non-IID data** — a camera, a thermostat and a smart lock have completely
  different "normal" traffic, so a model trained on one doesn't transfer cleanly.
- **Concept drift** — firmware updates and usage changes shift what "normal"
  looks like, and a stale model starts crying wolf.
- **Privacy / bandwidth** — shipping every device's raw traffic to a central
  trainer is expensive and sensitive.

This project measures those trade-offs instead of assuming them.

## What it compares

| | Centralized | Federated (FedAvg) |
|---|---|---|
| Data movement | raw samples pooled on a server | only model parameters exchanged |
| Training | one model on all data | per-device local training + weighted averaging |
| Reported | detection rate, false-alarm rate, F1 | same, plus communication cost |

Both are evaluated **in-distribution** and on a **drifted** test set where benign
baselines have shifted toward the attack region.

**Honest communication accounting.** For the federated comm cost (model updates
only) to be meaningful, the federated path must not quietly move raw data for
anything else. Feature normalization is therefore a **pre-agreed scaler**
(`AGREED_SCALER_MEAN/STD` in `ids/dataset.py`, published to every device in
advance from domain knowledge) rather than one fit by pooling all devices' rows
on a server. The centralized baseline is still allowed to pool — that's its
nature — and its comm cost reflects shipping the raw samples.

> Federated learning is treated here as an experiment in **robustness and
> communication cost**, not as a privacy guarantee in itself — parameter updates
> can still leak information, which this project does not attempt to prevent.

## Features & attacks (synthetic)

Each record is a window of flow statistics: `packet_rate`, `mean_payload`,
`unique_dst_ports`, `failed_conn_ratio`, `inter_arrival_var`, `outbound_ratio`.
Four device types (camera, thermostat, lock, hub) each have their own benign
baseline. Two attack families are injected: `flood` (volumetric/DDoS-style) and
`scan` (port/host scanning). Class distributions deliberately **overlap**, so
detection is a multi-feature decision rather than a single threshold.

## Layout

```
ids/
  dataset.py      synthetic per-device traffic, non-IID shards, concept drift
  model.py        numpy logistic-regression detector (params = flat vector)
  centralized.py  pooled-data baseline
  federated.py    FedAvg across devices + honest communication accounting
                  (pre-agreed scaler; no raw rows pooled)
  evaluate.py     detection rate, false-alarm rate, precision, F1
experiments/
  run.py          the comparison report (+ optional matplotlib plot)
tests/            pytest over data, models, metrics
```

## Quick start

```bash
make setup     # numpy, matplotlib, pytest
make run       # centralized vs federated comparison report
make plot      # same, plus out/comparison.png
make test      # pytest
```

Example `make run` output:

```
[centralized]
  in-distribution       DR=0.979  FAR=0.000  F1=0.989
  under concept drift   DR=0.993  FAR=0.060  F1=0.931
  communication (floats): 9000  (raw samples shipped to server)
[federated: 25 rounds, non-IID per-device shards]
  in-distribution       DR=0.979  FAR=0.000  F1=0.989
  under concept drift   DR=0.993  FAR=0.067  F1=0.924
  communication (floats): 1400  (only model params exchanged)
federated / centralized comm cost: 0.156x
```

Reading it: both detect attacks well in-distribution; **concept drift raises the
false-alarm rate** for both; the federated model is slightly behind centralized
on the non-IID shards but moves a small fraction of the data.

## Knobs to explore

- `make_dataset(..., drift=...)` — how far the world has moved.
- `attack_fraction` — class balance.
- `train_federated(rounds=, local_epochs=)` — the accuracy/communication trade-off.
- device profiles in `ids/dataset.py` — make the fleet more or less heterogeneous.

## Scope & safety

The detector is trained and evaluated entirely on self-generated synthetic
traffic. No real network is captured, scanned, or attacked.
