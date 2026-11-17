# NeuralEdge-IDS — Project Overview

**Federated IoT Intrusion Detection Under Concept Drift**
Project overview

## Project summary

NeuralEdge-IDS studies how a machine-learning intrusion detector behaves when the
world doesn't stay still: when device behaviour drifts over time, and when training
data is split across devices instead of pooled. It trains a detector two ways — a
single centralized model and a federated model (FedAvg) — and reports detection
quality, false alarms and communication cost, in-distribution and under drift. All
on synthetic, non-IID IoT traffic.

## Main features

- **Interactive dashboard (web UI):** sliders for concept drift, attack rate,
  federated rounds and dropped devices; retrain and compare on demand.
- **Centralized vs federated:** detection rate, false-alarm rate and F1 by
  condition, shown as a live chart.
- **Confusion matrices:** TP / FP / FN / TN for both models on the in-distribution
  test set.
- **Per-device results:** detection and false-alarm rate for each device type,
  including the effect of dropping devices from the federation.
- **Communication cost:** federated moves a fraction of the data — shown as a ratio.

## Run with Docker

```sh
docker compose up --build -d      # then open http://localhost:8094
docker compose ps
docker compose down
```

Binds to `127.0.0.1` only, non-root, dropped capabilities.

## Honest caveat

This is a **simulation** of federated training (pre-agreed scaler; no raw-data
pooling). Federated learning alone does not guarantee privacy — model updates can
still leak information, which this project does not attempt to prevent.
