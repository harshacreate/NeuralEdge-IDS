"""IoT intrusion-detection experiments: centralized vs federated, with drift.

Modules:
  dataset     synthetic per-device IoT traffic with benign/attack labels + drift
  model       a small numpy logistic-regression detector (easy to average)
  centralized pool all device data, train one model
  federated   FedAvg across devices + communication-cost accounting
  evaluate    detection rate, false-alarm rate, F1
"""
