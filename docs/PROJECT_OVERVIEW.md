# NeuralEdge-IDS project overview

NeuralEdge-IDS compares two ways to train a small network threat detector with made-up IoT traffic.

## What it does

The sample data represents cameras, thermostats, locks, and hubs. One model trains on all the data. The other trains on separate device data and combines the results.

The project checks how both methods work on normal sample traffic and after normal traffic changes. It also counts the numbers sent during training.

## Run it

```bash
docker compose up --build
```

Open http://127.0.0.1:8094.

## Limits

The traffic is made up and the feature set is small. All device types appear in both the training and test data. Sending model updates does not guarantee privacy.
