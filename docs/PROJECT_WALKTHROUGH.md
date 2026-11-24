Run and read the project

This guide shows how to start NeuralEdge-IDS and read its results.

Start with Docker

```bash
docker compose up --build
```

Open http://127.0.0.1:8094.

Run it on your computer

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
make run
make plot
```

The plot command saves a chart in the output folder.

Main files

`ids/dataset.py` makes sample traffic for each device type.

`ids/model.py` has the small classifier.

`ids/centralized.py` trains one model on all data.

`ids/federated.py` trains on separate device data and combines model updates.

`ids/evaluate.py` calculates the scores.

`experiments/run.py` runs both methods and prints the results.

`static/index.html` contains the dashboard.

Read the results

The README shows one run with a fixed seed. Check the detection and false alert rates together. A model can find more attacks but also flag more normal traffic.

The number of values sent is not a count of bytes or all network use. It is not a privacy score.

Limits

This project uses made-up traffic. It does not capture or send live network data, and its results do not predict how the model will work on real devices.
