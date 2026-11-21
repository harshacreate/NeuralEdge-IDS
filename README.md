NeuralEdge-IDS

NeuralEdge-IDS compares two ways to train a small threat detector using made-up IoT traffic. It checks how the results change when normal traffic changes.

![CI](https://github.com/harshacreate/NeuralEdge-IDS/actions/workflows/ci.yml/badge.svg)

How it works

The project makes sample traffic for cameras, thermostats, locks, and hubs. One model trains on all the data. The other trains on separate device data and combines the results.

The traffic is generated for this project. No live network traffic is used.

Results

These results come from one run with a fixed seed.

| Model and test | Detection rate | False alert rate | F1 |
|---|---:|---:|---:|
| One model, normal test data | 0.979 | 0.000 | 0.989 |
| One model, changed traffic | 0.993 | 0.060 | 0.931 |
| Combined model, normal test data | 0.979 | 0.000 | 0.989 |
| Combined model, changed traffic | 0.993 | 0.067 | 0.924 |

The combined model sent 1,400 numbers during training. The one-model setup counted 9,000. This is only a count of numbers, not total network use. The combined method does not guarantee privacy.

Run it

With Docker:

```bash
docker compose up --build
```

Open http://127.0.0.1:8094.

For local tests and runs:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make test
make run
make plot
```

Limits

The traffic is made up, the feature set is small, and each device type appears in both the training and test data. The results do not show how the model would work on real devices. Shared model updates can also leak information, so this test is not a privacy guarantee.

See the [project overview](docs/PROJECT_OVERVIEW.md), [learning guide](docs/LEARNING_GUIDE.md), and [walkthrough](docs/PROJECT_WALKTHROUGH.md).
