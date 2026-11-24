Learning guide

This guide explains a few terms used in NeuralEdge-IDS.

Two ways to train

In central training, one computer gets all the sample data and trains one model.

In federated training, each device trains with its own data. A server then combines the model updates.

Federated training moves model updates instead of all the sample rows. This project counts the numbers sent, but it does not count all network costs.

Changes in normal traffic

Each device type has its own normal traffic. The project calls this data non-IID. It also shifts some normal traffic to test what happens when a device changes over time. This kind of change is called drift.

Reading the scores

Detection rate shows how many sample attacks were found.

False alert rate shows how often normal sample traffic was marked as an attack.

F1 combines precision and recall into one score.

Limits

All traffic is made up. The model is small and the test data includes the same device types as the training data. Federated training alone does not protect private information.
