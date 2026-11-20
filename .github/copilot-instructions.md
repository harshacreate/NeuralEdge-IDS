# GitHub Copilot instructions

## Project scope
NeuralEdge-IDS compares centralized and federated models on generated IoT traffic. Results are specific to the checked-in generator, split, seed, and metrics.

- Keep experiments reproducible with explicit seeds and documented train/test splits.
- Never ingest live device traffic or add real device identifiers or user data.
- Do not describe federated training as a privacy guarantee; model updates can still leak information.
- Report detection rate, false-alert rate, and F1 together. Keep normal-traffic and shifted-traffic results separate.
- Do not silently change the generator, features, or split while retaining old result claims.

## Changes and validation
- Read the data generator, training/evaluation code, README limitations, and tests before editing.
- Add deterministic tests for data shapes, split boundaries, and metric calculations.
- Use `python -m pytest -q`, `make test`, and documented evaluation/plot commands.
- Keep generated datasets and plots reproducible and avoid committing large or externally sourced data.
