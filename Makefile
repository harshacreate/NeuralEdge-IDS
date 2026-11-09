.PHONY: help setup test run plot clean

help:
	@echo "setup  - install dependencies"
	@echo "run    - run the centralized vs federated comparison report"
	@echo "plot   - run the comparison and write out/comparison.png"
	@echo "test   - run the pytest suite"

setup:
	python3 -m pip install -r requirements.txt

run:
	python3 -m experiments.run

plot:
	python3 -m experiments.run --plot

test:
	python3 -m pytest -q

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache out
