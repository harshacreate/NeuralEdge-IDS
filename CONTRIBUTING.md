Project notes

Use sample traffic. Keep real device traffic, device IDs and user data out of this project. Record the seed, the parts of each sample the model reads, and how you split the data for training and testing.

If the data or split changes, run the checks again. For both normal and changed traffic, report the detection rate, false alert rate and F1 score. Sharing model updates can still reveal information.

Check the shape of the data, how the split works and how the scores are counted. Use `python -m pytest -q` or `make test`. Follow the steps in the README to make the results and charts again. Keep large data files out of Git.
