"""Detection metrics.

detection_rate (recall) - fraction of attacks correctly flagged
false_alarm_rate (FPR)  - fraction of benign traffic wrongly flagged
precision, f1           - standard summary metrics
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np


@dataclass
class Metrics:
    detection_rate: float   # recall / TPR
    false_alarm_rate: float # FPR
    precision: float
    f1: float
    n: int

    def as_dict(self) -> Dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Metrics(recall, fpr, precision, f1, len(y_true))
