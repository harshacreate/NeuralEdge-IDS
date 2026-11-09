import numpy as np

from ids import evaluate
from ids.centralized import train_centralized
from ids.dataset import (
    DEVICES,
    make_dataset,
    partition_by_device,
    train_test_split,
)
from ids.federated import train_federated
from ids.model import LogisticIDS, Scaler


def test_dataset_shapes_and_balance():
    ds = make_dataset(n_per_device=100, attack_fraction=0.3, seed=0)
    assert ds.X.shape[1] == 6
    assert len(ds) == 100 * len(DEVICES)
    # roughly 30% attacks
    assert 0.25 < ds.y.mean() < 0.35


def test_partition_covers_all_devices():
    ds = make_dataset(n_per_device=100, seed=0)
    shards = partition_by_device(ds)
    assert set(shards) == set(DEVICES)
    assert sum(len(s) for s in shards.values()) == len(ds)


def test_centralized_detects_attacks():
    full = make_dataset(n_per_device=400, seed=0)
    train, test = train_test_split(full, seed=1)
    model = train_centralized(train)
    m = evaluate.score(test.y, model.predict(test.X))
    assert m.detection_rate > 0.8
    assert m.false_alarm_rate < 0.15


def test_federated_is_competitive_in_distribution():
    full = make_dataset(n_per_device=400, seed=0)
    train, test = train_test_split(full, seed=1)
    shards = partition_by_device(train)
    fed = train_federated(shards, rounds=25, local_epochs=20)
    m = evaluate.score(test.y, fed.model.predict(test.X))
    # federated should still be a usable detector on non-IID shards
    assert m.detection_rate > 0.75
    assert fed.floats_transferred > 0


def test_federated_uses_pre_agreed_scaler_not_pooled_data():
    """Federated training must not pool raw rows to fit a scaler; it uses the
    pre-agreed normalization, so the model-updates-only comm cost stays honest."""
    from ids.dataset import AGREED_SCALER_MEAN, AGREED_SCALER_STD
    full = make_dataset(n_per_device=300, seed=0)
    train, _ = train_test_split(full, seed=1)
    shards = partition_by_device(train)
    fed = train_federated(shards, rounds=5, local_epochs=10)
    assert np.allclose(fed.model.scaler.mean, AGREED_SCALER_MEAN)
    assert np.allclose(fed.model.scaler.std, AGREED_SCALER_STD)


def test_param_vector_roundtrip():
    m = LogisticIDS(6)
    v = np.arange(7, dtype=float)
    m.set_params(v)
    assert np.allclose(m.params, v)


def test_metrics_math():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 0, 1])
    s = evaluate.score(y, pred)
    assert s.detection_rate == 0.5     # 1 of 2 attacks caught
    assert s.false_alarm_rate == 0.5   # 1 of 2 benign flagged
