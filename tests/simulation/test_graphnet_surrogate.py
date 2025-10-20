from __future__ import annotations

from src.simulation import GraphNetSurrogate, RolloutExample


def _trained_surrogate() -> GraphNetSurrogate:
    adjacency = {"A": ["B"], "B": ["A"]}
    features = {"A": {"bias": 1.0}, "B": {"bias": 0.5}}
    surrogate = GraphNetSurrogate(adjacency, features, max_hops=1)
    surrogate.train(
        [
            RolloutExample("A", 1.0, {"sharpe": 1.0, "turnover": 0.4}),
            RolloutExample("B", 2.0, {"sharpe": 0.8, "turnover": 0.6}),
        ]
    )
    return surrogate


def test_predict_accepts_single_metric_string() -> None:
    surrogate = _trained_surrogate()

    result = surrogate.predict("A", 1.0, metrics="sharpe")

    assert set(result.keys()) == {"sharpe"}


def test_rollout_accepts_single_metric_string() -> None:
    surrogate = _trained_surrogate()

    rollout = surrogate.rollout("A", 1.0, metrics="turnover")

    assert set(rollout.metrics.keys()) == {"turnover"}
