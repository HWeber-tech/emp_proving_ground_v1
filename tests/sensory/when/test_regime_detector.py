from __future__ import annotations

import numpy as np
import pytest

from src.sensory.when import RegimeDetector


def _generate_sample_data(length: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    log_returns = rng.normal(loc=0.0005, scale=0.01, size=length)
    prices = 100 * np.exp(np.cumsum(log_returns))
    volumes = rng.uniform(low=1_000, high=5_000, size=length)
    return prices, volumes


def test_regime_detector_train_and_predict() -> None:
    prices, volumes = _generate_sample_data()
    detector = RegimeDetector()
    detector.train(prices, volumes)

    regime = detector.predict_regime(prices, volumes)
    probabilities = detector.predict_regime_probabilities(prices, volumes)
    stats = detector.get_regime_statistics(prices, volumes)

    assert detector.is_trained is True
    assert 0 <= regime < detector.n_regimes
    assert probabilities.shape == (detector.n_regimes,)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-6)

    assert stats.regime == regime
    assert stats.regime_name
    assert pytest.approx(1.0) == sum(stats.probabilities.values())
    assert stats.confidence == pytest.approx(max(probabilities))


def test_regime_detector_requires_sufficient_data() -> None:
    prices = np.linspace(100, 105, num=40)
    volumes = np.linspace(1_000, 1_200, num=40)
    detector = RegimeDetector()

    with pytest.raises(ValueError):
        detector.train(prices, volumes)
