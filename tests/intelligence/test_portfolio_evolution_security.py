import builtins
import logging

import numpy as np
import pytest

from src.intelligence.portfolio_evolution import (
    CorrelationOptimizer,
    DiversificationMaximizer,
    PortfolioStrategy,
    SynergyDetector,
)


@pytest.fixture(name="portfolio_strategies")
def _portfolio_strategies_fixture() -> list[PortfolioStrategy]:
    return [
        PortfolioStrategy(
            strategy_id="s1",
            strategy_type="momentum",
            weight=0.5,
            expected_return=0.1,
            expected_volatility=0.2,
            correlation_vector=[0.1, 0.2, 0.3],
            risk_contribution=0.05,
            performance_metrics={"alpha": 1.0},
        ),
        PortfolioStrategy(
            strategy_id="s2",
            strategy_type="mean_reversion",
            weight=0.5,
            expected_return=0.08,
            expected_volatility=0.15,
            correlation_vector=[0.4, 0.5, 0.6],
            risk_contribution=0.04,
            performance_metrics={"alpha": 0.8},
        ),
    ]


def test_correlation_optimizer_logs_when_optional_dependencies_missing(monkeypatch, caplog):
    real_import = builtins.__import__

    def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sklearn.covariance" and "LedoitWolf" in fromlist:
            raise ImportError("sklearn not installed")
        if name == "sklearn.cluster" and "AgglomerativeClustering" in fromlist:
            raise ImportError("sklearn not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    with caplog.at_level(logging.WARNING):
        optimizer = CorrelationOptimizer()

    assert optimizer.correlation_estimator is None
    assert optimizer.clustering is None
    assert "LedoitWolf estimator unavailable" in caplog.text
    assert "AgglomerativeClustering unavailable" in caplog.text


def test_get_correlation_logs_numeric_failures(monkeypatch, caplog, portfolio_strategies):
    monkeypatch.setattr(
        SynergyDetector,
        "_build_synergy_model",
        lambda self: lambda *args, **kwargs: 0.5,
    )
    detector = SynergyDetector()

    def raising_corrcoef(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(np, "corrcoef", raising_corrcoef)

    with caplog.at_level(logging.WARNING):
        result = detector._get_correlation(portfolio_strategies[0], portfolio_strategies[1])

    assert result == 0.0
    assert "Correlation computation failed" in caplog.text


def test_build_correlation_matrix_logs_numeric_failures(monkeypatch, caplog, portfolio_strategies):
    optimizer = DiversificationMaximizer()

    def raising_corrcoef(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("bad input")

    monkeypatch.setattr(np, "corrcoef", raising_corrcoef)

    with caplog.at_level(logging.WARNING):
        matrix = optimizer._build_correlation_matrix(portfolio_strategies)

    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, np.eye(2))
    assert "Correlation matrix entry failed" in caplog.text


def test_synergy_detector_falls_back_to_stub(monkeypatch, caplog):
    real_import = builtins.__import__

    def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch.nn":
            raise ModuleNotFoundError("torch not available")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    with caplog.at_level(logging.WARNING):
        detector = SynergyDetector()

    assert callable(detector.synergy_model)
    assert detector.synergy_model() == pytest.approx(0.5)
    assert "torch.nn unavailable" in caplog.text
