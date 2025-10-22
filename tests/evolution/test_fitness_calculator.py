from __future__ import annotations

import pytest

from src.evolution.fitness import FitnessCalculator, PerformanceRule


@pytest.fixture()
def default_calculator() -> FitnessCalculator:
    return FitnessCalculator()


def test_comprehensive_fitness_balances_components(default_calculator: FitnessCalculator) -> None:
    performance = {"return": 0.12, "sharpe": 1.5, "win_rate": 0.6}
    risk = {"volatility": 0.18, "max_drawdown": 0.12}
    behaviour = {"trade_count": 80, "turnover": 0.8}

    report = default_calculator.evaluate(performance, risk, behaviour)

    assert report.performance_score == pytest.approx(1.2681818, rel=1e-6)
    assert report.risk_penalty == pytest.approx(0.0)
    assert report.behaviour_penalty == pytest.approx(0.0)
    assert report.risk_adjusted_return == pytest.approx(0.1333333, rel=1e-6)
    assert report.total_score == pytest.approx(1.4015151, rel=1e-6)

    detail = report.detail
    assert detail["performance:return"] == pytest.approx(0.6, rel=1e-6)
    assert detail["performance:sharpe"] == pytest.approx(0.45, rel=1e-6)
    assert detail["performance:win_rate"] == pytest.approx(0.2181818, rel=1e-6)
    assert detail["risk_adjusted_return"] == pytest.approx(0.1333333, rel=1e-6)


def test_penalties_reduce_score_when_thresholds_exceeded(default_calculator: FitnessCalculator) -> None:
    performance = {"return": 0.12, "sharpe": 1.5, "win_rate": 0.6}
    risk = {"volatility": 0.4, "max_drawdown": 0.3}
    behaviour = {"trade_count": 200, "turnover": 1.5}

    report = default_calculator.evaluate(performance, risk, behaviour)

    assert report.risk_penalty == pytest.approx(2.5, rel=1e-6)
    assert report.behaviour_penalty == pytest.approx(0.75, rel=1e-6)
    assert report.risk_adjusted_return == pytest.approx(-0.04, rel=1e-6)
    assert report.total_score == pytest.approx(-2.0218182, rel=1e-6)

    detail = report.detail
    assert detail["risk_penalty:volatility"] == pytest.approx(-1.0, rel=1e-6)
    assert detail["risk_penalty:max_drawdown"] == pytest.approx(-1.5, rel=1e-6)
    assert detail["behaviour_penalty:trade_count"] == pytest.approx(-0.5, rel=1e-6)
    assert detail["behaviour_penalty:turnover"] == pytest.approx(-0.25, rel=1e-6)


def test_rules_can_reward_inverse_metrics() -> None:
    calculator = FitnessCalculator(
        performance_rules={
            "stability": PerformanceRule(weight=1.0, target=0.1, higher_is_better=False)
        },
        risk_penalties={},
        behaviour_penalties={},
    )

    report = calculator.evaluate({"stability": 0.05})

    assert report.performance_score == pytest.approx(2.0, rel=1e-6)
    assert report.total_score == pytest.approx(2.0, rel=1e-6)
    assert report.detail["performance:stability"] == pytest.approx(2.0, rel=1e-6)
