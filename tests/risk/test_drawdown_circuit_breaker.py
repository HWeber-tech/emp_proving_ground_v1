from __future__ import annotations

import pytest

from src.risk.manager import CircuitBreakerState, DrawdownCircuitBreaker


def test_drawdown_circuit_breaker_trips_and_recovers() -> None:
    breaker = DrawdownCircuitBreaker(max_drawdown=0.2, cooldown_steps=2, recovery_ratio=0.5)

    event = breaker.record(100_000)
    assert event.state is CircuitBreakerState.NORMAL
    assert event.scaling_factor == pytest.approx(1.0)

    # Breach the configured drawdown – breaker should trip and clamp risk budgets.
    event = breaker.record(75_000)
    assert event.state is CircuitBreakerState.TRIPPED
    assert breaker.should_halt_trading() is True
    assert event.scaling_factor == pytest.approx(0.25)

    # Recovery requires sustained improvements within the recovery ratio window.
    event = breaker.record(90_000)
    assert event.state is CircuitBreakerState.TRIPPED
    assert breaker.should_halt_trading() is True
    assert event.cooldown_remaining == 1

    event = breaker.record(98_000)
    assert event.state is CircuitBreakerState.NORMAL
    assert breaker.should_halt_trading() is False
    assert event.scaling_factor == pytest.approx(0.9, rel=1e-6)


@pytest.mark.parametrize(
    "equity_curve, expected_state, expected_scaling",
    [
        ([100_000, 86_000], CircuitBreakerState.WARNING, 0.3),
        ([100_000, 80_000], CircuitBreakerState.TRIPPED, 0.25),
        ([100_000, 105_000], CircuitBreakerState.NORMAL, 1.0),
    ],
)
def test_drawdown_circuit_breaker_scenarios(
    equity_curve: list[float], expected_state: CircuitBreakerState, expected_scaling: float
) -> None:
    breaker = DrawdownCircuitBreaker(max_drawdown=0.2, warn_threshold=0.7, floor=0.25)

    event = None
    for value in equity_curve:
        event = breaker.record(value)

    assert event is not None
    assert event.state is expected_state
    assert event.scaling_factor == pytest.approx(expected_scaling, rel=1e-6)
