"""Tests for the evolution safety controller guard rails."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from src.evolution.safety import EvolutionSafetyController, EvolutionSafetyPolicy


@dataclass
class _FakeClock:
    moment: datetime

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, delta: timedelta) -> None:
        self.moment = self.moment + delta


@pytest.fixture
def clock() -> _FakeClock:
    return _FakeClock(moment=datetime(2024, 1, 1, tzinfo=UTC))


def test_allows_execution_when_metrics_within_limits(clock: _FakeClock) -> None:
    controller = EvolutionSafetyController(now=clock)

    decision = controller.evaluate(
        {
            "max_drawdown": 0.05,
            "value_at_risk": 0.03,
            "gross_exposure": 1.2,
            "position_concentration": 0.1,
            "latency_ms_p95": 120.0,
            "slippage_bps": 8.0,
            "data_completeness": 0.99,
        }
    )

    assert decision.allowed is True
    assert decision.violations == ()
    assert decision.cooldown_active is False
    assert decision.lockdown_active is False
    assert decision.reasons == ()


def test_violation_triggers_cooldown(clock: _FakeClock) -> None:
    policy = EvolutionSafetyPolicy(cooldown=timedelta(minutes=10), lockout_threshold=5)
    controller = EvolutionSafetyController(policy=policy, now=clock)

    breach_decision = controller.evaluate({"max_drawdown": 0.5})
    assert breach_decision.allowed is False
    assert breach_decision.cooldown_active is False
    assert breach_decision.reasons == ("metric_violations",)

    # Safe metrics but still inside cooldown window.
    clock.advance(timedelta(minutes=5))
    cooldown_decision = controller.evaluate({"max_drawdown": 0.01, "data_completeness": 0.98})
    assert cooldown_decision.allowed is False
    assert cooldown_decision.cooldown_active is True
    assert cooldown_decision.violations == ()
    assert "cooldown_active" in cooldown_decision.reasons

    # Once cooldown elapsed the decision becomes allowed again.
    clock.advance(timedelta(minutes=6))
    post_cooldown_decision = controller.evaluate({"max_drawdown": 0.01, "data_completeness": 0.98})
    assert post_cooldown_decision.allowed is True


def test_lockdown_engages_after_repeated_breaches(clock: _FakeClock) -> None:
    policy = EvolutionSafetyPolicy(
        cooldown=timedelta(minutes=1),
        lockout_threshold=2,
        lockout_window=timedelta(minutes=30),
    )
    controller = EvolutionSafetyController(policy=policy, now=clock)

    controller.evaluate({"gross_exposure": 3.1})
    clock.advance(timedelta(minutes=5))
    second = controller.evaluate({"value_at_risk": 0.2})
    assert second.allowed is False
    assert second.lockdown_active is True

    clock.advance(timedelta(minutes=5))
    safe = controller.evaluate({"gross_exposure": 1.0, "data_completeness": 0.99})
    assert safe.allowed is False
    assert safe.lockdown_active is True
    assert "lockdown_active" in safe.reasons

    # Once the lockout window clears the controller permits execution again.
    clock.advance(timedelta(minutes=30))
    cleared = controller.evaluate({"gross_exposure": 1.0, "data_completeness": 0.99})
    assert cleared.allowed is True


def test_state_reports_breach_metadata(clock: _FakeClock) -> None:
    controller = EvolutionSafetyController(now=clock)
    controller.evaluate({"max_drawdown": 0.2})

    snapshot = controller.state()
    assert snapshot.last_breach_at is not None
    assert snapshot.cooldown_expires_at is not None
    assert snapshot.breach_count == 1
    assert snapshot.lockdown_active is False
    assert snapshot.as_dict()["breach_count"] == 1
