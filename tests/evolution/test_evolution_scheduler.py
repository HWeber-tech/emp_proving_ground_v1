from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.evolution.engine.scheduler import (
    EvolutionScheduler,
    EvolutionSchedulerConfig,
    EvolutionTelemetrySample,
)


def _moment(minutes: float) -> datetime:
    return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes)


def _build_scheduler(**kwargs: object) -> EvolutionScheduler:
    config = EvolutionSchedulerConfig(**kwargs)
    return EvolutionScheduler(config=config)


def test_scheduler_triggers_on_negative_pnl() -> None:
    scheduler = _build_scheduler(
        window=timedelta(minutes=30),
        min_interval=timedelta(minutes=10),
        min_samples=3,
        pnl_floor=-20.0,
    )
    for index, pnl in enumerate((-10.0, -15.0, -25.0)):
        scheduler.record_sample(
            EvolutionTelemetrySample(
                timestamp=_moment(index),
                pnl=pnl,
                drawdown=0.02,
                latency_ms=120.0,
            )
        )

    decision = scheduler.evaluate(now=_moment(12))

    assert decision.triggered is True
    assert "pnl_breach" in decision.reasons


def test_scheduler_respects_cooldown_after_trigger() -> None:
    scheduler = _build_scheduler(
        window=timedelta(hours=1),
        min_interval=timedelta(minutes=20),
        min_samples=3,
        pnl_floor=-5.0,
    )
    for index, pnl in enumerate((-1.0, -2.0, -6.0)):
        scheduler.record_sample(
            EvolutionTelemetrySample(
                timestamp=_moment(index),
                pnl=pnl,
                drawdown=0.03,
                latency_ms=110.0,
            )
        )

    first_decision = scheduler.evaluate(now=_moment(25))
    assert first_decision.triggered is True

    cooldown_decision = scheduler.evaluate(now=_moment(35))
    assert cooldown_decision.triggered is False
    assert "cooldown_active" in cooldown_decision.reasons
    assert "pnl_breach" in cooldown_decision.reasons


def test_scheduler_triggers_on_drawdown_breach() -> None:
    scheduler = _build_scheduler(
        window=timedelta(minutes=20),
        min_interval=timedelta(minutes=5),
        min_samples=3,
        drawdown_ceiling=0.05,
    )
    for index, drawdown in enumerate((0.01, 0.02, 0.06)):
        scheduler.record_sample(
            EvolutionTelemetrySample(
                timestamp=_moment(index),
                pnl=5.0,
                drawdown=drawdown,
                latency_ms=80.0,
            )
        )

    decision = scheduler.evaluate(now=_moment(10))
    assert decision.triggered is True
    assert "drawdown_breach" in decision.reasons


def test_scheduler_triggers_on_latency_percentile() -> None:
    scheduler = _build_scheduler(
        window=timedelta(minutes=15),
        min_interval=timedelta(minutes=5),
        min_samples=4,
        latency_ceiling_ms=100.0,
        latency_percentile=0.9,
        pnl_floor=-100.0,
    )
    latencies = (50.0, 60.0, 65.0, 150.0)
    for index, latency in enumerate(latencies):
        scheduler.record_sample(
            EvolutionTelemetrySample(
                timestamp=_moment(index),
                pnl=10.0,
                drawdown=0.01,
                latency_ms=latency,
            )
        )

    decision = scheduler.evaluate(now=_moment(6))
    assert decision.triggered is True
    assert "latency_breach" in decision.reasons


def test_scheduler_requires_min_samples() -> None:
    scheduler = _build_scheduler(
        window=timedelta(minutes=15),
        min_interval=timedelta(minutes=5),
        min_samples=4,
        pnl_floor=-1.0,
    )
    for index in range(2):
        scheduler.record_sample(
            EvolutionTelemetrySample(
                timestamp=_moment(index),
                pnl=1.0,
                drawdown=0.01,
                latency_ms=10.0,
            )
        )

    decision = scheduler.evaluate(now=_moment(5))
    assert decision.triggered is False
    assert "insufficient_samples" in decision.reasons


def test_scheduler_triggers_on_stagnation() -> None:
    scheduler = _build_scheduler(
        window=timedelta(hours=1),
        min_interval=timedelta(minutes=5),
        min_samples=1,
        pnl_floor=-100.0,
        stagnation_timeout=timedelta(minutes=15),
    )
    scheduler.record_sample(
        EvolutionTelemetrySample(
            timestamp=_moment(0),
            pnl=5.0,
            drawdown=0.01,
            latency_ms=50.0,
        )
    )

    decision = scheduler.evaluate(now=_moment(20))
    assert decision.triggered is True
    assert "stagnation_detected" in decision.reasons


def test_scheduler_state_serialises_last_decision() -> None:
    scheduler = _build_scheduler(
        window=timedelta(minutes=20),
        min_interval=timedelta(minutes=5),
        min_samples=2,
        pnl_floor=-5.0,
    )
    with pytest.raises(ValueError):
        EvolutionSchedulerConfig(window=timedelta(minutes=0))

    for index in range(2):
        scheduler.record_sample(
            EvolutionTelemetrySample(
                timestamp=_moment(index),
                pnl=-10.0,
                drawdown=0.02,
                latency_ms=70.0,
            )
        )

    decision = scheduler.evaluate(now=_moment(6))
    snapshot = scheduler.state().as_dict()

    assert decision.triggered is True
    assert snapshot["last_decision"]["triggered"] is True
    assert snapshot["sample_count"] >= 0
