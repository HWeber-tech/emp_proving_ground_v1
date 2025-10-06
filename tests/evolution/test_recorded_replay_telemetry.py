from datetime import datetime, timedelta, timezone

import pytest

from src.evolution.evaluation.publisher import (
    EVENT_TYPE_RECORDED_REPLAY,
    build_recorded_replay_event,
    format_recorded_replay_markdown,
    publish_recorded_replay_snapshot,
)
from src.evolution.evaluation.recorded_replay import (
    RecordedEvaluationResult,
    RecordedSensoryEvaluator,
    RecordedSensorySnapshot,
    RecordedTrade,
)
from src.evolution.evaluation.telemetry import (
    RecordedReplayTelemetrySnapshot,
    summarise_recorded_replay,
)
from src.sensory.signals import IntegratedSignal
from src.operations.event_bus_failover import EventPublishError


def _snapshot(ts: datetime, price: float, strength: float, confidence: float) -> RecordedSensorySnapshot:
    payload = {
        "generated_at": ts,
        "integrated_signal": IntegratedSignal(
            direction=1.0 if strength >= 0 else -1.0,
            strength=strength,
            confidence=confidence,
            contributing=["WHY", "WHAT", "WHEN", "HOW", "ANOMALY"],
        ),
        "dimensions": {
            "WHAT": {
                "signal": strength,
                "confidence": confidence,
                "value": {"last_close": price},
                "metadata": {"last_close": price},
            }
        },
    }
    return RecordedSensorySnapshot.from_snapshot(payload)


def _build_summary() -> RecordedReplayTelemetrySnapshot:
    start = datetime.now(timezone.utc) - timedelta(minutes=30)
    price = 100.0
    frames: list[RecordedSensorySnapshot] = []
    for idx in range(20):
        ts = start + timedelta(minutes=idx)
        price += 0.45
        strength = 0.55 if idx > 2 else 0.1
        confidence = 0.8
        frames.append(_snapshot(ts, price, strength, confidence))

    genome_params = {
        "entry_threshold": 0.35,
        "exit_threshold": 0.15,
        "risk_fraction": 0.4,
        "min_confidence": 0.7,
        "cooldown_steps": 1,
    }

    evaluator = RecordedSensoryEvaluator(frames)
    result = evaluator.evaluate(genome_params)
    summary = summarise_recorded_replay(
        result,
        genome_id="genome-for-publish",
        dataset_id="timescale-session-42",
        evaluation_id="replay-20240101",
        parameters=genome_params,
    )
    return summary


def test_recorded_replay_summary_builds_lineage_and_markdown() -> None:
    start = datetime.now(timezone.utc) - timedelta(minutes=40)
    price = 100.0
    snapshots: list[RecordedSensorySnapshot] = []
    for idx in range(30):
        ts = start + timedelta(minutes=idx)
        price += 0.45
        strength = 0.6 if idx > 3 else 0.2
        confidence = 0.78
        snapshots.append(_snapshot(ts, price, strength, confidence))

    evaluator = RecordedSensoryEvaluator(snapshots)
    genome_params = {
        "entry_threshold": 0.4,
        "exit_threshold": 0.2,
        "risk_fraction": 0.4,
        "min_confidence": 0.7,
        "cooldown_steps": 1,
    }
    result = evaluator.evaluate(genome_params)

    summary = summarise_recorded_replay(
        result,
        genome_id="genome-telemetry",
        dataset_id="timescale-session-42",
        evaluation_id="replay-20240101",
        parameters=genome_params,
        metadata={"run": "integration"},
    )

    data = summary.as_dict()
    assert data["status"] in {"normal", "warn", "alert"}
    assert data["lineage"]["metadata"]["genome_id"] == "genome-telemetry"
    assert data["lineage"]["inputs"]["dataset_id"] == "timescale-session-42"
    assert pytest.approx(result.total_return, rel=1e-6) == data["metrics"]["total_return"]
    assert "Total return" in summary.to_markdown()
    assert data["trade_summary"]["profit_factor"] >= 0
    assert "best_trade" in data["trade_summary"]


def test_recorded_replay_summary_flags_alert_on_large_drawdown() -> None:
    now = datetime.now(timezone.utc)
    trade = RecordedTrade(
        opened_at=now - timedelta(minutes=10),
        closed_at=now,
        direction=-1,
        entry_price=100.0,
        exit_price=80.0,
        return_pct=-0.2,
        confidence_open=0.9,
        confidence_close=0.2,
        strength_open=0.5,
        strength_close=-0.3,
    )
    result = RecordedEvaluationResult(
        equity_curve=(1.0, 0.8),
        total_return=-0.2,
        max_drawdown=0.4,
        sharpe_ratio=-1.1,
        volatility=0.3,
        win_rate=0.0,
        trades=1,
        wins=0,
        losses=1,
        trade_log=(trade,),
        max_consecutive_losses=1,
        average_trade_duration_minutes=10.0,
    )

    summary = summarise_recorded_replay(
        result,
        genome_id="stress-genome",
        warn_drawdown=0.1,
        alert_drawdown=0.3,
    )

    assert summary.status == "alert"
    assert summary.trade_summary["worst_trade"]["return_pct"] == pytest.approx(-0.2)
    assert summary.trade_summary["exposure_minutes"] == pytest.approx(10.0)


def test_build_recorded_replay_event_wraps_snapshot_payload() -> None:
    summary = _build_summary()

    event = build_recorded_replay_event(summary)

    assert event.type == EVENT_TYPE_RECORDED_REPLAY
    assert event.source
    assert event.payload["lineage"]["metadata"]["genome_id"] == "genome-for-publish"


def test_publish_recorded_replay_snapshot_uses_event_bus() -> None:
    summary = _build_summary()

    class StubBus:
        def __init__(self) -> None:
            self.events: list[object] = []

        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: object) -> int:  # pragma: no cover - trivial
            self.events.append(event)
            return 1

    bus = StubBus()
    publish_recorded_replay_snapshot(bus, summary)

    assert bus.events, "expected recorded replay telemetry to publish via stub bus"
    published = bus.events[0]
    assert getattr(published, "payload")["lineage"]["metadata"]["genome_id"] == "genome-for-publish"


def test_publish_recorded_replay_snapshot_falls_back_on_runtime_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    summary = _build_summary()

    class PrimaryBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: object) -> None:
            raise RuntimeError("loop stopped")

    published: list[tuple[str, dict[str, object], str | None]] = []

    class GlobalBus:
        def publish_sync(
            self, event_type: str, payload: dict[str, object], source: str | None = None
        ) -> None:
            published.append((event_type, payload, source))

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus", lambda: GlobalBus()
    )

    caplog.set_level("WARNING", logger="src.evolution.evaluation.publisher")
    publish_recorded_replay_snapshot(PrimaryBus(), summary)

    assert published, "expected fallback to publish via global bus"
    assert any("falling back to global bus" in message for message in caplog.messages)


def test_publish_recorded_replay_snapshot_raises_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = _build_summary()

    class PrimaryBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: object) -> None:
            raise ValueError("boom")

    def failing_global_bus() -> object:
        raise AssertionError("global bus should not be used when runtime raises non-runtime error")

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus", failing_global_bus
    )

    with pytest.raises(EventPublishError) as excinfo:
        publish_recorded_replay_snapshot(PrimaryBus(), summary)

    assert excinfo.value.stage == "runtime"
    assert excinfo.value.event_type == EVENT_TYPE_RECORDED_REPLAY


def test_format_recorded_replay_markdown_delegates_to_snapshot() -> None:
    summary = _build_summary()

    markdown = format_recorded_replay_markdown(summary)

    assert "Total return" in markdown
