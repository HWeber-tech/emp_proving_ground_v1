from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.operations.regulatory_telemetry import (
    RegulatoryTelemetrySignal,
    RegulatoryTelemetrySnapshot,
    RegulatoryTelemetryStatus,
    evaluate_regulatory_telemetry,
    format_regulatory_markdown,
    publish_regulatory_telemetry,
)


class DummyEventBus:
    def __init__(self) -> None:
        self.events: list[object] = []
        self._running = True

    def publish_from_sync(self, event: object) -> int:
        self.events.append(event)
        return 1

    def is_running(self) -> bool:
        return self._running


def _fresh_timestamp(minutes: int = 0) -> str:
    return (datetime.now(tz=UTC) - timedelta(minutes=minutes)).isoformat()


def test_evaluate_regulatory_telemetry_marks_missing_domains() -> None:
    snapshot = evaluate_regulatory_telemetry(
        signals=[
            {
                "name": "trade_compliance",
                "status": "ok",
                "summary": "All checks passing",
                "observed_at": _fresh_timestamp(),
            },
            {
                "name": "kyc_aml",
                "status": "warn",
                "summary": "Backlog present",
                "observed_at": _fresh_timestamp(),
            },
        ],
        required_domains=("trade_compliance", "kyc_aml", "trade_reporting"),
    )

    assert snapshot.status == RegulatoryTelemetryStatus.fail
    assert set(snapshot.missing_domains) == {"trade_reporting"}
    placeholders = [
        signal
        for signal in snapshot.signals
        if signal.metadata.get("reason") == "telemetry_missing"
    ]
    assert placeholders and placeholders[0].name == "trade_reporting"


def test_evaluate_regulatory_telemetry_flags_stale_signals() -> None:
    snapshot = evaluate_regulatory_telemetry(
        signals=[
            {
                "name": "trade_reporting",
                "status": "ok",
                "summary": "Reports delivered",
                "observed_at": _fresh_timestamp(minutes=180),
            }
        ],
        required_domains=("trade_reporting",),
        stale_after=timedelta(minutes=30),
    )

    signal = snapshot.signals[0]
    assert signal.status == RegulatoryTelemetryStatus.warn
    assert signal.metadata.get("stale") is True


def test_evaluate_regulatory_telemetry_handles_existing_signal_instances() -> None:
    signal = RegulatoryTelemetrySignal(
        name="surveillance",
        status=RegulatoryTelemetryStatus.fail,
        summary="Alert backlog",
        observed_at=datetime.now(tz=UTC),
        metadata={"violations": 3},
    )

    snapshot = evaluate_regulatory_telemetry(
        signals=[signal],
        required_domains=("surveillance",),
    )

    assert isinstance(snapshot, RegulatoryTelemetrySnapshot)
    assert snapshot.status == RegulatoryTelemetryStatus.fail
    assert snapshot.coverage_ratio == pytest.approx(1.0)


def test_publish_regulatory_telemetry_uses_event_bus() -> None:
    snapshot = evaluate_regulatory_telemetry(
        signals=[
            {
                "name": "trade_compliance",
                "status": "ok",
                "summary": "All good",
                "observed_at": _fresh_timestamp(),
            }
        ],
        required_domains=("trade_compliance",),
    )

    bus = DummyEventBus()
    publish_regulatory_telemetry(bus, snapshot)

    assert bus.events, "expected snapshot to be published"
    payload = bus.events[0].payload
    assert payload["status"] == RegulatoryTelemetryStatus.ok.value


def test_publish_regulatory_telemetry_falls_back_to_global_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = evaluate_regulatory_telemetry(
        signals=[
            {
                "name": "kyc_aml",
                "status": "warn",
                "summary": "Backlog present",
                "observed_at": _fresh_timestamp(),
            }
        ],
        required_domains=("kyc_aml",),
    )

    class FailingEventBus(DummyEventBus):
        def publish_from_sync(self, event: object) -> int:
            raise RuntimeError("runtime bus failure")

    published: list[tuple[str, dict[str, object], str | None]] = []

    class DummyTopicBus:
        def publish_sync(
            self, topic: str, payload: dict[str, object], *, source: str | None = None
        ) -> None:
            published.append((topic, payload, source))

    monkeypatch.setattr(
        "src.operations.event_bus_failover.get_global_bus",
        lambda: DummyTopicBus(),
    )

    bus = FailingEventBus()
    publish_regulatory_telemetry(bus, snapshot)

    assert published, "expected snapshot to be published via global bus"
    topic, payload, source = published[0]
    assert topic == "telemetry.compliance.regulatory"
    assert source == "regulatory_telemetry"
    assert payload["status"] == RegulatoryTelemetryStatus.warn.value


def test_format_regulatory_markdown_renders_table() -> None:
    snapshot = evaluate_regulatory_telemetry(
        signals=[
            {
                "name": "trade_compliance",
                "status": "ok",
                "summary": "policy green",
                "observed_at": _fresh_timestamp(),
            },
            {
                "name": "trade_reporting",
                "status": "fail",
                "summary": "reports missing",
                "observed_at": _fresh_timestamp(),
            },
        ],
        required_domains=("trade_compliance", "trade_reporting"),
        metadata={"cadence": "daily"},
    )

    markdown = format_regulatory_markdown(snapshot)

    assert "| Domain | Status | Summary |" in markdown
    assert "trade_reporting" in markdown
    assert "Coverage" in markdown
    assert "cadence" in markdown
