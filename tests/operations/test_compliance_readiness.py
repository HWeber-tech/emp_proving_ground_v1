from datetime import datetime, timezone

import pytest

from src.operations.event_bus_failover import EventPublishError
from src.operations.compliance_readiness import (
    ComplianceReadinessStatus,
    evaluate_compliance_readiness,
    publish_compliance_readiness,
)


def _trade_summary(status: str, *, failed: bool = False) -> dict[str, object]:
    checks: list[dict[str, object]] = [{"rule_id": "baseline", "passed": True, "severity": "info"}]
    if failed:
        checks.append({"rule_id": "limit", "passed": False, "severity": "critical"})
    return {
        "policy": {"policy_name": "inst-policy"},
        "last_snapshot": {
            "status": status,
            "checks": checks,
        },
        "daily_totals": {"EURUSD": {"notional": 1_000_000, "trades": 12}},
    }


def _kyc_summary(
    status: str,
    *,
    risk: str = "LOW",
    outstanding: int = 0,
    watchlist: int = 0,
    alerts: int = 0,
    open_cases: int = 0,
    escalations: int = 0,
    next_due: datetime | None = None,
) -> dict[str, object]:
    return {
        "last_snapshot": {
            "status": status,
            "risk_rating": risk,
            "outstanding_items": [f"item-{i}" for i in range(outstanding)],
            "watchlist_hits": [f"hit-{i}" for i in range(watchlist)],
            "alerts": [f"alert-{i}" for i in range(alerts)],
            "next_review_due": next_due.isoformat() if next_due else None,
        },
        "open_cases": open_cases,
        "escalations": escalations,
    }


def test_compliance_readiness_flags_trade_failures() -> None:
    snapshot = evaluate_compliance_readiness(trade_summary=_trade_summary("fail", failed=True))

    assert snapshot.status is ComplianceReadinessStatus.fail
    component = next(comp for comp in snapshot.components if comp.name == "trade_compliance")
    assert component.status is ComplianceReadinessStatus.fail
    assert component.metadata["critical_failures"] == 1


def test_compliance_readiness_ok_when_all_surfaces_clear() -> None:
    snapshot = evaluate_compliance_readiness(
        trade_summary=_trade_summary("pass"),
        kyc_summary=_kyc_summary("APPROVED"),
    )

    assert snapshot.status is ComplianceReadinessStatus.ok
    statuses = {component.name: component.status for component in snapshot.components}
    assert statuses["trade_compliance"] is ComplianceReadinessStatus.ok
    assert statuses["kyc_aml"] is ComplianceReadinessStatus.ok


def test_compliance_readiness_warns_on_kyc_outstanding_items() -> None:
    snapshot = evaluate_compliance_readiness(
        kyc_summary=_kyc_summary(
            "REVIEW_REQUIRED",
            outstanding=2,
            open_cases=1,
            next_due=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
    )

    assert snapshot.status is ComplianceReadinessStatus.warn
    kyc_component = next(comp for comp in snapshot.components if comp.name == "kyc_aml")
    assert kyc_component.status is ComplianceReadinessStatus.warn
    assert kyc_component.metadata["outstanding_items"] == 2


class _StubRuntimeBus:
    def __init__(self) -> None:
        self.events: list[object] = []

    def publish_from_sync(self, event: object) -> None:
        self.events.append(event)

    def is_running(self) -> bool:
        return True


class _RaisingRuntimeBus(_StubRuntimeBus):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def publish_from_sync(self, event: object) -> None:  # type: ignore[override]
        raise self._exc


class _StubTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, object, str | None]] = []

    def publish_sync(self, topic: str, payload: object, *, source: str | None = None) -> None:
        self.events.append((topic, payload, source))


def _snapshot() -> object:
    return evaluate_compliance_readiness(
        trade_summary=_trade_summary("pass"),
        kyc_summary=_kyc_summary("APPROVED"),
    )


def test_publish_compliance_readiness_prefers_runtime_bus() -> None:
    runtime_bus = _StubRuntimeBus()
    snapshot = _snapshot()

    publish_compliance_readiness(runtime_bus, snapshot)

    assert runtime_bus.events
    event = runtime_bus.events[-1]
    assert getattr(event, "type", "") == "telemetry.compliance.readiness"


def test_publish_compliance_readiness_falls_back_to_global_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_bus = _RaisingRuntimeBus(RuntimeError("loop stopped"))
    topic_bus = _StubTopicBus()
    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: topic_bus)

    publish_compliance_readiness(runtime_bus, _snapshot())

    assert not runtime_bus.events
    assert topic_bus.events
    topic, payload, source = topic_bus.events[-1]
    assert topic == "telemetry.compliance.readiness"
    assert payload.get("status") == ComplianceReadinessStatus.ok.value
    assert source == "compliance_readiness"


def test_publish_compliance_readiness_raises_on_unexpected_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bus = _RaisingRuntimeBus(ValueError("boom"))
    topic_bus = _StubTopicBus()
    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: topic_bus)

    with pytest.raises(EventPublishError) as exc_info:
        publish_compliance_readiness(runtime_bus, _snapshot())

    assert not runtime_bus.events
    assert not topic_bus.events
    assert exc_info.value.stage == "runtime"


def test_publish_compliance_readiness_raises_when_global_bus_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bus = _StubRuntimeBus()

    class _FailingTopicBus(_StubTopicBus):
        def publish_sync(self, topic: str, payload: object, *, source: str | None = None) -> None:  # type: ignore[override]
            raise RuntimeError("global bus stopped")

    topic_bus = _FailingTopicBus()
    monkeypatch.setattr("src.operations.event_bus_failover.get_global_bus", lambda: topic_bus)
    monkeypatch.setattr(runtime_bus, "is_running", lambda: False)

    with pytest.raises(EventPublishError) as exc_info:
        publish_compliance_readiness(runtime_bus, _snapshot())

    assert not runtime_bus.events
    assert not topic_bus.events
    assert exc_info.value.stage == "global"
