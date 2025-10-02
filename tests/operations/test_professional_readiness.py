from datetime import UTC, datetime

import pytest

from src.core.event_bus import Event
from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.health import IngestHealthStatus
from src.data_foundation.ingest.recovery import IngestRecoveryRecommendation
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)
from src.operations.backup import BackupReadinessSnapshot, BackupStatus
from src.operations.data_backbone import (
    BackboneComponentSnapshot,
    BackboneStatus,
    DataBackboneReadinessSnapshot,
)

from src.operations.event_bus_failover import EventPublishError
from src.operations.professional_readiness import (
    ProfessionalReadinessStatus,
    evaluate_professional_readiness,
    format_professional_readiness_markdown,
    publish_professional_readiness_snapshot,
)
from src.operations.slo import OperationalSLOSnapshot, ServiceSLO, SLOStatus


def _backbone_snapshot(status: BackboneStatus) -> DataBackboneReadinessSnapshot:
    return DataBackboneReadinessSnapshot(
        status=status,
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
        components=(
            BackboneComponentSnapshot(
                name="plan",
                status=status,
                summary="configured",
                metadata={"dimensions": ["daily_bars"]},
            ),
        ),
        metadata={"plan": {"daily_bars": {"lookback_days": 30}}},
    )


def _backup_snapshot(status: BackupStatus) -> BackupReadinessSnapshot:
    return BackupReadinessSnapshot(
        service="timescale",
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
        status=status,
        latest_backup_at=datetime(2023, 12, 31, tzinfo=UTC),
        next_backup_due_at=datetime(2024, 1, 1, 12, tzinfo=UTC),
        retention_days=7,
        issues=tuple(),
    )


def _slo_snapshot(status: SLOStatus) -> OperationalSLOSnapshot:
    slo = ServiceSLO(
        name="freshness",
        status=status,
        message="within bounds",
        target={"max_age_minutes": 30},
        observed={"max_age_minutes": 25},
    )
    return OperationalSLOSnapshot(
        service="timescale_ingest",
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
        status=status,
        slos=(slo,),
    )


def test_professional_readiness_combines_signals() -> None:
    backbone = _backbone_snapshot(BackboneStatus.ok)
    backup = _backup_snapshot(BackupStatus.warn)
    slo = _slo_snapshot(SLOStatus.met)
    failover = IngestFailoverDecision(
        should_failover=False,
        status=IngestHealthStatus.warn,
        reason=None,
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
        triggered_dimensions=tuple(),
        optional_triggers=("macro_events",),
        planned_dimensions=("daily_bars",),
        metadata={},
    )
    recommendation = IngestRecoveryRecommendation(
        plan=TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=("EURUSD",), lookback_days=5, source="yahoo")
        ),
        reasons={"daily_bars": "missing symbols"},
        missing_symbols={"daily_bars": ("EURUSD",)},
    )

    snapshot = evaluate_professional_readiness(
        backbone_snapshot=backbone,
        backup_snapshot=backup,
        slo_snapshot=slo,
        failover_decision=failover,
        recovery_recommendation=recommendation,
    )

    assert snapshot.status is ProfessionalReadinessStatus.warn
    names = {component.name for component in snapshot.components}
    assert names == {"data_backbone", "backups", "ingest_slos", "failover", "recovery"}


def test_professional_readiness_escalates_on_failover() -> None:
    backbone = _backbone_snapshot(BackboneStatus.warn)
    backup = _backup_snapshot(BackupStatus.ok)
    slo = _slo_snapshot(SLOStatus.met)
    failover = IngestFailoverDecision(
        should_failover=True,
        status=IngestHealthStatus.error,
        reason="daily bars failed",
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
        triggered_dimensions=("daily_bars",),
        optional_triggers=tuple(),
        planned_dimensions=("daily_bars",),
        metadata={},
    )

    snapshot = evaluate_professional_readiness(
        backbone_snapshot=backbone,
        backup_snapshot=backup,
        slo_snapshot=slo,
        failover_decision=failover,
    )

    assert snapshot.status is ProfessionalReadinessStatus.fail
    failover_component = next(
        component for component in snapshot.components if component.name == "failover"
    )
    assert failover_component.summary == "daily bars failed"


def test_professional_readiness_markdown_lists_components() -> None:
    snapshot = evaluate_professional_readiness(
        backbone_snapshot=_backbone_snapshot(BackboneStatus.ok),
        backup_snapshot=_backup_snapshot(BackupStatus.ok),
        slo_snapshot=_slo_snapshot(SLOStatus.met),
    )

    markdown = format_professional_readiness_markdown(snapshot)
    assert "data_backbone" in markdown
    assert "backups" in markdown
    assert "| ingest_slos | OK |" in markdown
    assert snapshot.as_dict()["status"] == "ok"


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def is_running(self) -> bool:  # pragma: no cover - trivial
        return True

    def publish_from_sync(self, event: Event) -> None:
        self.events.append(event)
        return True


def test_publish_professional_readiness_snapshot_uses_event_bus() -> None:
    snapshot = evaluate_professional_readiness(
        backbone_snapshot=_backbone_snapshot(BackboneStatus.ok),
        backup_snapshot=_backup_snapshot(BackupStatus.ok),
        slo_snapshot=_slo_snapshot(SLOStatus.met),
    )

    bus = _StubEventBus()
    publish_professional_readiness_snapshot(bus, snapshot, source="unit_test")

    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.type == "telemetry.operational.readiness"
    assert event.source == "unit_test"
    assert event.payload["status"] == "ok"


def test_publish_professional_readiness_snapshot_falls_back_to_global_bus() -> None:
    snapshot = evaluate_professional_readiness(
        backbone_snapshot=_backbone_snapshot(BackboneStatus.ok),
        backup_snapshot=_backup_snapshot(BackupStatus.ok),
        slo_snapshot=_slo_snapshot(SLOStatus.met),
    )

    class _FailingRuntimeBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: Event) -> None:
            raise RuntimeError("runtime bus unavailable")

    published: list[tuple[str, dict[str, object], str]] = []

    class _StubTopicBus:
        def publish_sync(self, event_type: str, payload: dict[str, object], *, source: str) -> None:
            published.append((event_type, payload, source))

    publish_professional_readiness_snapshot(
        _FailingRuntimeBus(),
        snapshot,
        global_bus_factory=lambda: _StubTopicBus(),
    )

    assert published == [
        (
            "telemetry.operational.readiness",
            snapshot.as_dict(),
            "operations.professional_readiness",
        )
    ]


def test_publish_professional_readiness_snapshot_raises_on_unexpected_error() -> None:
    snapshot = evaluate_professional_readiness(
        backbone_snapshot=_backbone_snapshot(BackboneStatus.ok),
        backup_snapshot=_backup_snapshot(BackupStatus.ok),
        slo_snapshot=_slo_snapshot(SLOStatus.met),
    )

    class _UnexpectedRuntimeBus:
        def is_running(self) -> bool:
            return True

        def publish_from_sync(self, event: Event) -> None:
            raise ValueError("boom")

    class _UnusedTopicBus:
        def publish_sync(self, event_type: str, payload: dict[str, object], *, source: str) -> None:
            raise AssertionError("global bus should not be used for unexpected runtime errors")

    with pytest.raises(EventPublishError):
        publish_professional_readiness_snapshot(
            _UnexpectedRuntimeBus(),
            snapshot,
            global_bus_factory=lambda: _UnusedTopicBus(),
        )
