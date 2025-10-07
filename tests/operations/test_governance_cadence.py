from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Mapping

from src.governance.system_config import SystemConfig
from src.operations.compliance_readiness import (
    ComplianceReadinessComponent,
    ComplianceReadinessSnapshot,
    ComplianceReadinessStatus,
)
from src.operations.governance_cadence import GovernanceCadenceRunner
from src.operations.governance_reporting import GovernanceReportStatus
from src.operations.regulatory_telemetry import (
    RegulatoryTelemetrySnapshot,
    RegulatoryTelemetryStatus,
)


class _FailingProvider:
    def __call__(self) -> None:  # pragma: no cover - defensive
        raise AssertionError("provider should not be invoked when cadence not due")


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[Mapping[str, object]] = []
        self.running = True

    def is_running(self) -> bool:
        return self.running

    def publish_from_sync(self, event: object) -> None:
        self.events.append(event)  # type: ignore[assignment]


def test_governance_cadence_skips_until_interval_elapsed(tmp_path: Path) -> None:
    path = tmp_path / "governance.json"
    path.write_text(
        """
        {
          "latest": {
            "generated_at": "2024-01-01T00:00:00+00:00"
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    runner = GovernanceCadenceRunner(
        event_bus=_StubEventBus(),
        config_provider=SystemConfig,
        compliance_provider=_FailingProvider(),
        regulatory_provider=_FailingProvider(),
        report_path=path,
        interval=timedelta(days=1),
    )

    result = runner.run(reference=datetime(2024, 1, 1, 12, tzinfo=UTC))

    assert result is None
    assert runner.last_generated_at == datetime(2024, 1, 1, tzinfo=UTC)


def test_governance_cadence_runs_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "governance.json"
    bus = _StubEventBus()

    compliance_snapshot = ComplianceReadinessSnapshot(
        status=ComplianceReadinessStatus.ok,
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
        components=(
            ComplianceReadinessComponent(
                name="kyc_aml",
                status=ComplianceReadinessStatus.ok,
                summary="KYC monitors green",
                metadata={},
            ),
        ),
        metadata={},
    )

    regulatory_snapshot = RegulatoryTelemetrySnapshot(
        generated_at=datetime(2024, 1, 1, 12, tzinfo=UTC),
        status=RegulatoryTelemetryStatus.ok,
        coverage_ratio=1.0,
        signals=(),
        required_domains=(),
        missing_domains=(),
        metadata={},
    )

    audit_payload = {"metadata": {"configured": True, "dialect": "sqlite"}}

    persisted: list[Mapping[str, object]] = []

    def _persist(report, persist_path: Path, history_limit: int) -> None:
        persist_path.write_text("persisted", encoding="utf-8")
        persisted.append({"status": report.status.value, "limit": history_limit})

    runner = GovernanceCadenceRunner(
        event_bus=bus,
        config_provider=SystemConfig,
        compliance_provider=lambda: compliance_snapshot,
        regulatory_provider=lambda: regulatory_snapshot,
        report_path=path,
        interval=timedelta(hours=6),
        history_limit=3,
        strategy_id_provider=lambda: "primary",
        metadata_provider=lambda: {"cadence": "6h"},
        audit_collector=lambda _config, _strategy_id: audit_payload,
        persister=_persist,
    )

    reference = datetime(2024, 1, 2, tzinfo=UTC)
    report = runner.run(reference=reference)

    assert report is not None
    assert report.generated_at == reference
    assert report.metadata["cadence_interval_seconds"] == 6 * 3600
    assert report.metadata["cadence"] == "6h"
    assert report.metadata["strategy_id"] == "primary"
    assert report.status is GovernanceReportStatus.ok
    assert runner.last_generated_at == reference
    assert len(bus.events) == 1
    assert persisted == [{"status": GovernanceReportStatus.ok.value, "limit": 3}]


def test_governance_cadence_force_overrides_interval(tmp_path: Path) -> None:
    path = tmp_path / "governance.json"
    path.write_text(
        """
        {
          "latest": {
            "generated_at": "2024-01-02T00:00:00+00:00"
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    bus = _StubEventBus()

    compliance_snapshot = ComplianceReadinessSnapshot(
        status=ComplianceReadinessStatus.ok,
        generated_at=datetime(2024, 1, 2, tzinfo=UTC),
        components=(
            ComplianceReadinessComponent(
                name="kyc_aml",
                status=ComplianceReadinessStatus.ok,
                summary="Stable",
                metadata={},
            ),
        ),
        metadata={},
    )

    regulatory_snapshot = RegulatoryTelemetrySnapshot(
        generated_at=datetime(2024, 1, 2, 6, tzinfo=UTC),
        status=RegulatoryTelemetryStatus.ok,
        coverage_ratio=1.0,
        signals=(),
        required_domains=(),
        missing_domains=(),
        metadata={},
    )

    runner = GovernanceCadenceRunner(
        event_bus=bus,
        config_provider=SystemConfig,
        compliance_provider=lambda: compliance_snapshot,
        regulatory_provider=lambda: regulatory_snapshot,
        report_path=path,
        interval=timedelta(days=7),
        audit_collector=lambda _config, _strategy_id: {
            "metadata": {"configured": True, "dialect": "sqlite"}
        },
        persister=lambda _report, _path, _limit: None,
    )

    reference = datetime(2024, 1, 2, 6, tzinfo=UTC)
    report = runner.run(reference=reference, force=True)

    assert report is not None
    assert runner.last_generated_at == reference
    assert len(bus.events) == 1
