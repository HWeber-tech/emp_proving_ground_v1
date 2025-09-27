from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import tools.roadmap.snapshot as roadmap_snapshot


def _status_map() -> dict[str, roadmap_snapshot.InitiativeStatus]:
    return {status.initiative: status for status in roadmap_snapshot.evaluate_portfolio_snapshot()}


def test_data_backbone_marked_ready() -> None:
    statuses = _status_map()
    backbone = statuses["Institutional data backbone"]
    assert backbone.status == "Ready"
    assert not backbone.missing
    assert any(
        evidence.startswith("data_foundation.ingest.timescale_pipeline")
        for evidence in backbone.evidence
    )


def test_data_backbone_guardrails_cover_full_slice() -> None:
    statuses = _status_map()
    backbone = statuses["Institutional data backbone"]
    expected = {
        "data_foundation.ingest.timescale_pipeline.TimescaleBackboneOrchestrator",
        "data_foundation.ingest.configuration.build_institutional_ingest_config",
        "data_foundation.cache.redis_cache.ManagedRedisCache",
        "data_foundation.streaming.kafka_stream.KafkaIngestEventPublisher",
        "operations.data_backbone.evaluate_data_backbone_readiness",
        "data_foundation.batch.spark_export.execute_spark_export_plan",
        "data_foundation.ingest.metrics.summarise_ingest_metrics",
        "data_foundation.ingest.quality.evaluate_ingest_quality",
        "data_foundation.ingest.observability.build_ingest_observability_snapshot",
        "operations.ingest_trends.evaluate_ingest_trends",
        "operations.data_backbone.evaluate_data_backbone_validation",
        "operations.retention.evaluate_data_retention",
        "operations.cache_health.evaluate_cache_health",
        "data_foundation.ingest.recovery.plan_ingest_recovery",
        "data_foundation.ingest.failover.decide_ingest_failover",
        "operations.failover_drill.execute_failover_drill",
        "operations.cross_region_failover.evaluate_cross_region_failover",
        "operations.spark_stress.execute_spark_stress_drill",
        "tools.telemetry.export_data_backbone_snapshots.main",
    }
    assert set(backbone.evidence) == expected


def test_data_backbone_guardrails_include_ingest_telemetry() -> None:
    statuses = _status_map()
    backbone = statuses["Institutional data backbone"]
    guardrails = {
        "data_foundation.ingest.metrics.summarise_ingest_metrics",
        "data_foundation.ingest.quality.evaluate_ingest_quality",
        "data_foundation.ingest.observability.build_ingest_observability_snapshot",
        "operations.ingest_trends.evaluate_ingest_trends",
        "operations.data_backbone.evaluate_data_backbone_validation",
        "operations.retention.evaluate_data_retention",
        "operations.cache_health.evaluate_cache_health",
        "data_foundation.ingest.recovery.plan_ingest_recovery",
        "data_foundation.ingest.failover.decide_ingest_failover",
        "operations.failover_drill.execute_failover_drill",
        "operations.cross_region_failover.evaluate_cross_region_failover",
        "data_foundation.batch.spark_export.execute_spark_export_plan",
        "operations.spark_stress.execute_spark_stress_drill",
        "tools.telemetry.export_data_backbone_snapshots.main",
    }
    assert guardrails.issubset(set(backbone.evidence))


def test_execution_and_compliance_marked_ready() -> None:
    statuses = _status_map()
    ops = statuses["Execution, risk, compliance, ops readiness"]
    assert ops.status == "Ready"
    assert "runtime.fix_pilot.FixIntegrationPilot" in ops.evidence


def test_operational_guardrails_cover_backbone_and_telemetry() -> None:
    statuses = _status_map()
    ops = statuses["Execution, risk, compliance, ops readiness"]
    expected = {
        "runtime.fix_pilot.FixIntegrationPilot",
        "operations.execution.evaluate_execution_readiness",
        "operations.security.evaluate_security_posture",
        "operations.incident_response.evaluate_incident_response",
        "compliance.workflow.evaluate_compliance_workflows",
        "trading.risk.risk_policy.RiskPolicy",
        "operations.professional_readiness.evaluate_professional_readiness",
        "operations.backup.evaluate_backup_readiness",
        "operations.event_bus_health.evaluate_event_bus_health",
        "operations.slo.evaluate_ingest_slos",
        "operations.system_validation.evaluate_system_validation",
        "operations.kafka_readiness.evaluate_kafka_readiness",
        "operations.roi.evaluate_roi_posture",
        "operations.strategy_performance.evaluate_strategy_performance",
        "risk.telemetry.evaluate_risk_posture",
        "trading.risk.policy_telemetry.build_policy_snapshot",
        "operations.data_backbone.evaluate_data_backbone_validation",
        "data_foundation.persist.timescale.TimescaleComplianceJournal",
        "data_foundation.persist.timescale.TimescaleKycJournal",
        "data_foundation.persist.timescale.TimescaleExecutionJournal",
        "tools.telemetry.export_operational_snapshots.main",
        "tools.telemetry.export_risk_compliance_snapshots.main",
    }
    assert set(ops.evidence) == expected


def test_risk_and_compliance_guardrails_surface_audit_evidence() -> None:
    statuses = _status_map()
    ops = statuses["Execution, risk, compliance, ops readiness"]
    required = {
        "data_foundation.persist.timescale.TimescaleComplianceJournal",
        "data_foundation.persist.timescale.TimescaleKycJournal",
        "data_foundation.persist.timescale.TimescaleExecutionJournal",
        "tools.telemetry.export_risk_compliance_snapshots.main",
        "operations.roi.evaluate_roi_posture",
        "operations.strategy_performance.evaluate_strategy_performance",
    }
    assert required.issubset(set(ops.evidence))


def test_sensory_and_evolution_guardrails_cover_organs_and_catalogue() -> None:
    statuses = _status_map()
    sensory = statuses["Sensory cortex & evolution uplift"]
    expected = {
        "sensory.how.how_sensor.HowSensor",
        "sensory.anomaly.anomaly_sensor.AnomalySensor",
        "sensory.when.gamma_exposure.GammaExposureAnalyzer",
        "sensory.why.why_sensor.WhySensor",
        "operations.sensory_drift.evaluate_sensory_drift",
        "evolution.lineage_telemetry.EvolutionLineageSnapshot",
        "genome.catalogue.load_default_catalogue",
        "operations.evolution_experiments.evaluate_evolution_experiments",
        "operations.evolution_tuning.evaluate_evolution_tuning",
    }
    assert set(sensory.evidence) == expected


def test_supporting_modernization_guardrails_cover_ci_telemetry() -> None:
    statuses = _status_map()
    hygiene = statuses["Supporting modernization (formatter, regression, telemetry)"]
    expected = {
        "tools.telemetry.ci_metrics.load_metrics",
        "tests/.telemetry/ci_metrics.json",
        "docs/status/ci_health.md",
    }
    assert set(hygiene.evidence) == expected


def test_markdown_formatter_outputs_table(capsys: pytest.CaptureFixture[str]) -> None:
    markdown = roadmap_snapshot.format_markdown(roadmap_snapshot.evaluate_portfolio_snapshot())
    assert "| Initiative |" in markdown
    assert "Institutional data backbone" in markdown

    roadmap_snapshot.main(["--format", "markdown"])
    captured = capsys.readouterr().out
    assert "Institutional data backbone" in captured
    assert "Ready" in captured


def test_json_format_includes_evidence(capsys: pytest.CaptureFixture[str]) -> None:
    roadmap_snapshot.main(["--format", "json"])
    captured = capsys.readouterr().out
    assert "evidence" in captured
    assert "Institutional data backbone" in captured


def test_cli_recovers_when_src_not_on_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    src_path = Path(__file__).resolve().parents[2] / "src"
    filtered_path = [
        entry for entry in sys.path if Path(entry or ".").resolve() != src_path.resolve()
    ]
    monkeypatch.setattr(sys, "path", filtered_path, raising=False)

    module = importlib.reload(roadmap_snapshot)
    statuses = {status.initiative: status for status in module.evaluate_portfolio_snapshot()}
    assert statuses["Institutional data backbone"].status == "Ready"
    assert statuses["Sensory cortex & evolution uplift"].status == "Ready"
