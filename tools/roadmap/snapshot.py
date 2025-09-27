"""Generate modernization roadmap snapshots from the repository state."""

from __future__ import annotations

import argparse
import json
import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@dataclass(frozen=True)
class Requirement:
    """Executable requirement used to score roadmap initiatives."""

    label: str
    check: Callable[[Path], bool]

    def evaluate(self, repo_root: Path) -> tuple[bool, str]:
        try:
            ok = bool(self.check(repo_root))
        except Exception as exc:  # pragma: no cover - exercised via tests
            return False, f"{self.label} ({exc.__class__.__name__}: {exc})"
        if not ok:
            return False, self.label
        return True, self.label


@dataclass(frozen=True)
class InitiativeDefinition:
    """Context describing how we evaluate a roadmap initiative."""

    initiative: str
    phase: str
    ready_summary: str
    attention_summary: str
    next_checkpoint: str
    requirements: Sequence[Requirement]

    def evaluate(self, repo_root: Path) -> "InitiativeStatus":
        evidence: list[str] = []
        missing: list[str] = []
        for requirement in self.requirements:
            ok, label = requirement.evaluate(repo_root)
            if ok:
                evidence.append(label)
            else:
                missing.append(label)

        status = "Ready" if not missing else "Attention needed"
        summary = self.ready_summary if not missing else self.attention_summary

        return InitiativeStatus(
            initiative=self.initiative,
            phase=self.phase,
            status=status,
            summary=summary,
            next_checkpoint=self.next_checkpoint,
            evidence=tuple(evidence),
            missing=tuple(missing),
        )


@dataclass(frozen=True)
class InitiativeStatus:
    """Computed modernization status for a roadmap initiative."""

    initiative: str
    phase: str
    status: str
    summary: str
    next_checkpoint: str
    evidence: tuple[str, ...]
    missing: tuple[str, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_module_attr(module_name: str, attribute: str | None = None) -> Requirement:
    label = module_name if attribute is None else f"{module_name}.{attribute}"

    def check(_: Path) -> bool:
        module = importlib.import_module(module_name)
        if attribute is not None:
            getattr(module, attribute)
        return True

    return Requirement(label=label, check=check)


def _require_path(relative_path: str) -> Requirement:
    def check(repo_root: Path) -> bool:
        return (repo_root / relative_path).exists()

    return Requirement(label=relative_path, check=check)


def _initiative_definitions() -> Sequence[InitiativeDefinition]:
    return (
        InitiativeDefinition(
            initiative="Institutional data backbone",
            phase="A",
            ready_summary=(
                "Timescale ingest, Redis caching, Kafka streaming, and Spark exports ship "
                "with readiness telemetry and runtime toggles."
            ),
            attention_summary=(
                "Data backbone scaffolding is incomplete; ensure Timescale ingest, Redis caches, "
                "Kafka publishers, and readiness evaluators are available."
            ),
            next_checkpoint=(
                "Prove cross-region failover and automated scheduler cutover using the failover drills and readiness feeds."
            ),
            requirements=(
                _require_module_attr(
                    "data_foundation.ingest.timescale_pipeline", "TimescaleBackboneOrchestrator"
                ),
                _require_module_attr(
                    "data_foundation.ingest.configuration", "build_institutional_ingest_config"
                ),
                _require_module_attr("data_foundation.cache.redis_cache", "ManagedRedisCache"),
                _require_module_attr(
                    "data_foundation.streaming.kafka_stream", "KafkaIngestEventPublisher"
                ),
                _require_module_attr(
                    "operations.data_backbone", "evaluate_data_backbone_readiness"
                ),
                _require_module_attr(
                    "data_foundation.batch.spark_export", "execute_spark_export_plan"
                ),
                _require_module_attr(
                    "data_foundation.ingest.metrics", "summarise_ingest_metrics"
                ),
                _require_module_attr(
                    "data_foundation.ingest.quality", "evaluate_ingest_quality"
                ),
                _require_module_attr(
                    "data_foundation.ingest.observability",
                    "build_ingest_observability_snapshot",
                ),
                _require_module_attr("operations.ingest_trends", "evaluate_ingest_trends"),
                _require_module_attr(
                    "operations.data_backbone", "evaluate_data_backbone_validation"
                ),
                _require_module_attr(
                    "operations.retention", "evaluate_data_retention"
                ),
                _require_module_attr(
                    "operations.cache_health", "evaluate_cache_health"
                ),
                _require_module_attr(
                    "data_foundation.ingest.recovery", "plan_ingest_recovery"
                ),
                _require_module_attr(
                    "data_foundation.ingest.failover", "decide_ingest_failover"
                ),
                _require_module_attr(
                    "operations.failover_drill", "execute_failover_drill"
                ),
                _require_module_attr(
                    "operations.cross_region_failover", "evaluate_cross_region_failover"
                ),
                _require_module_attr(
                    "operations.spark_stress", "execute_spark_stress_drill"
                ),
                _require_module_attr(
                    "tools.telemetry.export_data_backbone_snapshots", "main"
                ),
            ),
        ),
        InitiativeDefinition(
            initiative="Sensory cortex & evolution uplift",
            phase="B",
            ready_summary=(
                "All five sensory organs ship with drift telemetry while evolution runs use catalogue/lineage exports."
            ),
            attention_summary=(
                "Sensory cortex or evolution catalogue support is missing; ensure HOW/ANOMALY organs and lineage telemetry are shipped."
            ),
            next_checkpoint=(
                "Extend live-paper experiments and automated tuning loops using evolution experiment telemetry."
            ),
            requirements=(
                _require_module_attr("sensory.how.how_sensor", "HowSensor"),
                _require_module_attr("sensory.anomaly.anomaly_sensor", "AnomalySensor"),
                _require_module_attr("sensory.when.gamma_exposure", "GammaExposureAnalyzer"),
                _require_module_attr("sensory.why.why_sensor", "WhySensor"),
                _require_module_attr("operations.sensory_drift", "evaluate_sensory_drift"),
                _require_module_attr("evolution.lineage_telemetry", "EvolutionLineageSnapshot"),
                _require_module_attr("genome.catalogue", "load_default_catalogue"),
                _require_module_attr(
                    "operations.evolution_experiments", "evaluate_evolution_experiments"
                ),
                _require_module_attr(
                    "operations.evolution_tuning", "evaluate_evolution_tuning"
                ),
            ),
        ),
        InitiativeDefinition(
            initiative="Execution, risk, compliance, ops readiness",
            phase="C",
            ready_summary=(
                "FIX pilot, execution readiness, risk policy, and compliance workflows publish telemetry for operators."
            ),
            attention_summary=(
                "Execution, risk, or compliance telemetry is missing; ensure the FIX pilot and readiness evaluators are wired in."
            ),
            next_checkpoint=(
                "Expand broker connectivity with drop-copy ingestion and live reconciliation for the FIX pilot."
            ),
            requirements=(
                _require_module_attr("runtime.fix_pilot", "FixIntegrationPilot"),
                _require_module_attr("operations.execution", "evaluate_execution_readiness"),
                _require_module_attr("operations.security", "evaluate_security_posture"),
                _require_module_attr("operations.incident_response", "evaluate_incident_response"),
                _require_module_attr("compliance.workflow", "evaluate_compliance_workflows"),
                _require_module_attr("trading.risk.risk_policy", "RiskPolicy"),
                _require_module_attr(
                    "operations.professional_readiness", "evaluate_professional_readiness"
                ),
                _require_module_attr("operations.backup", "evaluate_backup_readiness"),
                _require_module_attr("operations.event_bus_health", "evaluate_event_bus_health"),
                _require_module_attr("operations.slo", "evaluate_ingest_slos"),
                _require_module_attr("operations.system_validation", "evaluate_system_validation"),
                _require_module_attr("operations.kafka_readiness", "evaluate_kafka_readiness"),
                _require_module_attr("operations.roi", "evaluate_roi_posture"),
                _require_module_attr(
                    "operations.strategy_performance", "evaluate_strategy_performance"
                ),
                _require_module_attr("risk.telemetry", "evaluate_risk_posture"),
                _require_module_attr(
                    "trading.risk.policy_telemetry", "build_policy_snapshot"
                ),
                _require_module_attr(
                    "operations.data_backbone", "evaluate_data_backbone_validation"
                ),
                _require_module_attr(
                    "data_foundation.persist.timescale", "TimescaleComplianceJournal"
                ),
                _require_module_attr(
                    "data_foundation.persist.timescale", "TimescaleKycJournal"
                ),
                _require_module_attr(
                    "data_foundation.persist.timescale", "TimescaleExecutionJournal"
                ),
                _require_module_attr(
                    "tools.telemetry.export_operational_snapshots", "main"
                ),
                _require_module_attr(
                    "tools.telemetry.export_risk_compliance_snapshots", "main"
                ),
            ),
        ),
        InitiativeDefinition(
            initiative="Supporting modernization (formatter, regression, telemetry)",
            phase="Legacy",
            ready_summary=(
                "Formatter rollout, coverage telemetry, and flake tracking stay green alongside modernization work."
            ),
            attention_summary=(
                "Engineering hygiene telemetry now records repo-wide formatter enforcement; continue feeding metrics into dashboards."
            ),
            next_checkpoint=("Extend formatter and coverage telemetry into deployment dashboards."),
            requirements=(
                _require_module_attr("tools.telemetry.ci_metrics", "load_metrics"),
                _require_path("tests/.telemetry/ci_metrics.json"),
                _require_path("docs/status/ci_health.md"),
            ),
        ),
    )


def evaluate_portfolio_snapshot() -> list[InitiativeStatus]:
    """Return modernization statuses for the roadmap portfolio snapshot."""

    repo_root = _repo_root()
    return [definition.evaluate(repo_root) for definition in _initiative_definitions()]


def format_markdown(statuses: Iterable[InitiativeStatus]) -> str:
    lines = [
        "| Initiative | Phase | Status | Summary | Next checkpoint |",
        "| --- | --- | --- | --- | --- |",
    ]
    for status in statuses:
        lines.append(
            "| {initiative} | {phase} | {status} | {summary} | {next_checkpoint} |".format(
                initiative=status.initiative,
                phase=status.phase,
                status=status.status,
                summary=status.summary.replace("|", "\\|"),
                next_checkpoint=status.next_checkpoint.replace("|", "\\|"),
            )
        )
    return "\n".join(lines)


def _format_json(statuses: Iterable[InitiativeStatus]) -> str:
    return json.dumps([asdict(status) for status in statuses], indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the roadmap snapshot CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown table)",
    )
    args = parser.parse_args(argv)

    statuses = evaluate_portfolio_snapshot()
    if args.format == "json":
        output = _format_json(statuses)
    else:
        output = format_markdown(statuses)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
