"""Runtime application builder that separates ingestion and trading workloads.

This module translates the roadmap's runtime orchestration goals into concrete
helpers that assemble the Professional Predator runtime.  It exposes
``RuntimeApplication`` and ``RuntimeWorkload`` so the entrypoint can coordinate
ingest and trading lifecycles independently while keeping shutdown hooks
explicit and testable.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from collections.abc import Mapping as MappingABC
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Sequence, cast
from uuid import uuid4

from src.config.risk.risk_config import RiskConfig
from src.core.coercion import coerce_float
from src.core.event_bus import Event, EventBus, get_global_bus
from src.observability.logging import (
    OpenTelemetryLoggingSettings,
    configure_structured_logging,
    load_opentelemetry_logging_settings,
)
from src.observability.tracing import NullRuntimeTracer, RuntimeTracer
from src.data_foundation.batch.spark_export import (
    SparkExportSnapshot,
    SparkExportStatus,
    execute_spark_export_plan,
)
from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisConnectionSettings,
    configure_redis_client,
)
from src.data_foundation.ingest.configuration import (
    InstitutionalIngestConfig,
    TimescaleFailoverDrillSettings,
    TimescaleRetentionSettings,
    TimescaleSparkStressSettings,
    build_institutional_ingest_config,
)
from src.data_foundation.ingest.failover import (
    IngestFailoverDecision,
    decide_ingest_failover,
)
from src.data_foundation.ingest.health import (
    IngestHealthReport,
    IngestHealthStatus,
    evaluate_ingest_health,
)
from src.data_foundation.ingest.metrics import (
    IngestMetricsSnapshot,
    summarise_ingest_metrics,
)
from src.data_foundation.ingest.observability import build_ingest_observability_snapshot
from src.data_foundation.ingest.recovery import (
    IngestRecoveryRecommendation,
    plan_ingest_recovery,
)
from src.data_foundation.ingest.scheduler import IngestSchedulerState, TimescaleIngestScheduler
from src.data_foundation.ingest.scheduler_telemetry import (
    IngestSchedulerSnapshot,
    build_scheduler_snapshot,
    publish_scheduler_snapshot,
)
from src.data_foundation.ingest.telemetry import (
    EventBusIngestPublisher,
    combine_ingest_publishers,
)
from src.data_foundation.ingest.institutional_vertical import (
    InstitutionalIngestProvisioner,
    InstitutionalIngestServices,
)
from src.data_foundation.ingest.quality import (
    IngestQualityReport,
    IngestQualityStatus,
    evaluate_ingest_quality,
)
from src.data_foundation.ingest.timescale_pipeline import (
    TimescaleBackboneOrchestrator,
    TimescaleBackbonePlan,
)
from src.data_foundation.ingest.yahoo_ingest import fetch_daily_bars, store_duckdb
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestJournal,
    TimescaleIngestResult,
    TimescaleIngestRunRecord,
    TimescaleConfigurationAuditJournal,
    TimescaleMigrator,
)
from src.data_foundation.persist.timescale_reader import TimescaleReader
from src.data_foundation.pipelines.operational_backbone import (
    OperationalBackbonePipeline,
    OperationalIngestRequest,
)
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaConsumerLagSnapshot,
    KafkaTopicProvisioner,
    KafkaTopicProvisioningSummary,
    create_ingest_event_publisher,
    create_ingest_health_publisher,
    create_ingest_metrics_publisher,
    create_ingest_quality_publisher,
    resolve_ingest_topic_specs,
    should_auto_create_topics,
)
from src.data_integration.real_data_integration import RealDataManager
from src.governance.system_config import DataBackboneMode, EmpTier, SystemConfig
from src.operations.backup import (
    BackupPolicy,
    BackupReadinessSnapshot,
    BackupState,
    BackupStatus,
    evaluate_backup_readiness,
    format_backup_markdown,
)
from src.operations.cache_health import evaluate_cache_health, publish_cache_health
from src.operations.event_bus_health import (
    evaluate_event_bus_health,
    format_event_bus_markdown,
    publish_event_bus_health,
)
from src.operations.incident_response import (
    IncidentResponsePolicy,
    IncidentResponseState,
    evaluate_incident_response,
    format_incident_response_markdown,
    publish_incident_response_snapshot,
)
from src.operations.system_validation import (
    SystemValidationSnapshot,
    format_system_validation_markdown,
    load_system_validation_snapshot,
    publish_system_validation_snapshot,
)
from src.operations.data_backbone import (
    BackboneComponentSnapshot,
    BackboneRuntimeContext,
    BackboneStatus,
    DataBackboneReadinessSnapshot,
    DataBackboneValidationSnapshot,
    evaluate_data_backbone_readiness,
    evaluate_data_backbone_validation,
)
from src.compliance.workflow import (
    ComplianceWorkflowChecklist,
    ComplianceWorkflowSnapshot,
    WorkflowTaskStatus,
    evaluate_compliance_workflows,
    publish_compliance_workflows,
)
from src.operations.compliance_readiness import (
    ComplianceReadinessSnapshot,
    ComplianceReadinessStatus,
    evaluate_compliance_readiness,
    publish_compliance_readiness,
)
from src.operations.cross_region_failover import (
    CrossRegionFailoverSnapshot,
    CrossRegionStatus,
    evaluate_cross_region_failover,
    format_cross_region_markdown,
    publish_cross_region_snapshot,
)
from src.operations.failover_drill import (
    FailoverDrillSnapshot,
    FailoverDrillStatus,
    execute_failover_drill,
    format_failover_drill_markdown,
)
from src.operations.kafka_readiness import (
    KafkaReadinessSnapshot,
    KafkaReadinessStatus,
    evaluate_kafka_readiness,
    format_kafka_readiness_markdown,
    publish_kafka_readiness,
)
from src.operations.sensory_metrics import (
    build_sensory_metrics,
    publish_sensory_metrics,
)
from src.operations.sensory_summary import (
    build_sensory_summary,
    publish_sensory_summary,
)
from src.operations.regulatory_telemetry import (
    RegulatoryTelemetrySnapshot,
    RegulatoryTelemetryStatus,
    evaluate_regulatory_telemetry,
    publish_regulatory_telemetry,
)
from src.operations.governance_cadence import (
    build_governance_cadence_runner_from_config,
)
from src.operations.governance_reporting import (
    GovernanceReport,
    collect_audit_evidence,
    generate_governance_report,
    publish_governance_report,
)
from src.operations.spark_stress import (
    SparkStressSnapshot,
    SparkStressStatus,
    execute_spark_stress_drill,
    format_spark_stress_markdown,
)
from src.operations.ingest_trends import (
    IngestTrendSnapshot,
    IngestTrendStatus,
    evaluate_ingest_trends,
    format_ingest_trends_markdown,
    publish_ingest_trends,
)
from src.operations.configuration_audit import (
    evaluate_configuration_audit,
    format_configuration_audit_markdown,
    publish_configuration_audit_snapshot,
)
from src.operations.retention import (
    DataRetentionSnapshot,
    RetentionPolicy,
    RetentionStatus,
    evaluate_data_retention,
    format_data_retention_markdown,
    publish_data_retention,
)
from src.operations.evolution_experiments import (
    EvolutionExperimentSnapshot,
    evaluate_evolution_experiments,
    format_evolution_experiment_markdown,
    publish_evolution_experiment_snapshot,
)
from src.operations.evolution_tuning import (
    EvolutionTuningSnapshot,
    evaluate_evolution_tuning,
    format_evolution_tuning_markdown,
    publish_evolution_tuning_snapshot,
)
from src.operations.strategy_performance import (
    StrategyPerformanceSnapshot,
    evaluate_strategy_performance,
    format_strategy_performance_markdown,
    publish_strategy_performance_snapshot,
)
from src.operations.professional_readiness import (
    ProfessionalReadinessSnapshot,
    evaluate_professional_readiness,
)
from src.operations.security import (
    SecurityPolicy,
    SecurityState,
    evaluate_security_posture,
    publish_security_posture,
)
from src.operations.execution import (
    ExecutionPolicy,
    ExecutionReadinessSnapshot,
    ExecutionState,
    evaluate_execution_readiness,
    format_execution_markdown,
    publish_execution_snapshot,
)
from src.operations.fix_pilot import (
    FixPilotPolicy,
    evaluate_fix_pilot,
    format_fix_pilot_markdown,
    publish_fix_pilot_snapshot,
)
from src.operations.slo import evaluate_ingest_slos
from src.operations.sensory_drift import (
    evaluate_sensory_drift,
    publish_sensory_drift,
)
from src.runtime.healthcheck import RuntimeHealthServer
from src.runtime.predator_app import ProfessionalPredatorApp
from src.runtime.task_supervisor import TaskSupervisor
from src.trading.risk.risk_api import (
    RISK_API_RUNBOOK,
    RiskApiError,
    build_runtime_risk_metadata,
    resolve_trading_risk_config,
)


logger = logging.getLogger(__name__)


def _supervise_background_task(
    owner: "ProfessionalPredatorApp | RuntimeApplication" | Any,
    coro: Awaitable[Any],
    *,
    name: str,
    metadata: Mapping[str, Any] | None = None,
) -> asyncio.Task[Any]:
    """Schedule ``coro`` under a task supervisor associated with ``app``.

    The helper prefers the runtime application's ``create_background_task``
    API, falling back to the bound :class:`TaskSupervisor`. When neither is
    present (e.g. in lightweight tests), it lazily initialises a private
    supervisor so the coroutine still inherits managed lifecycle semantics
    rather than leaking through ``asyncio.create_task``.
    """

    create_background_task = getattr(owner, "create_background_task", None)
    if callable(create_background_task):
        return create_background_task(coro, name=name, metadata=metadata)

    supervisor = getattr(owner, "task_supervisor", None)
    if isinstance(supervisor, TaskSupervisor):
        return supervisor.create(coro, name=name, metadata=metadata)

    fallback = getattr(owner, "_fallback_task_supervisor", None)
    if not isinstance(fallback, TaskSupervisor):
        fallback = TaskSupervisor(namespace=f"runtime.{name}")
        setattr(owner, "_fallback_task_supervisor", fallback)
    return fallback.create(coro, name=name, metadata=metadata)


_RETENTION_TABLES: dict[str, tuple[str, str]] = {
    "daily_bars": ("market_data", "daily_bars"),
    "intraday_trades": ("market_data", "intraday_trades"),
    "macro_events": ("macro_data", "events"),
}


def _plan_dimensions(plan: TimescaleBackbonePlan) -> list[str]:
    dimensions: list[str] = []
    if plan.daily is not None:
        dimensions.append("daily_bars")
    if plan.intraday is not None:
        dimensions.append("intraday_trades")
    if plan.macro is not None:
        dimensions.append("macro_events")
    return dimensions


def _build_operational_request_from_plan(
    plan: TimescaleBackbonePlan,
) -> OperationalIngestRequest | None:
    """Translate a Timescale ingest plan into an operational ingest request."""

    daily_plan = plan.daily
    intraday_plan = plan.intraday
    macro_plan = plan.macro

    symbols: list[str] = []
    if daily_plan is not None:
        symbols.extend(daily_plan.normalised_symbols())
    if intraday_plan is not None:
        for symbol in intraday_plan.normalised_symbols():
            if symbol not in symbols:
                symbols.append(symbol)

    macro_events: tuple[Mapping[str, object] | object, ...] | None = None
    if macro_plan is not None and macro_plan.events:
        macro_events = tuple(macro_plan.events)

    macro_start = macro_plan.start if macro_plan is not None else None
    macro_end = macro_plan.end if macro_plan is not None else None

    if not symbols and macro_events is None and not (macro_start and macro_end):
        return None

    if not symbols:
        symbols = ["MACRO_ONLY"]

    source = "yahoo"
    if daily_plan is not None and daily_plan.source:
        source = daily_plan.source
    elif intraday_plan is not None and intraday_plan.source:
        source = intraday_plan.source

    macro_source = "fred"
    if macro_plan is not None and macro_plan.source:
        macro_source = macro_plan.source

    return OperationalIngestRequest(
        symbols=tuple(symbols),
        daily_lookback_days=daily_plan.lookback_days if daily_plan is not None else None,
        intraday_lookback_days=(
            intraday_plan.lookback_days if intraday_plan is not None else None
        ),
        intraday_interval=intraday_plan.interval if intraday_plan is not None else "1m",
        macro_start=macro_start,
        macro_end=macro_end,
        macro_events=macro_events,
        source=source,
        macro_source=macro_source,
    )


def _build_retention_policies(
    settings: TimescaleRetentionSettings,
) -> tuple[RetentionPolicy, ...]:
    policies: list[RetentionPolicy] = []
    for policy in settings.policies:
        schema, table = _RETENTION_TABLES.get(policy.dimension, ("public", policy.dimension))
        policies.append(
            RetentionPolicy(
                dimension=policy.dimension,
                schema=schema,
                table=table,
                target_days=policy.target_days,
                minimum_days=policy.minimum_days,
                optional=policy.optional,
            )
        )
    return tuple(policies)


def _normalise_ingest_plan_metadata(value: object) -> list[str]:
    if isinstance(value, MappingABC):
        return [str(key) for key in value.keys()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


_REGULATORY_STATUS_ORDER: Mapping[RegulatoryTelemetryStatus, int] = {
    RegulatoryTelemetryStatus.ok: 0,
    RegulatoryTelemetryStatus.warn: 1,
    RegulatoryTelemetryStatus.fail: 2,
}


def _escalate_regulatory_status(
    current: RegulatoryTelemetryStatus,
    candidate: RegulatoryTelemetryStatus,
) -> RegulatoryTelemetryStatus:
    if _REGULATORY_STATUS_ORDER[candidate] > _REGULATORY_STATUS_ORDER[current]:
        return candidate
    return current


def _compliance_status_to_regulatory(
    status: ComplianceReadinessStatus | None,
) -> RegulatoryTelemetryStatus:
    if status is ComplianceReadinessStatus.fail:
        return RegulatoryTelemetryStatus.fail
    if status is ComplianceReadinessStatus.warn:
        return RegulatoryTelemetryStatus.warn
    if status is ComplianceReadinessStatus.ok:
        return RegulatoryTelemetryStatus.ok
    return RegulatoryTelemetryStatus.warn


def _workflow_status_to_regulatory(
    status: WorkflowTaskStatus | None,
) -> RegulatoryTelemetryStatus:
    if status is WorkflowTaskStatus.blocked:
        return RegulatoryTelemetryStatus.fail
    if status in {WorkflowTaskStatus.in_progress, WorkflowTaskStatus.todo}:
        return RegulatoryTelemetryStatus.warn
    if status is WorkflowTaskStatus.completed:
        return RegulatoryTelemetryStatus.ok
    return RegulatoryTelemetryStatus.warn


def _workflow_details(workflow: ComplianceWorkflowChecklist) -> Mapping[str, object]:
    return {
        "name": workflow.name,
        "regulation": workflow.regulation,
        "status": workflow.status.value,
        "tasks": [
            {
                "task_id": task.task_id,
                "status": task.status.value,
                "summary": task.summary,
                "severity": task.severity,
                "metadata": dict(task.metadata),
            }
            for task in workflow.tasks
        ],
        "metadata": dict(workflow.metadata),
    }


def _summarise_workflows(
    workflows: Sequence[ComplianceWorkflowChecklist],
) -> tuple[RegulatoryTelemetryStatus, str, list[Mapping[str, object]]]:
    if not workflows:
        return (
            RegulatoryTelemetryStatus.warn,
            "no workflows configured",
            [],
        )

    status = RegulatoryTelemetryStatus.ok
    summary_parts: list[str] = []
    details: list[Mapping[str, object]] = []
    for workflow in workflows:
        workflow_status = _workflow_status_to_regulatory(workflow.status)
        status = _escalate_regulatory_status(status, workflow_status)
        summary_parts.append(f"{workflow.name} {workflow.status.value.upper()}")
        details.append(_workflow_details(workflow))

    return status, "; ".join(summary_parts), details


def _is_trade_reporting_workflow(workflow: ComplianceWorkflowChecklist) -> bool:
    name = workflow.name.lower()
    regulation = workflow.regulation.lower()
    return (
        "mifid" in name
        or "dodd" in name
        or "transaction" in name
        or regulation in {"mifid ii", "dodd-frank"}
    )


def _is_surveillance_workflow(workflow: ComplianceWorkflowChecklist) -> bool:
    name = workflow.name.lower()
    regulation = workflow.regulation.lower()
    return (
        "kyc" in name
        or "aml" in name
        or "surveillance" in name
        or "audit trail" in name
        or regulation in {"kyc/aml"}
    )


def _build_regulatory_signals(
    compliance_snapshot: ComplianceReadinessSnapshot | None,
    workflow_snapshot: ComplianceWorkflowSnapshot | None,
) -> list[Mapping[str, object]]:
    signals: list[Mapping[str, object]] = []

    component_map: dict[str, Any] = {}
    compliance_generated_at: datetime | None = None
    if isinstance(compliance_snapshot, ComplianceReadinessSnapshot):
        component_map = {component.name: component for component in compliance_snapshot.components}
        compliance_generated_at = compliance_snapshot.generated_at

    trade_component = component_map.get("trade_compliance")
    if trade_component is not None:
        signals.append(
            {
                "name": "trade_compliance",
                "status": trade_component.status.value,
                "summary": trade_component.summary,
                "observed_at": compliance_generated_at,
                "metadata": {
                    "source": "compliance_readiness",
                    "component": "trade_compliance",
                    "details": dict(trade_component.metadata),
                },
            }
        )
    else:
        signals.append(
            {
                "name": "trade_compliance",
                "status": RegulatoryTelemetryStatus.warn.value,
                "summary": "trade compliance readiness component missing",
                "metadata": {"reason": "component_missing"},
            }
        )

    kyc_component = component_map.get("kyc_aml")
    if kyc_component is not None:
        signals.append(
            {
                "name": "kyc_aml",
                "status": kyc_component.status.value,
                "summary": kyc_component.summary,
                "observed_at": compliance_generated_at,
                "metadata": {
                    "source": "compliance_readiness",
                    "component": "kyc_aml",
                    "details": dict(kyc_component.metadata),
                },
            }
        )
    else:
        signals.append(
            {
                "name": "kyc_aml",
                "status": RegulatoryTelemetryStatus.warn.value,
                "summary": "kyc readiness component missing",
                "metadata": {"reason": "component_missing"},
            }
        )

    workflows = () if workflow_snapshot is None else workflow_snapshot.workflows
    trade_reporting_workflows = [wf for wf in workflows if _is_trade_reporting_workflow(wf)]
    trade_status, trade_summary, trade_details = _summarise_workflows(trade_reporting_workflows)
    trade_metadata: dict[str, object] = {
        "workflows": trade_details,
    }
    if workflow_snapshot is not None and workflow_snapshot.metadata:
        trade_metadata["workflow_metadata"] = dict(workflow_snapshot.metadata)

    signals.append(
        {
            "name": "trade_reporting",
            "status": trade_status.value,
            "summary": trade_summary,
            "observed_at": None if workflow_snapshot is None else workflow_snapshot.generated_at,
            "metadata": trade_metadata,
        }
    )

    surveillance_workflows = [wf for wf in workflows if _is_surveillance_workflow(wf)]
    surveillance_status, surveillance_summary, surveillance_details = _summarise_workflows(
        surveillance_workflows
    )

    if kyc_component is not None:
        surveillance_status = _escalate_regulatory_status(
            surveillance_status, _compliance_status_to_regulatory(kyc_component.status)
        )

    surveillance_metadata: dict[str, object] = {
        "workflows": surveillance_details,
    }
    if kyc_component is not None:
        surveillance_metadata["kyc_component"] = {
            "status": kyc_component.status.value,
            "summary": kyc_component.summary,
            "metadata": dict(kyc_component.metadata),
        }
    if workflow_snapshot is not None and workflow_snapshot.metadata:
        surveillance_metadata.setdefault("workflow_metadata", dict(workflow_snapshot.metadata))

    signals.append(
        {
            "name": "surveillance",
            "status": surveillance_status.value,
            "summary": surveillance_summary,
            "observed_at": None if workflow_snapshot is None else workflow_snapshot.generated_at,
            "metadata": surveillance_metadata,
        }
    )

    return signals


def _resolve_strategy_identifier(extras: Mapping[str, object] | None) -> str | None:
    if not extras:
        return None
    for key in (
        "COMPLIANCE_STRATEGY_ID",
        "KYC_STRATEGY_ID",
        "EXECUTION_STRATEGY_ID",
        "TRADE_STRATEGY_ID",
        "STRATEGY_ID",
    ):
        value = extras.get(key)
        if value:
            return str(value)
    return None


StartupCallback = Callable[[], Awaitable[None] | None]
ShutdownCallback = Callable[[], Awaitable[None] | None]
WorkloadFactory = Callable[[], Awaitable[None]]


def _locate_trading_manager(app: ProfessionalPredatorApp) -> Any | None:
    sensory = getattr(app, "sensory_organ", None)
    if sensory is None:
        return None
    return getattr(sensory, "trading_manager", None)


def _process_sensory_status(
    app: ProfessionalPredatorApp,
    sensory_status: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    """Publish sensory summary/metrics and return audit entries for drift analysis."""

    if not isinstance(sensory_status, Mapping):
        return []

    try:
        summary = build_sensory_summary(sensory_status)
    except Exception:  # pragma: no cover - defensive guardrail
        logger.debug("Failed to build sensory summary from status payload", exc_info=True)
        audit_payload = sensory_status.get("sensor_audit")
        if isinstance(audit_payload, Sequence):
            return [entry for entry in audit_payload if isinstance(entry, Mapping)]
        return []

    logger.info("🧠 Sensory summary:\n%s", summary.to_markdown(limit=5))
    try:
        publish_sensory_summary(summary, event_bus=app.event_bus)
    finally:
        record_summary = getattr(app, "record_sensory_summary", None)
        if callable(record_summary):
            try:
                record_summary(summary)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Failed to record sensory summary on runtime app", exc_info=True)

    metrics = build_sensory_metrics(summary)
    logger.info(
        "📊 Sensory metrics: symbol=%s samples=%d drift_alerts=%s",
        metrics.symbol,
        metrics.samples,
        ",".join(metrics.drift_alerts) if metrics.drift_alerts else "none",
    )
    try:
        publish_sensory_metrics(metrics, event_bus=app.event_bus)
    finally:
        record_metrics = getattr(app, "record_sensory_metrics", None)
        if callable(record_metrics):
            try:
                record_metrics(metrics)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Failed to record sensory metrics on runtime app", exc_info=True)

    return [dict(entry) for entry in summary.audit_entries]


def _evaluate_and_publish_drift(
    app: ProfessionalPredatorApp,
    sensory_audit_entries: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Build and emit a sensory drift snapshot from audit entries."""

    if not sensory_audit_entries:
        logger.debug("Sensory drift evaluation skipped (no audit entries)")
        return

    drift_metadata: dict[str, object] = {
        "samples": len(sensory_audit_entries),
    }
    if metadata:
        for key, value in metadata.items():
            if value is not None:
                drift_metadata[str(key)] = value

    latest_entry = sensory_audit_entries[0]
    symbol = latest_entry.get("symbol")
    if symbol is not None:
        drift_metadata.setdefault("latest_symbol", str(symbol))

    unified_score_value = latest_entry.get("unified_score")
    unified_score = coerce_float(unified_score_value, default=None)
    if unified_score is not None:
        drift_metadata["latest_unified_score"] = unified_score

    confidence_value = latest_entry.get("confidence")
    confidence = coerce_float(confidence_value, default=None)
    if confidence is not None:
        drift_metadata["latest_confidence"] = confidence

    drift_snapshot = evaluate_sensory_drift(
        sensory_audit_entries,
        metadata=drift_metadata,
    )
    logger.info("🧠 Sensory drift snapshot:\n%s", drift_snapshot.to_markdown())
    publish_sensory_drift(app.event_bus, drift_snapshot)

    record_drift = getattr(app, "record_sensory_drift_snapshot", None)
    if callable(record_drift):
        try:
            record_drift(drift_snapshot)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug(
                "Failed to record sensory drift snapshot on runtime app",
                exc_info=True,
            )


def _publish_runtime_risk_configuration(
    app: "ProfessionalPredatorApp", payload: Mapping[str, object]
) -> None:
    """Publish a risk configuration telemetry event via the supplied bus."""

    event = Event(
        type="telemetry.risk.configuration",
        payload=dict(payload),
        source="runtime.builder",
    )

    event_bus = app.event_bus
    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to publish risk configuration via publish_from_sync", exc_info=True)

    publish_async = getattr(event_bus, "publish", None)
    if callable(publish_async):
        try:
            maybe_coro = publish_async(event)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to publish risk configuration asynchronously", exc_info=True)
        else:
            if inspect.isawaitable(maybe_coro):
                try:
                    _supervise_background_task(
                        app,
                        maybe_coro,
                        name="publish-risk-configuration",
                        metadata={"component": "runtime.risk"},
                    )
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to supervise risk configuration publish task",
                        exc_info=True,
                    )


def _record_and_publish_risk_configuration(
    app: ProfessionalPredatorApp,
    trading_manager: Any,
    risk_metadata: Mapping[str, object],
) -> None:
    """Persist and emit the enforced risk configuration snapshot."""

    payload: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "manager": type(trading_manager).__name__,
        "risk": dict(risk_metadata),
        "runbook": RISK_API_RUNBOOK,
    }

    try:
        _publish_runtime_risk_configuration(app, payload)
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("Failed to publish runtime risk configuration", exc_info=True)

    record_risk_configuration = getattr(app, "record_risk_configuration", None)
    if callable(record_risk_configuration):
        try:
            record_risk_configuration(payload)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to record runtime risk configuration on app", exc_info=True)


def _prepare_trading_risk_enforcement(
    app: ProfessionalPredatorApp, metadata: MutableMapping[str, object]
) -> StartupCallback | None:
    trading_manager = _locate_trading_manager(app)
    if trading_manager is None:
        message = "Trading manager not attached; cannot enforce RiskConfig"
        metadata["risk_error"] = {
            "message": message,
            "runbook": RISK_API_RUNBOOK,
        }
        logger.error(message)
        raise RuntimeError(f"{message}. See {RISK_API_RUNBOOK}")

    try:
        risk_config = resolve_trading_risk_config(trading_manager)
    except RiskApiError as exc:
        metadata["risk_error"] = exc.to_metadata()
        raise RuntimeError(f"{exc}. See {exc.runbook}") from exc
    if not risk_config.mandatory_stop_loss and not risk_config.research_mode:
        message = (
            "RiskConfig.mandatory_stop_loss must remain enabled outside research mode."
        )
        metadata["risk_error"] = {
            "message": message,
            "runbook": RISK_API_RUNBOOK,
            "mandatory_stop_loss": bool(risk_config.mandatory_stop_loss),
            "research_mode": bool(risk_config.research_mode),
        }
        raise RuntimeError(f"{message} See {RISK_API_RUNBOOK}")
    if risk_config.max_risk_per_trade_pct <= 0:
        raise RuntimeError("RiskConfig.max_risk_per_trade_pct must be positive")
    if risk_config.max_total_exposure_pct <= 0:
        raise RuntimeError("RiskConfig.max_total_exposure_pct must be positive")

    risk_metadata = dict(build_runtime_risk_metadata(trading_manager))
    metadata["risk"] = risk_metadata

    async def _enforce_trading_risk_config() -> None:
        logger.info(
            "🛡️ Trading risk configuration enforced: risk_per_trade=%.4f exposure_cap=%.4f"
            " mandatory_stop=%s research_mode=%s",
            float(risk_config.max_risk_per_trade_pct),
            float(risk_config.max_total_exposure_pct),
            bool(risk_config.mandatory_stop_loss),
            bool(risk_config.research_mode),
        )
        _record_and_publish_risk_configuration(app, trading_manager, risk_metadata)

    _enforce_trading_risk_config.__name__ = "enforce_trading_risk_config"
    return _enforce_trading_risk_config


@dataclass(frozen=True)
class WorkloadRestartPolicy:
    """Restart behaviour for runtime workloads."""

    max_restarts: int | None = None
    backoff_seconds: float = 5.0


@dataclass(frozen=True)
class RuntimeWorkload:
    """Description of an async workload managed by the runtime application."""

    name: str
    factory: WorkloadFactory
    description: str
    metadata: Mapping[str, object] | None = None
    restart_policy: WorkloadRestartPolicy | None = None


@dataclass
class RuntimeApplication:
    """Container that supervises ingestion and trading workloads."""

    ingestion: RuntimeWorkload | None = None
    trading: RuntimeWorkload | None = None
    shutdown_callbacks: list[ShutdownCallback] = field(default_factory=list)
    startup_callbacks: list[StartupCallback] = field(default_factory=list)
    tracer: RuntimeTracer | None = None
    task_supervisor: TaskSupervisor | None = field(default=None, repr=False)
    auxiliary: tuple[RuntimeWorkload, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.RuntimeApplication")
        self._shutdown_invoked = False
        self._tracer: RuntimeTracer = self.tracer or NullRuntimeTracer()
        self._workload_states: dict[str, str] = {}
        self._auxiliary_workloads: list[RuntimeWorkload] = list(self.auxiliary or ())
        supervisor = self.task_supervisor
        if supervisor is None:
            supervisor = TaskSupervisor(
                namespace="runtime.application",
                logger=self._logger,
            )
            self._owns_task_supervisor = True
        else:
            self._owns_task_supervisor = False
        self._task_supervisor: TaskSupervisor = supervisor
        self.task_supervisor = supervisor
        self.auxiliary = tuple(self._auxiliary_workloads)

    def create_background_task(
        self,
        coro: Awaitable[Any],
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> asyncio.Task[Any]:
        """Create and track a background task under the runtime supervisor."""

        if not asyncio.iscoroutine(coro):
            raise TypeError("RuntimeApplication.create_background_task expects a coroutine")
        return self._task_supervisor.create(coro, name=name, metadata=metadata)

    def register_background_task(
        self,
        task: asyncio.Task[Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Register an externally-created task with the runtime supervisor."""

        self._task_supervisor.track(task, metadata=metadata)

    def add_auxiliary_workload(self, workload: RuntimeWorkload) -> None:
        """Register an additional managed workload beyond ingest/trading."""

        if not isinstance(workload, RuntimeWorkload):
            raise TypeError("RuntimeApplication.add_auxiliary_workload expects RuntimeWorkload")
        self._auxiliary_workloads.append(workload)
        self.auxiliary = tuple(self._auxiliary_workloads)

    def bind_task_supervisor(self, supervisor: TaskSupervisor) -> None:
        """Bind the application to an external task supervisor."""

        if not isinstance(supervisor, TaskSupervisor):
            raise TypeError("RuntimeApplication.bind_task_supervisor expects TaskSupervisor")
        if supervisor is self._task_supervisor:
            return
        if getattr(self._task_supervisor, "active_count", 0):
            raise RuntimeError("Cannot rebind task supervisor while tasks are active")
        self._task_supervisor = supervisor
        self.task_supervisor = supervisor
        self._owns_task_supervisor = False

    def add_startup_callback(self, callback: StartupCallback) -> None:
        """Register a startup hook executed before workloads begin."""

        self.startup_callbacks.append(callback)

    def add_shutdown_callback(self, callback: ShutdownCallback) -> None:
        """Register a shutdown hook executed in reverse order when closing."""

        self.shutdown_callbacks.append(callback)

    async def shutdown(self) -> None:
        """Invoke registered shutdown callbacks once."""

        if self._shutdown_invoked:
            return
        self._shutdown_invoked = True

        try:
            await self._task_supervisor.cancel_all()
        except Exception:  # pragma: no cover - defensive logging
            self._logger.exception("Failed to cancel runtime background tasks during shutdown")

        for callback in reversed(self.shutdown_callbacks):
            callback_name = getattr(callback, "__name__", repr(callback))
            with self._tracer.operation_span(
                name="runtime.shutdown",
                metadata={"callback": callback_name},
            ) as span:
                try:
                    result = callback()
                    if inspect.isawaitable(result):
                        await result
                except Exception:  # pragma: no cover - defensive logging
                    if span is not None and hasattr(span, "set_attribute"):
                        span.set_attribute("runtime.operation.status", "error")
                    self._logger.exception("Runtime shutdown callback %s failed", callback)
                else:
                    if span is not None and hasattr(span, "set_attribute"):
                        span.set_attribute("runtime.operation.status", "completed")

    async def _run_workload(self, workload: RuntimeWorkload) -> None:
        name = workload.name
        self._workload_states[name] = "running"
        self._logger.info("▶️ starting %s", name)
        metadata: dict[str, object] = {}
        if workload.description:
            metadata["workload.description"] = workload.description
        if workload.metadata:
            for key, value in workload.metadata.items():
                if value is None:
                    continue
                metadata[f"workload.metadata.{key}"] = value

        with self._tracer.workload_span(
            workload=workload.name,
            metadata=metadata,
        ) as span:
            try:
                await workload.factory()
            except asyncio.CancelledError:
                self._workload_states[name] = "cancelled"
                if span is not None and hasattr(span, "set_attribute"):
                    span.set_attribute("runtime.workload.status", "cancelled")
                self._logger.info("⏹️ %s cancelled", name)
                raise
            except Exception:
                self._workload_states[name] = "failed"
                if span is not None and hasattr(span, "set_attribute"):
                    span.set_attribute("runtime.workload.status", "error")
                self._logger.exception("❌ runtime workload %s failed", name)
                raise
            else:
                self._workload_states[name] = "finished"
                if span is not None and hasattr(span, "set_attribute"):
                    span.set_attribute("runtime.workload.status", "completed")
                self._logger.info("✅ %s completed", name)

    async def run(self) -> None:
        """Run the configured workloads and execute shutdown hooks on exit."""

        try:
            for callback in list(self.startup_callbacks):
                callback_name = getattr(callback, "__name__", repr(callback))
                with self._tracer.operation_span(
                    name="runtime.startup",
                    metadata={"callback": callback_name},
                ) as span:
                    try:
                        result = callback()
                        if inspect.isawaitable(result):
                            await result
                    except Exception:  # pragma: no cover - defensive logging
                        if span is not None and hasattr(span, "set_attribute"):
                            span.set_attribute("runtime.operation.status", "error")
                        self._logger.exception("Runtime startup callback %s failed", callback)
                        raise
                    else:
                        if span is not None and hasattr(span, "set_attribute"):
                            span.set_attribute("runtime.operation.status", "completed")
            workloads: list[RuntimeWorkload] = []
            if self.ingestion is not None:
                workloads.append(self.ingestion)
            if self.trading is not None:
                workloads.append(self.trading)
            if self._auxiliary_workloads:
                workloads.extend(self._auxiliary_workloads)
            if workloads:
                task_mapping: dict[asyncio.Task[Any], RuntimeWorkload] = {}
                all_tasks: set[asyncio.Task[Any]] = set()
                try:
                    for workload in workloads:
                        task_name = f"{workload.name}-workload"
                        metadata_payload: dict[str, object] = {"workload": workload.name}
                        if workload.description:
                            metadata_payload["description"] = workload.description
                        if workload.metadata:
                            metadata_payload["workload_metadata"] = dict(workload.metadata)
                        restart_policy = workload.restart_policy
                        self._workload_states[workload.name] = "scheduled"
                        run_coro = self._run_workload(workload)
                        restart_factory: Callable[[], Awaitable[Any]] | None = None
                        if restart_policy is not None:
                            restart_factory = lambda workload=workload: self._run_workload(
                                workload
                            )
                        task = self._task_supervisor.create(
                            run_coro,
                            name=task_name,
                            metadata=metadata_payload,
                            restart_callback=restart_factory,
                            max_restarts=None
                            if restart_policy is None
                            else restart_policy.max_restarts,
                            restart_backoff=0.0
                            if restart_policy is None
                            else restart_policy.backoff_seconds,
                        )
                        task_mapping[task] = workload
                        all_tasks.add(task)

                    pending: set[asyncio.Task[Any]] = set(task_mapping.keys())

                    while pending:
                        done, pending = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done:
                            workload = task_mapping.get(task)
                            name = workload.name if workload is not None else "<unknown>"
                            if task.cancelled():
                                self._logger.debug(
                                    "Runtime workload %s cancelled under supervisor", name
                                )
                                continue
                            try:
                                exc = task.exception()
                            except Exception:
                                exc = None
                                self._logger.exception(
                                    "Unable to determine completion state for %s", name
                                )
                                continue
                            if exc is None:
                                self._logger.info("🏁 runtime workload %s exited", name)
                            else:
                                self._logger.error(
                                    "runtime workload %s terminated with error: %s",
                                    name,
                                    exc,
                                    exc_info=exc,
                                )
                        # remove finished tasks from mapping to avoid repeated logging
                        for task in done:
                            task_mapping.pop(task, None)
                finally:
                    for task in list(task_mapping.keys()):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*all_tasks, return_exceptions=True)
        finally:
            await self.shutdown()

    def task_snapshots(self) -> tuple[dict[str, object], ...]:
        """Expose metadata for the supervised runtime tasks.

        Operators and regression tests can call this helper to verify that
        ingestion/trading loops are running under the shared
        :class:`TaskSupervisor` and to introspect restart counters or
        metadata emitted for each task.  Returning an immutable tuple keeps
        callers from mutating the supervisor's internal tracking state while
        still providing a serialisable payload for diagnostics.
        """

        return tuple(self._task_supervisor.describe())

    def summary(self) -> dict[str, object]:
        """Summarise the configured workloads for logging/tests."""

        def _pack(workload: RuntimeWorkload | None) -> Mapping[str, object] | None:
            if workload is None:
                return None
            payload: MutableMapping[str, object] = {
                "name": workload.name,
                "description": workload.description,
            }
            if workload.metadata:
                payload["metadata"] = dict(workload.metadata)
            state = self._workload_states.get(workload.name)
            if state is not None:
                payload["state"] = state
            if workload.restart_policy is not None:
                payload["restart_policy"] = {
                    "max_restarts": workload.restart_policy.max_restarts,
                    "backoff_seconds": workload.restart_policy.backoff_seconds,
                }
            return payload

        summary: MutableMapping[str, object] = {
            "ingestion": _pack(self.ingestion),
            "trading": _pack(self.trading),
            "auxiliary": tuple(filter(None, (_pack(workload) for workload in self._auxiliary_workloads))),
            "shutdown_callbacks": len(self.shutdown_callbacks),
            "startup_callbacks": len(self.startup_callbacks),
            "workload_states": dict(self._workload_states),
        }

        supervisor_namespace = getattr(self._task_supervisor, "namespace", None)
        supervisor_details: MutableMapping[str, object] = {
            "namespace": supervisor_namespace,
            "active_tasks": self._task_supervisor.active_count,
        }
        snapshots = list(self.task_snapshots())
        if snapshots:
            supervisor_details["tasks"] = snapshots
        summary["task_supervisor"] = supervisor_details

        return summary


async def _run_tier0_ingest(
    app: ProfessionalPredatorApp,
    *,
    symbols_csv: str,
    db_path: str,
) -> None:
    """Execute Tier-0 ingest and fan out data through configured sensors."""

    symbols = [s.strip() for s in symbols_csv.split(",") if s.strip()]
    if not symbols:
        logger.info("No symbols supplied for Tier-0 ingest; skipping")
        return

    logger.info("📥 Tier-0 ingest for %s", symbols)
    sensor_items = list(app.sensors.items())
    destination = Path(db_path)

    def _ingest() -> tuple[int, int]:
        df = fetch_daily_bars(symbols)
        if df.empty:
            return 0, 0

        store_duckdb(df, destination)

        total_signals = 0
        for name, sensor in sensor_items:
            try:
                signals = sensor.process(df)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Sensor %s failed during Tier-0 ingest: %s", name, exc)
            else:
                total_signals += len(signals)

        return len(df), total_signals

    try:
        rows, signal_count = await asyncio.to_thread(_ingest)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Tier-0 ingest failed (continuing): %s", exc)
        return

    if rows:
        logger.info("✅ Stored %s rows to %s", rows, db_path)
    logger.info("🧠 Signals produced: count=%s", signal_count)


def _publish_ingest_health(event_bus: EventBus, payload: Mapping[str, object]) -> None:
    event = Event(type="telemetry.ingest.health", payload=dict(payload), source="timescale_ingest")

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to publish ingest health via runtime event bus", exc_info=True)

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Ingest health telemetry publish skipped", exc_info=True)


def _publish_ingest_failover(event_bus: EventBus, decision: IngestFailoverDecision) -> None:
    payload = decision.as_dict()
    event = Event(type="telemetry.ingest.failover", payload=payload, source="timescale_ingest")

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to publish ingest failover decision", exc_info=True)

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Ingest failover telemetry publish skipped", exc_info=True)


def _publish_failover_drill(event_bus: EventBus, snapshot: FailoverDrillSnapshot) -> None:
    payload = snapshot.as_dict()
    event = Event(
        type="telemetry.ingest.failover_drill",
        payload=payload,
        source="timescale_ingest",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to publish failover drill telemetry", exc_info=True)

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Failover drill telemetry publish skipped", exc_info=True)


def _publish_ingest_metrics(event_bus: EventBus, snapshot: IngestMetricsSnapshot) -> None:
    payload = snapshot.as_dict()
    event = Event(type="telemetry.ingest.metrics", payload=payload, source="timescale_ingest")

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to publish ingest metrics via runtime event bus", exc_info=True)

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Ingest metrics telemetry publish skipped", exc_info=True)


def _publish_ingest_quality(event_bus: EventBus, payload: Mapping[str, object]) -> None:
    event = Event(
        type="telemetry.ingest.quality",
        payload=dict(payload),
        source="timescale_ingest",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to publish ingest quality via runtime event bus", exc_info=True)

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Ingest quality telemetry publish skipped", exc_info=True)


def _publish_spark_exports(event_bus: EventBus, snapshot: SparkExportSnapshot) -> None:
    payload = snapshot.as_dict()
    event = Event(
        type="telemetry.ingest.spark_exports",
        payload=payload,
        source="timescale_ingest",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to publish spark export telemetry", exc_info=True)

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Spark export telemetry publish skipped", exc_info=True)


def _publish_spark_stress(event_bus: EventBus, snapshot: SparkStressSnapshot) -> None:
    payload = snapshot.as_dict()
    event = Event(
        type="telemetry.ingest.spark_stress",
        payload=payload,
        source="timescale_ingest",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to publish spark stress telemetry", exc_info=True)

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Spark stress telemetry publish skipped", exc_info=True)


def _publish_ingest_observability(event_bus: EventBus, payload: Mapping[str, object]) -> None:
    event = Event(
        type="telemetry.ingest.observability",
        payload=dict(payload),
        source="timescale_ingest",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to publish ingest observability via runtime event bus", exc_info=True
            )

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Ingest observability telemetry publish skipped", exc_info=True)


def _publish_operational_slos(event_bus: EventBus, snapshot) -> None:
    event = Event(
        type="telemetry.operational.slos",
        payload=snapshot.as_dict(),
        source="operations.timescale_ingest",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to publish operational SLO snapshot via runtime event bus",
                exc_info=True,
            )

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Operational SLO telemetry publish skipped", exc_info=True)


def _publish_backup_snapshot(event_bus: EventBus, snapshot: BackupReadinessSnapshot) -> None:
    event = Event(
        type="telemetry.operational.backups",
        payload=snapshot.as_dict(),
        source="operations.backup",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to publish backup snapshot via runtime event bus", exc_info=True)

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Backup telemetry publish skipped", exc_info=True)


def _publish_data_backbone_validation(
    event_bus: EventBus, snapshot: DataBackboneValidationSnapshot
) -> None:
    event = Event(
        type="telemetry.data_backbone.validation",
        payload=snapshot.as_dict(),
        source="timescale_ingest",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to publish data backbone validation via runtime event bus",
                exc_info=True,
            )

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Data backbone validation telemetry publish skipped", exc_info=True)


def _publish_data_backbone_readiness(
    event_bus: EventBus, snapshot: DataBackboneReadinessSnapshot
) -> None:
    event = Event(
        type="telemetry.data_backbone.readiness",
        payload=snapshot.as_dict(),
        source="timescale_ingest",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to publish data backbone readiness via runtime event bus",
                exc_info=True,
            )

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Data backbone readiness telemetry publish skipped", exc_info=True)


def _publish_professional_readiness(
    event_bus: EventBus, snapshot: ProfessionalReadinessSnapshot
) -> None:
    event = Event(
        type="telemetry.operational.readiness",
        payload=snapshot.as_dict(),
        source="professional_runtime",
    )

    publish_from_sync = getattr(event_bus, "publish_from_sync", None)
    if callable(publish_from_sync) and event_bus.is_running():
        try:
            publish_from_sync(event)
            return
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to publish professional readiness via runtime event bus",
                exc_info=True,
            )

    try:
        topic_bus = get_global_bus()
        topic_bus.publish_sync(event.type, event.payload, source=event.source)
    except Exception:  # pragma: no cover - defensive logging
        logger.debug("Professional readiness telemetry publish skipped", exc_info=True)


def _merge_ingest_results(
    existing: TimescaleIngestResult,
    new: TimescaleIngestResult,
) -> TimescaleIngestResult:
    symbols: list[str] = []
    seen: set[str] = set()
    for collection in (existing.symbols, new.symbols):
        for symbol in collection:
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)

    start_ts = existing.start_ts
    if new.start_ts is not None and (start_ts is None or new.start_ts < start_ts):
        start_ts = new.start_ts

    end_ts = existing.end_ts
    if new.end_ts is not None and (end_ts is None or new.end_ts > end_ts):
        end_ts = new.end_ts

    duration = existing.ingest_duration_seconds + new.ingest_duration_seconds

    if existing.freshness_seconds is None:
        freshness = new.freshness_seconds
    elif new.freshness_seconds is None:
        freshness = existing.freshness_seconds
    else:
        freshness = min(existing.freshness_seconds, new.freshness_seconds)

    rows_written = existing.rows_written + new.rows_written
    source = new.source or existing.source

    return TimescaleIngestResult(
        rows_written=rows_written,
        symbols=tuple(symbols),
        start_ts=start_ts,
        end_ts=end_ts,
        ingest_duration_seconds=duration,
        freshness_seconds=freshness,
        dimension=existing.dimension,
        source=source,
    )


def _record_ingest_journal(
    ingest_config: InstitutionalIngestConfig,
    results: Mapping[str, TimescaleIngestResult],
    health_report: IngestHealthReport,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    if not results and not health_report.checks:
        return

    engine = None
    try:
        engine = ingest_config.timescale_settings.create_engine()
        journal = TimescaleIngestJournal(engine)
        checks = {check.dimension: check for check in health_report.checks}
        overall_status = health_report.status.value
        executed_at = (
            health_report.generated_at
            if isinstance(health_report.generated_at, datetime)
            else datetime.now(UTC)
        )
        plan_metadata = dict(metadata or ingest_config.metadata)
        quality_metadata = plan_metadata.pop("quality", None)
        quality_info: dict[str, object] = {}
        if isinstance(quality_metadata, Mapping):
            status = quality_metadata.get("status")
            score = quality_metadata.get("score")
            if status is not None:
                quality_info["quality_status"] = status
            if score is not None:
                quality_info["quality_score"] = score
        records: list[TimescaleIngestRunRecord] = []
        recorded: set[str] = set()

        for dimension, check in checks.items():
            result = results.get(dimension, TimescaleIngestResult.empty(dimension=dimension))
            entry_metadata: dict[str, object] = {
                "message": check.message,
                "overall_status": overall_status,
            }
            if quality_info:
                entry_metadata.update(quality_info)
            if check.expected_symbols:
                entry_metadata["expected_symbols"] = list(check.expected_symbols)
            if check.missing_symbols:
                entry_metadata["missing_symbols"] = list(check.missing_symbols)
            if check.observed_symbols:
                entry_metadata["observed_symbols"] = list(check.observed_symbols)
            if check.metadata:
                entry_metadata["health_metadata"] = dict(check.metadata)
            if plan_metadata:
                entry_metadata["plan"] = dict(plan_metadata)
            entry_metadata = {
                key: value
                for key, value in entry_metadata.items()
                if value not in (None, (), [], {}, "")
            }
            symbols = result.symbols or tuple(check.observed_symbols)
            records.append(
                TimescaleIngestRunRecord(
                    run_id=str(uuid4()),
                    dimension=dimension,
                    status=check.status.value,
                    rows_written=result.rows_written,
                    freshness_seconds=result.freshness_seconds,
                    ingest_duration_seconds=result.ingest_duration_seconds,
                    executed_at=executed_at,
                    source=result.source,
                    symbols=symbols,
                    metadata=entry_metadata,
                )
            )
            recorded.add(dimension)

        for dimension, result in results.items():
            if dimension in recorded:
                continue
            fallback_metadata: dict[str, object] = {"overall_status": overall_status}
            if quality_info:
                fallback_metadata.update(quality_info)
            if plan_metadata:
                fallback_metadata["plan"] = dict(plan_metadata)
            fallback_metadata = {
                key: value
                for key, value in fallback_metadata.items()
                if value not in (None, (), [], {}, "")
            }
            records.append(
                TimescaleIngestRunRecord(
                    run_id=str(uuid4()),
                    dimension=dimension,
                    status="ok" if result.rows_written else "skipped",
                    rows_written=result.rows_written,
                    freshness_seconds=result.freshness_seconds,
                    ingest_duration_seconds=result.ingest_duration_seconds,
                    executed_at=executed_at,
                    source=result.source,
                    symbols=result.symbols,
                    metadata=fallback_metadata,
                )
            )

        if records:
            journal.record(records)
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to record Timescale ingest journal")
    finally:
        if engine is not None:
            engine.dispose()


async def _execute_timescale_ingest(
    *,
    ingest_config: InstitutionalIngestConfig,
    event_bus: EventBus,
    publisher,
    kafka_health_publisher,
    kafka_metrics_publisher,
    kafka_quality_publisher,
    fallback: Callable[[], Awaitable[None]] | None,
    orchestrator_cls: type[TimescaleBackboneOrchestrator] = TimescaleBackboneOrchestrator,
    data_manager: RealDataManager | None = None,
    backbone_context: BackboneRuntimeContext | None = None,
    scheduler_snapshot: IngestSchedulerSnapshot | None = None,
    record_backbone_validation_snapshot: (
        Callable[[DataBackboneValidationSnapshot], None] | None
    ) = None,
    record_backbone_snapshot: Callable[[DataBackboneReadinessSnapshot], None] | None = None,
    record_professional_snapshot: Callable[[ProfessionalReadinessSnapshot], None] | None = None,
    record_spark_snapshot: Callable[[SparkExportSnapshot], None] | None = None,
    record_data_retention_snapshot: Callable[[DataRetentionSnapshot], None] | None = None,
    spark_stress_settings: TimescaleSparkStressSettings | None = None,
    record_spark_stress_snapshot: Callable[[SparkStressSnapshot], None] | None = None,
    failover_drill_settings: TimescaleFailoverDrillSettings | None = None,
    record_failover_drill_snapshot: Callable[[FailoverDrillSnapshot], None] | None = None,
    record_ingest_trend_snapshot: Callable[[IngestTrendSnapshot], None] | None = None,
    record_cross_region_snapshot: Callable[[CrossRegionFailoverSnapshot], None] | None = None,
    kafka_settings: KafkaConnectionSettings | None = None,
    kafka_topics: Sequence[str] = (),
    kafka_publishers: Sequence[str] = (),
    kafka_provisioning: KafkaTopicProvisioningSummary | None = None,
    kafka_lag_snapshot: KafkaConsumerLagSnapshot | None = None,
    record_kafka_readiness_snapshot: Callable[[KafkaReadinessSnapshot], None] | None = None,
    managed_manifest: Sequence[Mapping[str, object]] | None = None,
    task_snapshots: Sequence[Mapping[str, object]] | None = None,
) -> tuple[bool, BackupReadinessSnapshot | None]:
    initial_results: dict[str, TimescaleIngestResult] = {}
    pipeline_result = None
    cache_metrics_before: Mapping[str, object] = {}
    cache_metrics_after_ingest: Mapping[str, object] = {}
    cache_metrics_after_fetch: Mapping[str, object] = {}
    kafka_events: tuple[Event, ...] = ()
    ingest_error: str | None = None
    manager_failed = False

    manager = data_manager
    if manager is not None:
        request = _build_operational_request_from_plan(ingest_config.plan)
        try:
            if request is not None:
                topics_for_pipeline = tuple(dict.fromkeys(kafka_topics)) or ("telemetry.ingest",)
                pipeline: OperationalBackbonePipeline | None = None
                try:
                    pipeline = OperationalBackbonePipeline(
                        manager=manager,
                        event_bus=event_bus,
                        event_topics=topics_for_pipeline,
                        auto_close_consumer=False,
                        shutdown_manager_on_close=False,
                    )
                    pipeline_result = await pipeline.execute(request, poll_consumer=False)
                    initial_results = dict(pipeline_result.ingest_results)
                    cache_metrics_before = dict(pipeline_result.cache_metrics_before)
                    cache_metrics_after_ingest = dict(pipeline_result.cache_metrics_after_ingest)
                    cache_metrics_after_fetch = dict(pipeline_result.cache_metrics_after_fetch)
                    kafka_events = pipeline_result.kafka_events
                    ingest_error = pipeline_result.ingest_error
                finally:
                    if pipeline is not None:
                        try:
                            await pipeline.shutdown()
                        except Exception:  # pragma: no cover - defensive cleanup
                            logger.exception(
                                "Operational backbone pipeline shutdown failed"
                            )
            else:
                cache_metrics_before = manager.cache_metrics(reset=True)
                initial_results = await asyncio.to_thread(
                    manager.run_ingest_plan,
                    ingest_config.plan,
                )
                cache_metrics_after_ingest = manager.cache_metrics(reset=False)
                cache_metrics_after_fetch = dict(cache_metrics_after_ingest)
        except Exception as exc:
            logger.exception(
                "Operational backbone ingest failed; falling back to Timescale orchestrator"
            )
            manager_failed = True
            pipeline_result = None
            cache_metrics_before = {}
            cache_metrics_after_ingest = {}
            cache_metrics_after_fetch = {}
            kafka_events = ()
            ingest_error = str(exc)
            initial_results = {}

    if manager is None or manager_failed:
        try:
            orchestrator = orchestrator_cls(
                ingest_config.timescale_settings,
                event_publisher=publisher,
            )
            initial_results = orchestrator.run(plan=ingest_config.plan)
        except Exception:
            logger.exception("Timescale ingest failed")
            return False, None

    if not initial_results:
        logger.info("Timescale ingest skipped: nothing requested")
        return True, None

    backbone_context = backbone_context or BackboneRuntimeContext()
    kafka_settings = kafka_settings or ingest_config.kafka_settings
    kafka_topics_tuple = tuple(kafka_topics)
    kafka_publishers_tuple = tuple(kafka_publishers)
    if managed_manifest is None:
        managed_manifest_tuple: tuple[dict[str, object], ...] = ()
    else:
        managed_manifest_tuple = tuple(dict(snapshot) for snapshot in managed_manifest)
    aggregated_results: dict[str, TimescaleIngestResult] = dict(initial_results)

    for dimension, outcome in initial_results.items():
        logger.info("🗄️ Timescale ingest %s: %s", dimension, outcome.as_dict())

    if kafka_events:
        logger.info("📨 Kafka ingest events emitted: %s", len(kafka_events))

    telemetry_metadata: dict[str, object] = dict(ingest_config.metadata)
    if cache_metrics_before:
        telemetry_metadata["cache_metrics_before"] = dict(cache_metrics_before)
    if cache_metrics_after_ingest:
        telemetry_metadata["cache_metrics_after_ingest"] = dict(cache_metrics_after_ingest)
    if cache_metrics_after_fetch:
        telemetry_metadata["cache_metrics_after_fetch"] = dict(cache_metrics_after_fetch)
    if ingest_error:
        telemetry_metadata["ingest_error"] = ingest_error
    if kafka_events:
        telemetry_metadata["kafka_event_count"] = len(kafka_events)
    telemetry_metadata["kafka_streaming_enabled"] = ingest_config.enable_streaming
    if task_snapshots:
        telemetry_metadata["task_supervisor"] = {
            "count": len(tuple(task_snapshots)),
        }
    validation_snapshot = evaluate_data_backbone_validation(
        ingest_config=ingest_config,
        context=backbone_context,
        scheduler_snapshot=scheduler_snapshot,
        metadata=telemetry_metadata,
    )
    logger.info(
        "🧪 Data backbone validation snapshot:\n%s",
        validation_snapshot.to_markdown(),
    )
    _publish_data_backbone_validation(event_bus, validation_snapshot)
    telemetry_metadata["validation"] = validation_snapshot.as_dict()
    if record_backbone_validation_snapshot is not None:
        try:
            record_backbone_validation_snapshot(validation_snapshot)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug(
                "Failed to record data backbone validation snapshot on runtime app",
                exc_info=True,
            )
    if validation_snapshot.status is BackboneStatus.fail:
        logger.error("🚫 Data backbone validation failed; skipping Timescale ingest run")
        base_snapshot = evaluate_data_backbone_readiness(
            ingest_config=ingest_config,
            context=backbone_context,
            metadata=telemetry_metadata,
            task_snapshots=task_snapshots,
        )
        components = list(base_snapshot.components)
        components.append(
            BackboneComponentSnapshot(
                name="validation",
                status=BackboneStatus.fail,
                summary="Validation failed; skipping Timescale ingest",
                metadata={
                    "status": validation_snapshot.status.value,
                    "checks": [check.as_dict() for check in validation_snapshot.checks],
                },
            )
        )
        readiness_metadata = dict(base_snapshot.metadata)
        readiness_metadata.setdefault(
            "validation",
            {
                "status": validation_snapshot.status.value,
                "checks": [check.as_dict() for check in validation_snapshot.checks],
            },
        )
        readiness_snapshot = DataBackboneReadinessSnapshot(
            status=BackboneStatus.fail,
            generated_at=base_snapshot.generated_at,
            components=tuple(components),
            metadata=readiness_metadata,
        )
        logger.error(
            "🏗️ Data backbone readiness snapshot after validation failure:\n%s",
            readiness_snapshot.to_markdown(),
        )
        _publish_data_backbone_readiness(event_bus, readiness_snapshot)
        if record_backbone_snapshot is not None:
            try:
                record_backbone_snapshot(readiness_snapshot)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to record data backbone snapshot on runtime app",
                    exc_info=True,
                )

        professional_snapshot = evaluate_professional_readiness(
            backbone_snapshot=readiness_snapshot,
            metadata={
                "ingest_attempted": False,
                "reason": "validation_failed",
            },
        )
        logger.error(
            "🏢 Professional readiness snapshot after validation failure:\n%s",
            professional_snapshot.to_markdown(),
        )
        _publish_professional_readiness(event_bus, professional_snapshot)
        if record_professional_snapshot is not None:
            try:
                record_professional_snapshot(professional_snapshot)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to record professional readiness snapshot on runtime app",
                    exc_info=True,
                )
        if fallback is not None:
            await fallback()
        return False, None
    recovery_steps: list[dict[str, object]] = []
    last_recovery_recommendation: IngestRecoveryRecommendation | None = None

    health_report = evaluate_ingest_health(
        aggregated_results,
        plan=ingest_config.plan,
        metadata=telemetry_metadata,
    )

    recovery_settings = ingest_config.recovery
    attempts = 0
    while (
        recovery_settings.should_attempt()
        and health_report.status is not IngestHealthStatus.ok
        and attempts < recovery_settings.max_attempts
    ):
        attempts += 1
        recommendation = plan_ingest_recovery(
            health_report,
            original_plan=ingest_config.plan,
            results=aggregated_results,
            settings=recovery_settings,
            attempt=attempts,
        )
        if recommendation.is_empty():
            logger.info(
                "♻️ Timescale ingest recovery attempt %s skipped: no actionable slices",
                attempts,
            )
            break

        last_recovery_recommendation = recommendation
        summary = recommendation.summary()
        logger.info(
            "♻️ Timescale ingest recovery attempt %s planned: %s",
            attempts,
            summary.get("plan", summary),
        )

        try:
            recovery_results = orchestrator.run(plan=recommendation.plan)
        except Exception:
            logger.exception("Timescale ingest recovery attempt %s failed", attempts)
            break

        if not recovery_results:
            logger.info("♻️ Timescale ingest recovery attempt %s produced no results", attempts)

        for dimension, outcome in recovery_results.items():
            logger.info("♻️ Timescale recovery %s: %s", dimension, outcome.as_dict())
            prior = aggregated_results.get(dimension)
            if prior is None:
                aggregated_results[dimension] = outcome
            else:
                aggregated_results[dimension] = _merge_ingest_results(prior, outcome)

        step_info: dict[str, object] = {
            "attempt": attempts,
            "dimensions": sorted(recovery_results.keys()),
            "plan": summary.get("plan", {}),
            "reasons": dict(recommendation.reasons),
            "missing_symbols": {
                dim: list(symbols) for dim, symbols in recommendation.missing_symbols.items()
            },
        }
        recovery_steps.append(step_info)

        telemetry_metadata = dict(ingest_config.metadata)
        telemetry_metadata["recovery"] = {
            "attempts": attempts,
            "steps": recovery_steps,
        }

        health_report = evaluate_ingest_health(
            aggregated_results,
            plan=ingest_config.plan,
            metadata=telemetry_metadata,
        )
        step_info["health_status"] = health_report.status.value

        if health_report.status is IngestHealthStatus.ok:
            logger.info(
                "♻️ Timescale ingest recovery attempt %s resolved remaining issues",
                attempts,
            )
            break
        logger.warning(
            "♻️ Timescale ingest recovery attempt %s status: %s",
            attempts,
            health_report.status.value,
        )

    if recovery_steps and health_report.status is not IngestHealthStatus.ok:
        logger.warning(
            "♻️ Timescale ingest recovery exhausted after %s attempts; latest status: %s",
            len(recovery_steps),
            health_report.status.value,
        )

    report_generated_at = health_report.generated_at
    if isinstance(report_generated_at, datetime):
        if report_generated_at.tzinfo is None:
            report_generated_at = report_generated_at.replace(tzinfo=UTC)
        else:
            report_generated_at = report_generated_at.astimezone(UTC)
    else:
        report_generated_at = datetime.now(tz=UTC)

    metrics_snapshot = summarise_ingest_metrics(aggregated_results)
    if metrics_snapshot.dimensions:
        logger.info(
            "📈 Timescale ingest metrics: total_rows=%s active=%s",
            metrics_snapshot.total_rows(),
            metrics_snapshot.active_dimensions(),
        )
        _publish_ingest_metrics(event_bus, metrics_snapshot)
        if kafka_metrics_publisher is not None:
            try:
                kafka_metrics_publisher.publish(
                    metrics_snapshot,
                    metadata=telemetry_metadata,
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to publish ingest metrics to Kafka")

    quality_report = evaluate_ingest_quality(
        aggregated_results,
        plan=ingest_config.plan,
        metadata=telemetry_metadata,
    )
    quality_payload = quality_report.as_dict()
    if quality_report.status is IngestQualityStatus.ok:
        logger.info(
            "🧪 Timescale ingest quality score=%.2f status=%s",
            quality_report.score,
            quality_report.status.value,
        )
    elif quality_report.status is IngestQualityStatus.warn:
        logger.warning(
            "🧪 Timescale ingest quality warnings: score=%.2f payload=%s",
            quality_report.score,
            quality_payload,
        )
    else:
        logger.error(
            "🧪 Timescale ingest quality failures: score=%.2f payload=%s",
            quality_report.score,
            quality_payload,
        )
    _publish_ingest_quality(event_bus, quality_payload)
    if kafka_quality_publisher is not None:
        try:
            kafka_quality_publisher.publish(
                quality_report,
                metadata=telemetry_metadata,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to publish ingest quality to Kafka")

    telemetry_metadata = dict(telemetry_metadata)
    telemetry_metadata["quality"] = {
        "status": quality_report.status.value,
        "score": quality_report.score,
    }

    _record_ingest_journal(
        ingest_config,
        aggregated_results,
        health_report,
        metadata=telemetry_metadata,
    )

    trend_snapshot: IngestTrendSnapshot | None = None
    retention_snapshot: DataRetentionSnapshot | None = None
    try:
        engine = ingest_config.timescale_settings.create_engine()
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug(
            "Failed to create Timescale engine for ingest trend analysis",
            exc_info=True,
        )
    else:
        try:
            if ingest_config.retention.enabled:
                try:
                    retention_snapshot = evaluate_data_retention(
                        engine,
                        _build_retention_policies(ingest_config.retention),
                        metadata={
                            "ingest_status": health_report.status.value,
                            "dimensions": _plan_dimensions(ingest_config.plan),
                        },
                    )
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to evaluate data retention window",
                        exc_info=True,
                    )
                else:
                    markdown = format_data_retention_markdown(retention_snapshot)
                    if retention_snapshot.status is RetentionStatus.fail:
                        logger.error("🗃️ Data retention failures:\n%s", markdown)
                    elif retention_snapshot.status is RetentionStatus.warn:
                        logger.warning("🗃️ Data retention warnings:\n%s", markdown)
                    else:
                        logger.info("🗃️ Data retention snapshot:\n%s", markdown)
                    publish_data_retention(event_bus, retention_snapshot)
                    telemetry_metadata = dict(telemetry_metadata)
                    telemetry_metadata["retention"] = retention_snapshot.as_dict()
                    if record_data_retention_snapshot is not None:
                        try:
                            record_data_retention_snapshot(retention_snapshot)
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to record data retention snapshot on runtime app",
                                exc_info=True,
                            )
            journal = TimescaleIngestJournal(engine)
            history = journal.fetch_recent(limit=50)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug(
                "Failed to load ingest history for trend evaluation",
                exc_info=True,
            )
        else:
            plan_metadata = dict(ingest_config.metadata or {})
            trend_snapshot = evaluate_ingest_trends(
                history,
                metadata={"plan": plan_metadata} if plan_metadata else None,
            )
            markdown = format_ingest_trends_markdown(trend_snapshot)
            if trend_snapshot.status is IngestTrendStatus.fail:
                logger.error("📈 Timescale ingest trend failures:\n%s", markdown)
            elif trend_snapshot.status is IngestTrendStatus.warn:
                logger.warning("📈 Timescale ingest trend warnings:\n%s", markdown)
            else:
                logger.info("📈 Timescale ingest trend snapshot:\n%s", markdown)
            publish_ingest_trends(event_bus, trend_snapshot)
            telemetry_metadata = dict(telemetry_metadata)
            telemetry_metadata["trends"] = trend_snapshot.as_dict()
            if record_ingest_trend_snapshot is not None:
                try:
                    record_ingest_trend_snapshot(trend_snapshot)
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to record ingest trend snapshot on runtime app",
                        exc_info=True,
                    )
        finally:
            try:
                engine.dispose()
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Failed to dispose Timescale engine", exc_info=True)

    report_payload = health_report.as_dict()
    if health_report.status is IngestHealthStatus.ok:
        logger.info("🩺 Timescale ingest health checks passed: %s", report_payload)
    elif health_report.status is IngestHealthStatus.warn:
        logger.warning("⚠️ Timescale ingest health warnings detected: %s", report_payload)
    else:
        logger.error("🚨 Timescale ingest health failures detected: %s", report_payload)

    _publish_ingest_health(event_bus, report_payload)
    if kafka_health_publisher is not None:
        try:
            kafka_health_publisher.publish(
                health_report,
                metadata=telemetry_metadata,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to publish ingest health to Kafka")

    failover_decision = decide_ingest_failover(
        health_report,
        plan=ingest_config.plan,
    )
    _publish_ingest_failover(event_bus, failover_decision)

    failover_drill_snapshot: FailoverDrillSnapshot | None = None
    if failover_drill_settings is not None and failover_drill_settings.enabled:
        drill_dimensions = (
            failover_drill_settings.dimensions or failover_decision.planned_dimensions
        )
        fallback_callable = fallback if failover_drill_settings.run_fallback else None
        try:
            failover_drill_snapshot = await execute_failover_drill(
                plan=ingest_config.plan,
                results=aggregated_results,
                fail_dimensions=drill_dimensions,
                scenario=failover_drill_settings.label,
                fallback=fallback_callable,
                metadata=telemetry_metadata,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failover drill execution failed")
        else:
            markdown = format_failover_drill_markdown(failover_drill_snapshot)
            if failover_drill_snapshot.status is FailoverDrillStatus.fail:
                logger.error("🧨 Timescale failover drill failed:\n%s", markdown)
            elif failover_drill_snapshot.status is FailoverDrillStatus.warn:
                logger.warning("🧨 Timescale failover drill warnings:\n%s", markdown)
            else:
                logger.info("🧨 Timescale failover drill snapshot:\n%s", markdown)
            _publish_failover_drill(event_bus, failover_drill_snapshot)
            telemetry_metadata = dict(telemetry_metadata)
            telemetry_metadata["failover_drill"] = failover_drill_snapshot.as_dict()
            if record_failover_drill_snapshot is not None:
                try:
                    record_failover_drill_snapshot(failover_drill_snapshot)
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to record failover drill snapshot on runtime app",
                        exc_info=True,
                    )

    kafka_snapshot: KafkaReadinessSnapshot | None = None
    kafka_readiness_settings = ingest_config.kafka_readiness
    if kafka_readiness_settings.enabled:
        try:
            kafka_snapshot = evaluate_kafka_readiness(
                generated_at=report_generated_at,
                settings=kafka_readiness_settings,
                connection=kafka_settings,
                topics=kafka_topics_tuple,
                provisioning=kafka_provisioning,
                publishers=kafka_publishers_tuple,
                lag_snapshot=kafka_lag_snapshot,
                metadata={"ingest_status": health_report.status.value},
            )
        except Exception:
            logger.exception("Failed to evaluate Kafka readiness")
        else:
            markdown = format_kafka_readiness_markdown(kafka_snapshot)
            if kafka_snapshot.status is KafkaReadinessStatus.fail:
                logger.error("📡 Kafka readiness failures:\n%s", markdown)
            elif kafka_snapshot.status is KafkaReadinessStatus.warn:
                logger.warning("📡 Kafka readiness warnings:\n%s", markdown)
            else:
                logger.info("📡 Kafka readiness snapshot:\n%s", markdown)
            publish_kafka_readiness(event_bus, kafka_snapshot)
            telemetry_metadata = dict(telemetry_metadata)
            telemetry_metadata["kafka_readiness"] = kafka_snapshot.as_dict()
            if record_kafka_readiness_snapshot is not None:
                try:
                    record_kafka_readiness_snapshot(kafka_snapshot)
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to record Kafka readiness snapshot on runtime app",
                        exc_info=True,
                    )

    cross_region_snapshot: CrossRegionFailoverSnapshot | None = None
    cross_region_settings = ingest_config.cross_region
    if cross_region_settings is not None and cross_region_settings.enabled:
        replica_records: dict[str, TimescaleIngestRunRecord] = {}
        replica_error: str | None = None
        replica_settings = cross_region_settings.replica_settings
        if replica_settings is None or not replica_settings.configured:
            replica_error = "replica_not_configured"
        else:
            try:
                replica_engine = replica_settings.create_engine()
            except Exception:
                logger.exception(
                    "Failed to initialise replica Timescale engine for cross-region evaluation"
                )
                replica_error = "replica_engine_error"
            else:
                try:
                    replica_journal = TimescaleIngestJournal(replica_engine)
                    replica_records = replica_journal.fetch_latest_by_dimension(
                        cross_region_settings.dimensions or tuple(aggregated_results.keys())
                    )
                except Exception:
                    logger.exception(
                        "Failed to fetch replica ingest history for cross-region evaluation"
                    )
                    replica_error = "replica_history_error"
                finally:
                    try:
                        replica_engine.dispose()
                    except Exception:  # pragma: no cover - diagnostics only
                        logger.debug("Failed to dispose replica Timescale engine", exc_info=True)

        generated_at = report_generated_at

        schedule_metadata = {
            "schedule_enabled": ingest_config.metadata.get("schedule_enabled"),
            "schedule_interval_seconds": ingest_config.metadata.get("schedule_interval_seconds"),
        }

        try:
            cross_region_snapshot = evaluate_cross_region_failover(
                generated_at=generated_at,
                settings=cross_region_settings,
                primary_results=aggregated_results,
                replica_records=replica_records,
                scheduler_snapshot=scheduler_snapshot,
                schedule_metadata=schedule_metadata,
                failover_snapshot=failover_drill_snapshot,
                replica_error=replica_error,
                metadata={"ingest_status": health_report.status.value},
            )
        except Exception:
            logger.exception("Failed to evaluate cross-region failover readiness")
        else:
            markdown = format_cross_region_markdown(cross_region_snapshot)
            if cross_region_snapshot.status is CrossRegionStatus.fail:
                logger.error("🌐 Cross-region failover readiness failed:\n%s", markdown)
            elif cross_region_snapshot.status is CrossRegionStatus.warn:
                logger.warning("🌐 Cross-region failover readiness warnings:\n%s", markdown)
            else:
                logger.info("🌐 Cross-region failover readiness:\n%s", markdown)
            publish_cross_region_snapshot(event_bus, cross_region_snapshot)
            telemetry_metadata = dict(telemetry_metadata)
            telemetry_metadata["cross_region"] = cross_region_snapshot.as_dict()
            if record_cross_region_snapshot is not None:
                try:
                    record_cross_region_snapshot(cross_region_snapshot)
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to record cross-region snapshot on runtime app",
                        exc_info=True,
                    )

    observability_snapshot = build_ingest_observability_snapshot(
        metrics_snapshot,
        health_report,
        failover=failover_decision,
        recovery=last_recovery_recommendation,
        metadata=telemetry_metadata,
    )
    logger.info(
        "🛡️ Timescale ingest observability snapshot:\n%s",
        observability_snapshot.to_markdown(),
    )
    _publish_ingest_observability(event_bus, observability_snapshot.as_dict())

    spark_snapshot: SparkExportSnapshot | None = None
    spark_stress_snapshot: SparkStressSnapshot | None = None
    plan = ingest_config.spark_export
    if plan is not None:
        try:
            engine = ingest_config.timescale_settings.create_engine()
        except Exception:
            logger.exception("Failed to create engine for Spark export plan")
            spark_snapshot = SparkExportSnapshot(
                generated_at=datetime.now(tz=UTC),
                status=SparkExportStatus.fail,
                format=plan.format,
                root_path=str(plan.root_path),
                jobs=tuple(),
                metadata={"error": "engine_initialisation_failed"},
            )
        else:
            try:
                reader = TimescaleReader(engine)
                spark_snapshot = execute_spark_export_plan(reader, plan)
            except Exception:
                logger.exception("Spark export execution failed")
                spark_snapshot = SparkExportSnapshot(
                    generated_at=datetime.now(tz=UTC),
                    status=SparkExportStatus.fail,
                    format=plan.format,
                    root_path=str(plan.root_path),
                    jobs=tuple(),
                    metadata={"error": "execution_failed"},
                )
            finally:
                try:
                    engine.dispose()
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to dispose Timescale engine after Spark export", exc_info=True
                    )

        if spark_snapshot is not None:
            markdown = spark_snapshot.to_markdown()
            if spark_snapshot.status is SparkExportStatus.fail:
                logger.error("✨ Spark export snapshot failed:\n%s", markdown)
            elif spark_snapshot.status is SparkExportStatus.warn:
                logger.warning("✨ Spark export snapshot warnings:\n%s", markdown)
            else:
                logger.info("✨ Spark export snapshot:\n%s", markdown)

            if plan.publish_telemetry:
                _publish_spark_exports(event_bus, spark_snapshot)

            telemetry_metadata = dict(telemetry_metadata)
            telemetry_metadata["spark_export"] = spark_snapshot.as_dict()

            if record_spark_snapshot is not None:
                try:
                    record_spark_snapshot(spark_snapshot)
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to record spark export snapshot on runtime app",
                        exc_info=True,
                    )

        if spark_stress_settings is not None and spark_stress_settings.enabled:
            try:
                engine = ingest_config.timescale_settings.create_engine()
            except Exception:
                logger.exception("Failed to create engine for Spark stress drill")
                spark_stress_snapshot = SparkStressSnapshot(
                    label=spark_stress_settings.label,
                    status=SparkStressStatus.fail,
                    generated_at=datetime.now(tz=UTC),
                    cycles=tuple(),
                    metadata={
                        "error": "engine_initialisation_failed",
                        "cycles": spark_stress_settings.cycles,
                        "plan_dimensions": [job.dimension for job in plan.jobs],
                    },
                )
            else:
                try:
                    reader = TimescaleReader(engine)

                    def _run_export() -> SparkExportSnapshot:
                        return execute_spark_export_plan(reader, plan)

                    stress_metadata = {
                        "plan_dimensions": [job.dimension for job in plan.jobs],
                        "root_path": str(plan.root_path),
                        "format": plan.format.value,
                    }
                    spark_stress_snapshot = execute_spark_stress_drill(
                        label=spark_stress_settings.label,
                        cycles=spark_stress_settings.cycles,
                        runner=_run_export,
                        warn_after_seconds=spark_stress_settings.warn_after_seconds,
                        fail_after_seconds=spark_stress_settings.fail_after_seconds,
                        metadata=stress_metadata,
                    )
                except Exception:
                    logger.exception("Spark stress drill execution failed")
                    spark_stress_snapshot = SparkStressSnapshot(
                        label=spark_stress_settings.label,
                        status=SparkStressStatus.fail,
                        generated_at=datetime.now(tz=UTC),
                        cycles=tuple(),
                        metadata={
                            "error": "execution_failed",
                            "cycles": spark_stress_settings.cycles,
                            "plan_dimensions": [job.dimension for job in plan.jobs],
                        },
                    )
                finally:
                    try:
                        engine.dispose()
                    except Exception:  # pragma: no cover - defensive logging
                        logger.debug(
                            "Failed to dispose Timescale engine after Spark stress drill",
                            exc_info=True,
                        )

        if spark_stress_snapshot is not None:
            markdown = format_spark_stress_markdown(spark_stress_snapshot)
            if spark_stress_snapshot.status is SparkStressStatus.fail:
                logger.error("🔥 Spark stress drill failed:\n%s", markdown)
            elif spark_stress_snapshot.status is SparkStressStatus.warn:
                logger.warning("🔥 Spark stress drill warnings:\n%s", markdown)
            else:
                logger.info("🔥 Spark stress drill snapshot:\n%s", markdown)

            _publish_spark_stress(event_bus, spark_stress_snapshot)

            telemetry_metadata = dict(telemetry_metadata)
            telemetry_metadata["spark_stress"] = spark_stress_snapshot.as_dict()

            if record_spark_stress_snapshot is not None:
                try:
                    record_spark_stress_snapshot(spark_stress_snapshot)
                except Exception:  # pragma: no cover - diagnostics only
                    logger.debug(
                        "Failed to record spark stress snapshot on runtime app",
                        exc_info=True,
                    )

    slo_snapshot = evaluate_ingest_slos(
        metrics_snapshot if metrics_snapshot.dimensions else None,
        health_report,
        alert_routes=ingest_config.operational_alert_routes,
        metadata={
            "recovery_attempts": len(recovery_steps),
            "recovery_enabled": ingest_config.recovery.enabled,
            "recovery_status": health_report.status.value,
        },
    )
    logger.info(
        "📊 Timescale ingest SLO snapshot:\n%s",
        slo_snapshot.to_markdown(),
    )
    _publish_operational_slos(event_bus, slo_snapshot)

    backup_settings = ingest_config.backup
    backup_policy = BackupPolicy(
        enabled=backup_settings.enabled,
        expected_frequency_seconds=backup_settings.expected_frequency_seconds,
        retention_days=backup_settings.retention_days,
        minimum_retention_days=backup_settings.minimum_retention_days,
        warn_after_seconds=backup_settings.warn_after_seconds,
        fail_after_seconds=backup_settings.fail_after_seconds,
        restore_test_interval_days=backup_settings.restore_test_interval_days,
        providers=backup_settings.providers,
        storage_location=backup_settings.storage_location,
    )
    backup_state = BackupState(
        last_backup_at=backup_settings.last_backup_at,
        last_backup_status=backup_settings.last_backup_status,
        last_restore_test_at=backup_settings.last_restore_test_at,
        last_restore_status=backup_settings.last_restore_status,
        recorded_failures=backup_settings.recorded_failures,
    )
    backup_metadata: dict[str, object] = {
        "health": health_report.as_dict(),
        "quality": quality_report.as_dict(),
        "failover": failover_decision.as_dict(),
    }
    if metrics_snapshot.dimensions:
        backup_metadata["metrics"] = metrics_snapshot.as_dict()

    backup_snapshot = evaluate_backup_readiness(
        backup_policy,
        backup_state,
        service="timescale_backups",
        metadata=backup_metadata,
    )
    markdown = format_backup_markdown(backup_snapshot)
    if backup_snapshot.status is BackupStatus.fail:
        logger.error("💾 Timescale backup readiness failures:\n%s", markdown)
    elif backup_snapshot.status is BackupStatus.warn:
        logger.warning("💾 Timescale backup readiness warnings:\n%s", markdown)
    else:
        logger.info("💾 Timescale backup readiness snapshot:\n%s", markdown)
    _publish_backup_snapshot(event_bus, backup_snapshot)

    if managed_manifest_tuple:
        telemetry_metadata = dict(telemetry_metadata)
        telemetry_metadata["managed_connectors"] = list(managed_manifest_tuple)

    backbone_snapshot = evaluate_data_backbone_readiness(
        ingest_config=ingest_config,
        health_report=health_report,
        quality_report=quality_report,
        metrics_snapshot=metrics_snapshot,
        failover_decision=failover_decision,
        recovery_recommendation=last_recovery_recommendation,
        backup_snapshot=backup_snapshot,
        context=backbone_context,
        metadata=telemetry_metadata,
        spark_snapshot=spark_snapshot,
        spark_stress_snapshot=spark_stress_snapshot,
        task_snapshots=task_snapshots,
    )
    logger.info(
        "🏗️ Data backbone readiness snapshot:\n%s",
        backbone_snapshot.to_markdown(),
    )
    _publish_data_backbone_readiness(event_bus, backbone_snapshot)
    if record_backbone_snapshot is not None:
        try:
            record_backbone_snapshot(backbone_snapshot)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug(
                "Failed to record data backbone snapshot on runtime app",
                exc_info=True,
            )

    ingest_plan = _normalise_ingest_plan_metadata(
        backbone_snapshot.metadata.get("plan")
    )

    readiness_metadata = {
        "recovery_attempts": len(recovery_steps),
        "ingest_plan": ingest_plan,
    }
    professional_snapshot = evaluate_professional_readiness(
        backbone_snapshot=backbone_snapshot,
        backup_snapshot=backup_snapshot,
        slo_snapshot=slo_snapshot,
        failover_decision=failover_decision,
        recovery_recommendation=last_recovery_recommendation,
        metadata=readiness_metadata,
    )
    logger.info(
        "🏢 Professional readiness snapshot:\n%s",
        professional_snapshot.to_markdown(),
    )
    _publish_professional_readiness(event_bus, professional_snapshot)
    if record_professional_snapshot is not None:
        try:
            record_professional_snapshot(professional_snapshot)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug(
                "Failed to record professional readiness snapshot on runtime app",
                exc_info=True,
            )

    if failover_decision.optional_triggers:
        logger.warning(
            "⚠️ Optional Timescale ingest slices degraded: %s",
            list(failover_decision.optional_triggers),
        )

    if failover_decision.should_failover:
        logger.error(
            "🔁 Falling back to DuckDB ingest due to Timescale health failure: %s",
            failover_decision.reason,
        )
        if fallback is not None:
            await fallback()
        return False, backup_snapshot

    return True, backup_snapshot


def _build_bootstrap_workload(
    app: ProfessionalPredatorApp,
    *,
    symbols_csv: str,
    db_path: str,
    reason: str | None = None,
    restart_policy: WorkloadRestartPolicy,
) -> RuntimeWorkload:
    metadata: dict[str, object] = {
        "mode": app.config.data_backbone_mode.value,
        "tier": app.config.tier.value,
    }
    if reason:
        metadata["reason"] = reason
    metadata["workload_kind"] = "data_backbone"
    metadata["supervised_components"] = ("data_backbone", "drift_monitor")

    async def _run_bootstrap() -> None:
        if reason:
            logger.info("🔄 Falling back to Tier-0 ingest: %s", reason)
        await _run_tier0_ingest(app, symbols_csv=symbols_csv, db_path=db_path)

    return RuntimeWorkload(
        name="tier0-ingest",
        factory=_run_bootstrap,
        description="Tier-0 DuckDB ingest",
        metadata=metadata,
        restart_policy=restart_policy,
    )


def _build_skip_workload(reason: str) -> RuntimeWorkload:
    metadata = {
        "reason": reason,
        "workload_kind": "data_backbone",
        "supervised_components": (),
    }

    async def _skip() -> None:
        logger.info("⏭️ Ingest skipped: %s", reason)

    return RuntimeWorkload(
        name="skip-ingest",
        factory=_skip,
        description="Ingest disabled for this run",
        metadata=metadata,
    )


def _parse_symbols_csv(symbols_csv: str) -> list[str]:
    return [s.strip() for s in symbols_csv.split(",") if s.strip()]


def _coerce_bool(raw: object, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    normalised = str(raw).strip().lower()
    if normalised in {"1", "true", "yes", "y", "on"}:
        return True
    if normalised in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_int(raw: object, default: int) -> int:
    if isinstance(raw, int):
        return raw
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _coerce_float(raw: object, default: float) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _coerce_log_level(raw: object, default: int) -> int:
    if isinstance(raw, int):
        return raw
    if raw is None:
        return default
    candidate = logging.getLevelName(str(raw).strip().upper())
    return candidate if isinstance(candidate, int) else default


@dataclass(frozen=True)
class _ResolvedRestartPolicy:
    policy: WorkloadRestartPolicy
    max_overridden: bool
    backoff_overridden: bool


def _resolve_workload_restart_policy(
    extras: Mapping[str, object],
    *,
    component: str,
    default_max_restarts: int | None,
    default_backoff: float,
) -> _ResolvedRestartPolicy:
    """Resolve restart policy configuration for a runtime workload."""

    key_prefix = f"RUNTIME_{component.upper()}"
    max_key = f"{key_prefix}_MAX_RESTARTS"
    backoff_key = f"{key_prefix}_RESTART_BACKOFF_SECONDS"

    max_restarts = default_max_restarts
    max_overridden = False
    raw_max = extras.get(max_key)
    if raw_max is not None:
        text = str(raw_max).strip()
        lowered = text.lower()
        if lowered in {"none", "null", "unlimited"}:
            max_restarts = None
            max_overridden = True
        elif text == "":
            pass
        else:
            try:
                candidate = int(text)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid %s value %r; using default %s",
                    max_key,
                    raw_max,
                    default_max_restarts,
                )
            else:
                if candidate < 0:
                    logger.warning(
                        "Negative %s value %r; using default %s",
                        max_key,
                        raw_max,
                        default_max_restarts,
                    )
                else:
                    max_restarts = candidate
                    max_overridden = True

    backoff_seconds = default_backoff
    backoff_overridden = False
    raw_backoff = extras.get(backoff_key)
    if raw_backoff is not None:
        text = str(raw_backoff).strip()
        if text == "":
            pass
        else:
            try:
                candidate_backoff = float(text)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid %s value %r; using default %.2f",
                    backoff_key,
                    raw_backoff,
                    default_backoff,
                )
            else:
                if candidate_backoff < 0:
                    logger.warning(
                        "Negative %s value %r; using default %.2f",
                        backoff_key,
                        raw_backoff,
                        default_backoff,
                    )
                else:
                    backoff_seconds = candidate_backoff
                    backoff_overridden = True

    return _ResolvedRestartPolicy(
        policy=WorkloadRestartPolicy(
            max_restarts=max_restarts,
            backoff_seconds=backoff_seconds,
        ),
        max_overridden=max_overridden,
        backoff_overridden=backoff_overridden,
    )


def _configure_runtime_logging(config: SystemConfig) -> None:
    extras = config.extras or {}
    if not _coerce_bool(extras.get("RUNTIME_LOG_STRUCTURED"), False):
        return

    level = _coerce_log_level(extras.get("RUNTIME_LOG_LEVEL"), logging.INFO)
    static_fields: dict[str, object] = {
        "runtime.tier": config.tier.value,
        "runtime.environment": config.environment.value,
        "runtime.run_mode": config.run_mode.value,
        "runtime.connection_protocol": config.connection_protocol.value,
        "runtime.data_backbone_mode": config.data_backbone_mode.value,
    }

    raw_context = extras.get("RUNTIME_LOG_CONTEXT")
    if raw_context:
        try:
            parsed = json.loads(raw_context)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse RUNTIME_LOG_CONTEXT JSON: %s", exc)
        else:
            if isinstance(parsed, MappingABC):
                for key, value in parsed.items():
                    static_fields[str(key)] = value
            else:
                logger.warning(
                    "RUNTIME_LOG_CONTEXT must be a JSON object, got %s",
                    type(parsed).__name__,
                )

    otel_settings: OpenTelemetryLoggingSettings | None = None
    if _coerce_bool(extras.get("RUNTIME_LOG_OTEL_ENABLED"), False):
        config_hint = extras.get("RUNTIME_LOG_OTEL_CONFIG") or "config/observability/logging.yaml"
        config_path = Path(str(config_hint))
        try:
            loaded_settings = load_opentelemetry_logging_settings(config_path)
        except FileNotFoundError:
            logger.warning(
                "OpenTelemetry logging requested but configuration file was not found",
                extra={"otel.config_path": str(config_path)},
            )
        except ValueError as exc:
            logger.warning(
                "Failed to parse OpenTelemetry logging configuration: %s",
                exc,
                extra={"otel.config_path": str(config_path)},
            )
        else:
            if not loaded_settings.enabled:
                loaded_settings = replace(loaded_settings, enabled=True)
            otel_settings = loaded_settings

    handler = configure_structured_logging(
        component="professional_runtime",
        level=level,
        static_fields=static_fields,
        otel_settings=otel_settings,
    )
    logger.info(
        "Structured logging enabled for professional runtime",
        extra={
            "logging.level": logging.getLevelName(level),
            "logging.handler": getattr(handler, "name", ""),
        },
    )
    if otel_settings and otel_settings.enabled:
        logger.info(
            "OpenTelemetry log export enabled",
            extra={
                "otel.endpoint": otel_settings.endpoint,
                "otel.compression": otel_settings.compression or "none",
                "otel.insecure": otel_settings.insecure,
            },
        )


def _configure_drift_monitor(
    runtime_app: RuntimeApplication,
    app: ProfessionalPredatorApp,
    tracer: RuntimeTracer,
    extras: Mapping[str, object],
) -> None:
    """Register a continuous sensory drift monitor workload when enabled."""

    sensory_component = getattr(app, "sensory_organ", None)
    if sensory_component is None:
        logger.debug("Drift monitor disabled: sensory organ not attached")
        return

    enabled = _coerce_bool(extras.get("RUNTIME_DRIFT_MONITOR_ENABLED"), True)
    if not enabled:
        logger.debug("Drift monitor disabled via configuration flag")
        return

    interval_seconds = max(1.0, _coerce_float(extras.get("RUNTIME_DRIFT_MONITOR_INTERVAL_SECONDS"), 120.0))

    workload_metadata = {
        "interval_seconds": interval_seconds,
        "workload_kind": "drift_monitor",
        "supervised_components": ("drift_monitor",),
    }

    async def _run_drift_monitor() -> None:
        logger.info("🛰️ Sensory drift monitor active (interval=%ss)", interval_seconds)
        try:
            while True:
                try:
                    component = getattr(app, "sensory_organ", None)
                    if component is None:
                        logger.debug("Drift monitor waiting for sensory organ attachment")
                        await asyncio.sleep(interval_seconds)
                        continue

                    status_method = getattr(component, "status", None)
                    if not callable(status_method):
                        logger.debug(
                            "Sensory component %s does not expose status(); skipping drift sample",
                            component.__class__.__name__,
                        )
                        await asyncio.sleep(interval_seconds)
                        continue

                    with tracer.operation_span(
                        name="runtime.drift_monitor",
                        metadata={"interval_seconds": interval_seconds},
                    ):
                        try:
                            sensory_status = status_method()
                        except Exception:
                            logger.debug(
                                "Failed to capture sensory status during drift monitor iteration",
                                exc_info=True,
                            )
                            await asyncio.sleep(interval_seconds)
                            continue

                        audit_entries = _process_sensory_status(app, sensory_status)
                        _evaluate_and_publish_drift(
                            app,
                            audit_entries,
                            metadata={
                                "source": "runtime.drift_monitor",
                                "interval_seconds": interval_seconds,
                            },
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Unexpected error during sensory drift monitor iteration",
                        exc_info=True,
                    )
                await asyncio.sleep(interval_seconds)
        finally:
            logger.info("🛰️ Sensory drift monitor stopped")

    runtime_app.add_auxiliary_workload(
        RuntimeWorkload(
            name="sensory-drift-monitor",
            factory=_run_drift_monitor,
            description="Continuous sensory drift evaluation loop",
            metadata=workload_metadata,
            restart_policy=WorkloadRestartPolicy(
                max_restarts=None,
                backoff_seconds=max(5.0, min(interval_seconds, 60.0)),
            ),
        )
    )


def _configure_governance_cadence(
    runtime_app: RuntimeApplication,
    app: ProfessionalPredatorApp,
    tracer: RuntimeTracer,
    extras: Mapping[str, str],
) -> None:
    """Attach the governance cadence runner when extras request it."""

    enabled = _coerce_bool(extras.get("GOVERNANCE_CADENCE_ENABLED"), False)
    if not enabled:
        return

    interval_seconds = _coerce_int(extras.get("GOVERNANCE_CADENCE_INTERVAL_SECONDS"), 86_400)
    if interval_seconds <= 0:
        logger.warning("Governance cadence disabled due to non-positive interval", extra={"cadence.interval": interval_seconds})
        return

    default_poll = max(60.0, min(interval_seconds / 4.0, 900.0))
    poll_seconds = _coerce_float(extras.get("GOVERNANCE_CADENCE_POLL_SECONDS"), default_poll)
    if poll_seconds <= 0:
        poll_seconds = default_poll

    report_path_hint = extras.get("GOVERNANCE_CADENCE_REPORT_PATH")
    report_path = Path(report_path_hint) if report_path_hint else Path("reports/governance.json")
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path

    base_path: Path | None = None
    base_path_hint = extras.get("GOVERNANCE_CONTEXT_BASE_DIR")
    if base_path_hint:
        candidate = Path(base_path_hint).expanduser()
        if not candidate.is_absolute():
            base_path = Path.cwd() / candidate
        else:
            base_path = candidate

    history_limit = max(0, _coerce_int(extras.get("GOVERNANCE_CADENCE_HISTORY_LIMIT"), 12))

    strategy_hint = (
        extras.get("GOVERNANCE_CADENCE_STRATEGY_ID")
        or extras.get("KYC_STRATEGY_ID")
        or extras.get("COMPLIANCE_STRATEGY_ID")
        or extras.get("STRATEGY_ID")
    )
    strategy_id = str(strategy_hint).strip() if strategy_hint else None

    metadata_entries: dict[str, object] = {
        "cadence_runner": "runtime.builder",
        "runtime_environment": app.config.environment.value,
        "runtime_tier": app.config.tier.value,
        "runtime_run_mode": app.config.run_mode.value,
        "cadence_poll_seconds": round(poll_seconds, 3),
    }
    prefix = "GOVERNANCE_CADENCE_META_"
    for key, value in extras.items():
        if key.startswith(prefix):
            meta_key = key[len(prefix) :].strip().lower()
            if meta_key:
                metadata_entries[meta_key] = value

    try:
        runner = build_governance_cadence_runner_from_config(
            event_bus=app.event_bus,
            config=app.config,
            report_path=report_path,
            interval=timedelta(seconds=interval_seconds),
            base_path=base_path,
            history_limit=history_limit,
            metadata=metadata_entries,
            strategy_id=strategy_id,
        )
    except Exception:
        logger.exception("Failed to initialise governance cadence runner")
        return

    original_compliance_provider = runner.compliance_provider
    original_regulatory_provider = runner.regulatory_provider

    def _compliance_provider() -> ComplianceReadinessSnapshot | Mapping[str, object] | None:
        snapshot = app.get_last_compliance_readiness_snapshot()
        if snapshot is not None:
            return snapshot
        return original_compliance_provider()

    def _regulatory_provider() -> RegulatoryTelemetrySnapshot | Mapping[str, object] | None:
        snapshot = app.get_last_regulatory_snapshot()
        if snapshot is not None:
            return snapshot
        return original_regulatory_provider()

    runner.compliance_provider = _compliance_provider
    runner.regulatory_provider = _regulatory_provider

    force_on_start = _coerce_bool(extras.get("GOVERNANCE_CADENCE_FORCE_ON_START"), False)
    poll_timeout = float(poll_seconds)

    cadence_task: asyncio.Task[object] | None = None
    cadence_stop_event: asyncio.Event | None = None
    runtime_app._governance_cadence_runner = runner  # type: ignore[attr-defined]

    async def _governance_cadence_loop(force_initial: bool) -> None:
        nonlocal cadence_stop_event
        assert cadence_stop_event is not None
        force_next = force_initial
        try:
            while True:
                if cadence_stop_event.is_set():
                    break
                reference = datetime.now(tz=UTC)
                metadata = {
                    "force": force_next,
                    "cadence_interval_seconds": interval_seconds,
                }
                with tracer.operation_span(
                    name="governance.cadence.run",
                    metadata=metadata,
                ):
                    try:
                        report = await asyncio.to_thread(
                            runner.run,
                            reference=reference,
                            force=force_next,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception("Governance cadence execution failed")
                    else:
                        if report is not None:
                            record_governance = getattr(app, "record_governance_report", None)
                            if callable(record_governance):
                                try:
                                    record_governance(report)
                                except Exception:
                                    logger.debug(
                                        "Failed to record governance report from cadence",
                                        exc_info=True,
                                    )
                force_next = False
                try:
                    await asyncio.wait_for(cadence_stop_event.wait(), timeout=poll_timeout)
                    break
                except asyncio.TimeoutError:
                    continue
        finally:
            if cadence_stop_event is not None:
                cadence_stop_event.set()

    async def _start_governance_cadence() -> None:
        nonlocal cadence_task, cadence_stop_event
        if cadence_task is not None:
            return
        cadence_stop_event = asyncio.Event()
        cadence_task = _supervise_background_task(
            runtime_app,
            _governance_cadence_loop(force_on_start),
            name="governance-cadence-runner",
            metadata={
                "component": "governance.cadence",
                "interval_seconds": interval_seconds,
            },
        )
        logger.info(
            "📑 Governance cadence runner enabled: interval=%ss poll=%.1fs path=%s",
            interval_seconds,
            poll_timeout,
            report_path,
        )

    async def _stop_governance_cadence() -> None:
        nonlocal cadence_task, cadence_stop_event
        if cadence_task is None:
            if hasattr(runtime_app, "_governance_cadence_runner"):
                delattr(runtime_app, "_governance_cadence_runner")
            return
        if cadence_stop_event is not None:
            cadence_stop_event.set()
        cadence_task.cancel()
        try:
            await cadence_task
        except asyncio.CancelledError:
            pass
        finally:
            cadence_task = None
            cadence_stop_event = None
            if hasattr(runtime_app, "_governance_cadence_runner"):
                delattr(runtime_app, "_governance_cadence_runner")
        logger.info("📑 Governance cadence runner stopped")

    runtime_app.add_startup_callback(_start_governance_cadence)
    runtime_app.add_shutdown_callback(_stop_governance_cadence)


def _resolve_system_validation_snapshot(
    extras: Mapping[str, str],
    metadata: Mapping[str, object],
) -> SystemValidationSnapshot | None:
    path_hint = extras.get("SYSTEM_VALIDATION_REPORT_PATH") or extras.get(
        "SYSTEM_VALIDATION_REPORT"
    )
    candidate_paths: list[Path] = []
    if path_hint:
        candidate_paths.append(Path(str(path_hint)))
    else:
        candidate_paths.append(Path("system_validation_report.json"))

    for candidate in candidate_paths:
        snapshot = load_system_validation_snapshot(candidate, metadata=metadata)
        if snapshot is not None:
            return snapshot
    return None


def build_professional_runtime_application(
    app: ProfessionalPredatorApp,
    *,
    skip_ingest: bool,
    symbols_csv: str,
    duckdb_path: str,
) -> RuntimeApplication:
    """Build a runtime application that coordinates ingest and trading workloads."""

    extras_mapping = app.config.extras or {}
    _configure_runtime_logging(app.config)
    configuration_metadata = {
        "tier": app.config.tier.value,
        "run_mode": app.config.run_mode.value,
        "environment": app.config.environment.value,
        "connection_protocol": app.config.connection_protocol.value,
        "data_backbone_mode": app.config.data_backbone_mode.value,
        "extras_count": len(extras_mapping),
    }

    configuration_journal: TimescaleConfigurationAuditJournal | None = None
    configuration_engine = None
    previous_configuration: Mapping[str, object] | None = None

    runtime_tracer_candidate = getattr(app, "runtime_tracer", None)
    if runtime_tracer_candidate is None:
        runtime_tracer: RuntimeTracer = NullRuntimeTracer()
    else:
        runtime_tracer = cast(RuntimeTracer, runtime_tracer_candidate)

    try:
        settings = TimescaleConnectionSettings.from_mapping(extras_mapping)
        if settings.configured:
            configuration_engine = settings.create_engine()
            TimescaleMigrator(configuration_engine).ensure_configuration_tables()
            configuration_journal = TimescaleConfigurationAuditJournal(configuration_engine)
            existing_record = configuration_journal.fetch_latest()
            if existing_record is not None:
                previous_configuration = existing_record.current_config
    except Exception:
        logger.debug("Failed to prepare configuration audit journal", exc_info=True)
        if configuration_engine is not None:
            try:
                configuration_engine.dispose()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to dispose configuration audit engine", exc_info=True)
            configuration_engine = None
            configuration_journal = None

    configuration_snapshot = evaluate_configuration_audit(
        app.config,
        previous=previous_configuration,
        metadata=configuration_metadata,
    )
    logger.info(
        "🧾 Configuration audit snapshot:\n%s",
        format_configuration_audit_markdown(configuration_snapshot),
    )
    publish_configuration_audit_snapshot(app.event_bus, configuration_snapshot)
    record_configuration = getattr(app, "record_configuration_snapshot", None)
    if callable(record_configuration):
        try:
            record_configuration(configuration_snapshot)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug(
                "Failed to record configuration snapshot on runtime app",
                exc_info=True,
            )

    if configuration_journal is not None:
        try:
            configuration_journal.record_snapshot(configuration_snapshot.as_dict())
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to persist configuration audit snapshot", exc_info=True)

    if configuration_engine is not None:
        try:
            configuration_engine.dispose()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to dispose configuration audit engine", exc_info=True)

    resolved_ingest_policy = _resolve_workload_restart_policy(
        extras_mapping,
        component="ingest",
        default_max_restarts=None,
        default_backoff=5.0,
    )
    ingest_restart_policy = resolved_ingest_policy.policy
    bootstrap_max_restarts = (
        ingest_restart_policy.max_restarts
        if resolved_ingest_policy.max_overridden
        else None
    )
    bootstrap_backoff = (
        ingest_restart_policy.backoff_seconds
        if resolved_ingest_policy.backoff_overridden
        else 2.0
    )
    bootstrap_ingest_restart_policy = WorkloadRestartPolicy(
        max_restarts=bootstrap_max_restarts,
        backoff_seconds=bootstrap_backoff,
    )

    resolved_trading_policy = _resolve_workload_restart_policy(
        extras_mapping,
        component="trading",
        default_max_restarts=None,
        default_backoff=2.0,
    )
    trading_restart_policy = resolved_trading_policy.policy

    ingestion: RuntimeWorkload | None

    if skip_ingest:
        ingestion = _build_skip_workload("skip-ingest flag enabled")
    else:
        backbone_mode = app.config.data_backbone_mode
        tier = app.config.tier

        if backbone_mode is DataBackboneMode.institutional and tier is EmpTier.tier_1:
            base_symbols = _parse_symbols_csv(symbols_csv)
            ingest_config = build_institutional_ingest_config(
                app.config, fallback_symbols=base_symbols
            )

            metadata = {
                "mode": backbone_mode.value,
                "tier": tier.value,
                "should_run": ingest_config.should_run,
                "reason": ingest_config.reason,
                "plan": dict(ingest_config.metadata),
            }
            metadata["workload_kind"] = "data_backbone"
            metadata["supervised_components"] = ("data_backbone", "drift_monitor")

            async def _run_institutional() -> None:
                plan_metadata: dict[str, object] = {
                    "ingest.should_run": ingest_config.should_run,
                    "ingest.plan_dimensions": len(_plan_dimensions(ingest_config.plan)),
                }
                if ingest_config.reason:
                    plan_metadata["ingest.reason"] = ingest_config.reason
                with runtime_tracer.operation_span(
                    name="ingest.plan_evaluation",
                    metadata=plan_metadata,
                ) as plan_span:
                    if not ingest_config.should_run:
                        logger.info(
                            "Timescale ingest skipped: %s",
                            ingest_config.reason or "no ingest slices configured",
                        )
                        if plan_span is not None and hasattr(plan_span, "set_attribute"):
                            plan_span.set_attribute("runtime.ingest.status", "skipped")
                        await _run_tier0_ingest(
                            app,
                            symbols_csv=symbols_csv,
                            db_path=duckdb_path,
                        )
                        return
                    if plan_span is not None and hasattr(plan_span, "set_attribute"):
                        plan_span.set_attribute("runtime.ingest.status", "scheduled")

                metadata_payload = {
                    key: value
                    for key, value in ingest_config.metadata.items()
                    if value not in (None, {}, [])
                }
                if metadata_payload:
                    logger.info("🏗️ Timescale ingest plan: %s", metadata_payload)

                kafka_settings = ingest_config.kafka_settings
                extras_mapping = app.config.extras or {}
                topic_specs = resolve_ingest_topic_specs(extras_mapping)
                kafka_topic_names = tuple(spec.name for spec in topic_specs)
                kafka_provisioning_summary: KafkaTopicProvisioningSummary | None = None
                auto_provision = should_auto_create_topics(extras_mapping)
                if topic_specs and auto_provision:
                    provisioner = KafkaTopicProvisioner(kafka_settings)
                    summary = provisioner.ensure_topics(topic_specs)
                    kafka_provisioning_summary = summary
                    if summary.notes:
                        logger.debug("Kafka topic provisioning notes: %s", list(summary.notes))
                    if summary.created:
                        logger.info("Kafka ingest topics created: %s", list(summary.created))
                    if summary.failed:
                        logger.error(
                            "Kafka ingest topic provisioning failures: %s",
                            summary.failed,
                        )
                    elif not summary.created and summary.existing:
                        logger.info(
                            "Kafka ingest topics already exist: %s",
                            list(summary.existing),
                        )
                elif topic_specs:
                    logger.debug(
                        "Kafka ingest topic auto-provisioning disabled; expected topics: %s",
                        list(kafka_topic_names),
                    )
                    kafka_provisioning_summary = KafkaTopicProvisioningSummary(
                        requested=kafka_topic_names,
                        notes=("auto_provision_disabled",),
                    )

                kafka_publisher = create_ingest_event_publisher(
                    kafka_settings,
                    extras_mapping,
                )
                kafka_health_publisher = create_ingest_health_publisher(
                    kafka_settings,
                    extras_mapping,
                )
                kafka_metrics_publisher = create_ingest_metrics_publisher(
                    kafka_settings,
                    extras_mapping,
                )
                kafka_quality_publisher = create_ingest_quality_publisher(
                    kafka_settings,
                    extras_mapping,
                )
                if kafka_settings.configured:
                    kafka_summary_text = kafka_settings.summary(redacted=True)
                    if kafka_publisher is None:
                        logger.warning(
                            "Kafka ingest configuration detected but publisher disabled; check topics or dependencies (%s)",
                            kafka_summary_text,
                        )
                    else:
                        logger.info("Kafka ingest publisher ready: %s", kafka_summary_text)
                    if kafka_health_publisher is None:
                        logger.info(
                            "Kafka ingest health publisher not configured; set KAFKA_INGEST_HEALTH_TOPIC to enable",
                        )
                    else:
                        logger.info(
                            "Kafka ingest health publisher ready: %s",
                            kafka_health_publisher.summary(),
                        )
                    if kafka_metrics_publisher is None:
                        logger.info(
                            "Kafka ingest metrics publisher not configured; set KAFKA_INGEST_METRICS_TOPIC to enable",
                        )
                    else:
                        logger.info(
                            "Kafka ingest metrics publisher ready: %s",
                            kafka_metrics_publisher.summary(),
                        )
                    if kafka_quality_publisher is None:
                        logger.info(
                            "Kafka ingest quality publisher not configured; set KAFKA_INGEST_QUALITY_TOPIC to enable",
                        )
                    else:
                        logger.info(
                            "Kafka ingest quality publisher ready: %s",
                            kafka_quality_publisher.summary(),
                        )

                event_bus_publisher = EventBusIngestPublisher(app.event_bus)
                publisher = combine_ingest_publishers(
                    event_bus_publisher,
                    kafka_publisher,
                )

                redis_cache_candidate = getattr(app, "redis_client", None)
                managed_cache = (
                    redis_cache_candidate
                    if isinstance(redis_cache_candidate, ManagedRedisCache)
                    else None
                )

                data_manager: RealDataManager | None = None
                try:
                    data_manager = RealDataManager(
                        system_config=app.config,
                        extras=extras_mapping,
                        timescale_settings=ingest_config.timescale_settings,
                        redis_settings=ingest_config.redis_settings,
                        cache_policy=ingest_config.redis_policy,
                        kafka_settings=ingest_config.kafka_settings,
                        ingest_publisher=publisher,
                        managed_cache=managed_cache,
                        task_supervisor=app.task_supervisor,
                        require_timescale=True,
                        require_redis=True,
                        require_kafka=True,
                    )
                except Exception:
                    logger.exception(
                        "Failed to initialise RealDataManager for operational ingest",
                    )
                    data_manager = None
                else:
                    async def _shutdown_real_data_manager() -> None:
                        await data_manager.shutdown()

                    app.add_cleanup_callback(_shutdown_real_data_manager)

                redis_settings = ingest_config.redis_settings
                services_holder: dict[str, InstitutionalIngestServices] = {}
                skip_next_scheduler_run = False

                def _redis_client_factory(settings: RedisConnectionSettings) -> object:
                    existing_client = getattr(app, "redis_client", None)
                    if isinstance(existing_client, ManagedRedisCache):
                        raw_client = existing_client.raw_client
                        if raw_client is not None:
                            return raw_client
                    client = configure_redis_client(settings)
                    if client is not None:
                        return client
                    logger.warning(
                        "Redis configuration present but ingest vertical falling back to in-memory cache",
                    )
                    return InMemoryRedis()

                async def _fallback() -> None:
                    fallback_metadata: dict[str, object] = {
                        "ingest.fallback.symbols": len(base_symbols),
                    }
                    if ingest_config.reason:
                        fallback_metadata["ingest.fallback.reason"] = ingest_config.reason
                    with runtime_tracer.operation_span(
                        name="ingest.tier0_fallback",
                        metadata=fallback_metadata,
                    ) as fallback_span:
                        try:
                            await _run_tier0_ingest(
                                app,
                                symbols_csv=symbols_csv,
                                db_path=duckdb_path,
                            )
                        except Exception:
                            if fallback_span is not None and hasattr(
                                fallback_span, "set_attribute"
                            ):
                                fallback_span.set_attribute(
                                    "runtime.ingest.fallback_status", "error"
                                )
                            raise
                        else:
                            if fallback_span is not None and hasattr(
                                fallback_span, "set_attribute"
                            ):
                                fallback_span.set_attribute(
                                    "runtime.ingest.fallback_status", "completed"
                                )

                async def _run_timescale_ingest() -> bool:
                    redis_client = getattr(app, "redis_client", None)
                    redis_namespace: str | None = None
                    redis_backing: str | None = None
                    redis_configured = False
                    cache_metrics: dict[str, object] = {}
                    cache_policy_details: dict[str, object] | None = None
                    if isinstance(redis_client, ManagedRedisCache):
                        redis_namespace = redis_client.policy.namespace
                        raw_client = redis_client.raw_client
                        cache_policy_details = {
                            "ttl_seconds": redis_client.policy.ttl_seconds,
                            "max_keys": redis_client.policy.max_keys,
                            "invalidate_prefixes": redis_client.policy.invalidate_prefixes,
                        }
                        try:
                            cache_metrics = dict(redis_client.metrics())
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug("Failed to collect cache metrics", exc_info=True)
                        if isinstance(raw_client, InMemoryRedis):
                            redis_backing = "in-memory"
                        else:
                            redis_backing = raw_client.__class__.__name__
                            redis_configured = True

                    kafka_publishers = tuple(
                        name
                        for name, active in (
                            ("events", kafka_publisher),
                            ("health", kafka_health_publisher),
                            ("metrics", kafka_metrics_publisher),
                            ("quality", kafka_quality_publisher),
                        )
                        if active is not None
                    )
                    raw_topics = ingest_config.metadata.get("kafka_topics", [])
                    if isinstance(raw_topics, (list, tuple)):
                        kafka_topics = tuple(str(topic) for topic in raw_topics)
                    else:
                        kafka_topics = tuple()
                    streaming_enabled = ingest_config.enable_streaming

                    services = services_holder.get("services")
                    scheduler_state = None
                    managed_manifest: tuple[dict[str, object], ...] = ()
                    if services is not None:
                        try:
                            scheduler_state = services.scheduler.state()
                        except Exception:  # pragma: no cover - defensive logging
                            logger.debug("Failed to capture scheduler state", exc_info=True)
                        try:
                            managed_manifest = tuple(
                                snapshot.as_dict() for snapshot in services.managed_manifest()
                            )
                        except Exception:  # pragma: no cover - defensive logging
                            logger.debug(
                                "Failed to build managed connector manifest for telemetry",
                                exc_info=True,
                            )

                    scheduler_snapshot = build_scheduler_snapshot(
                        enabled=services is not None,
                        schedule=ingest_config.schedule,
                        state=scheduler_state,
                    )
                    scheduler_metadata = scheduler_snapshot.as_dict()

                    backbone_context = BackboneRuntimeContext(
                        redis_expected=app.config.data_backbone_mode
                        is DataBackboneMode.institutional,
                        redis_configured=redis_configured,
                        redis_namespace=redis_namespace,
                        redis_backing=redis_backing,
                        kafka_expected=bool(
                            streaming_enabled
                            and (
                                ingest_config.metadata.get("kafka_configured")
                                or kafka_topics
                            )
                        ),
                        kafka_configured=kafka_settings.configured,
                        kafka_topics=kafka_topics,
                        kafka_publishers=kafka_publishers,
                        scheduler_enabled=services is not None,
                        scheduler_state=scheduler_state,
                    )
                    recorder = getattr(app, "record_data_backbone_snapshot", None)
                    validation_recorder = getattr(
                        app, "record_data_backbone_validation_snapshot", None
                    )
                    professional_recorder = getattr(
                        app, "record_professional_readiness_snapshot", None
                    )
                    spark_recorder = getattr(app, "record_spark_export_snapshot", None)
                    retention_recorder = getattr(app, "record_data_retention_snapshot", None)
                    spark_stress_recorder = getattr(app, "record_spark_stress_snapshot", None)
                    failover_drill_recorder = getattr(app, "record_failover_drill_snapshot", None)
                    trend_recorder = getattr(app, "record_ingest_trend_snapshot", None)
                    cross_region_recorder = getattr(app, "record_cross_region_snapshot", None)
                    kafka_readiness_recorder = getattr(app, "record_kafka_readiness_snapshot", None)

                    execution_metadata: dict[str, object] = {
                        "ingest.plan_dimensions": len(_plan_dimensions(ingest_config.plan)),
                        "ingest.kafka_publishers": len(kafka_publishers),
                        "ingest.kafka_topics": len(kafka_topic_names),
                        "ingest.redis_configured": redis_configured,
                        "ingest.kafka_streaming_enabled": streaming_enabled,
                    }
                    if kafka_provisioning_summary is not None:
                        created_topics = getattr(kafka_provisioning_summary, "created", ()) or ()
                        execution_metadata["ingest.kafka_provisioned"] = len(tuple(created_topics))
                    if managed_manifest:
                        execution_metadata["ingest.managed_connectors"] = len(managed_manifest)

                    snapshot_fn = getattr(app, "task_snapshots", None)
                    if callable(snapshot_fn):
                        try:
                            task_snapshots = snapshot_fn()
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug("Failed to collect task snapshots", exc_info=True)
                            task_snapshots = ()
                    else:
                        task_snapshots = ()

                    with runtime_tracer.operation_span(
                        name="ingest.timescale_execute",
                        metadata=execution_metadata,
                    ) as execution_span:
                        success, backup_snapshot = await _execute_timescale_ingest(
                            ingest_config=ingest_config,
                            event_bus=app.event_bus,
                            publisher=publisher,
                            kafka_health_publisher=kafka_health_publisher,
                            kafka_metrics_publisher=kafka_metrics_publisher,
                            kafka_quality_publisher=kafka_quality_publisher,
                            fallback=_fallback,
                            data_manager=data_manager,
                            backbone_context=backbone_context,
                            scheduler_snapshot=scheduler_snapshot,
                            record_backbone_validation_snapshot=validation_recorder,
                            record_backbone_snapshot=recorder,
                            record_professional_snapshot=professional_recorder,
                            record_spark_snapshot=spark_recorder,
                            record_data_retention_snapshot=retention_recorder,
                            spark_stress_settings=ingest_config.spark_stress,
                            record_spark_stress_snapshot=spark_stress_recorder,
                            failover_drill_settings=ingest_config.failover_drill,
                            record_failover_drill_snapshot=failover_drill_recorder,
                            record_ingest_trend_snapshot=trend_recorder,
                            record_cross_region_snapshot=cross_region_recorder,
                            kafka_settings=kafka_settings,
                            kafka_topics=kafka_topic_names,
                            kafka_publishers=kafka_publishers,
                            kafka_provisioning=kafka_provisioning_summary,
                            kafka_lag_snapshot=None,
                            record_kafka_readiness_snapshot=kafka_readiness_recorder,
                            managed_manifest=managed_manifest,
                            task_snapshots=task_snapshots,
                        )
                    if execution_span is not None and hasattr(execution_span, "set_attribute"):
                        execution_span.set_attribute("runtime.ingest.success", bool(success))
                        execution_span.set_attribute(
                            "runtime.ingest.backup_snapshot", backup_snapshot is not None
                        )
                    services = services_holder.get("services")
                    if services is not None:
                        try:
                            scheduler_state = services.scheduler.state()
                        except Exception:  # pragma: no cover - defensive logging
                            logger.debug("Failed to refresh scheduler state", exc_info=True)
                        else:
                            scheduler_snapshot = build_scheduler_snapshot(
                                enabled=True,
                                schedule=ingest_config.schedule,
                                state=scheduler_state,
                            )
                            scheduler_metadata = scheduler_snapshot.as_dict()

                    if data_manager is not None:
                        try:
                            cache_metrics = dict(data_manager.cache_metrics(reset=False))
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug("Failed to refresh cache metrics after ingest", exc_info=True)

                    if backup_snapshot is not None:
                        record_backup = getattr(app, "record_backup_snapshot", None)
                        if callable(record_backup):
                            try:
                                record_backup(backup_snapshot)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record backup snapshot on runtime app",
                                    exc_info=True,
                                )

                    trade_summary: Mapping[str, object] | None = None
                    kyc_summary: Mapping[str, object] | None = None
                    compliance_monitor = getattr(app, "compliance_monitor", None)
                    if compliance_monitor is not None:
                        try:
                            trade_summary = compliance_monitor.summary()
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to build trade compliance summary for readiness",
                                exc_info=True,
                            )
                    kyc_monitor = getattr(app, "kyc_monitor", None)
                    if kyc_monitor is not None:
                        try:
                            kyc_summary = kyc_monitor.summary()
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to build KYC summary for readiness",
                                exc_info=True,
                            )

                    strategy_registry_summary: Mapping[str, object] | None = None
                    strategy_registry = getattr(app, "strategy_registry", None)
                    get_registry_summary = getattr(
                        strategy_registry, "get_registry_summary", None
                    )
                    if callable(get_registry_summary):
                        try:
                            strategy_registry_summary = get_registry_summary()
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug("Failed to summarise strategy registry", exc_info=True)

                    if trade_summary is not None or kyc_summary is not None:
                        compliance_metadata: dict[str, object] = {
                            "ingest_success": success,
                            "monitors": {
                                "trade": trade_summary is not None,
                                "kyc": kyc_summary is not None,
                            },
                            "strategy_registry": strategy_registry_summary,
                        }
                        policy_workflow_snapshot = None
                        trading_manager = _locate_trading_manager(app)
                        if trading_manager is not None:
                            build_governance_snapshot = getattr(
                                trading_manager,
                                "build_policy_governance_snapshot",
                                None,
                            )
                            if callable(build_governance_snapshot):
                                try:
                                    policy_workflow_snapshot = build_governance_snapshot()
                                except Exception:  # pragma: no cover - diagnostics only
                                    logger.debug(
                                        "Failed to build policy governance workflow snapshot",
                                        exc_info=True,
                                    )
                        compliance_snapshot = evaluate_compliance_readiness(
                            trade_summary=trade_summary,
                            kyc_summary=kyc_summary,
                            metadata=compliance_metadata,
                        )
                        logger.info(
                            "⚖️ Compliance readiness snapshot:\n%s",
                            compliance_snapshot.to_markdown(),
                        )
                        publish_compliance_readiness(app.event_bus, compliance_snapshot)
                        record_compliance = getattr(
                            app, "record_compliance_readiness_snapshot", None
                        )
                        if callable(record_compliance):
                            try:
                                record_compliance(compliance_snapshot)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record compliance readiness snapshot on runtime app",
                                    exc_info=True,
                                )
                        workflow_metadata = dict(compliance_metadata)
                        workflow_metadata["readiness_status"] = compliance_snapshot.status.value
                        workflow_snapshot = evaluate_compliance_workflows(
                            trade_summary=trade_summary,
                            kyc_summary=kyc_summary,
                            strategy_registry=strategy_registry_summary,
                            policy_workflow_snapshot=policy_workflow_snapshot,
                            metadata=workflow_metadata,
                        )
                        logger.info(
                            "📋 Compliance workflow snapshot:\n%s",
                            workflow_snapshot.to_markdown(),
                        )
                        publish_compliance_workflows(app.event_bus, workflow_snapshot)
                        record_workflow = getattr(app, "record_compliance_workflow_snapshot", None)
                        if callable(record_workflow):
                            try:
                                record_workflow(workflow_snapshot)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record compliance workflow snapshot on runtime app",
                                    exc_info=True,
                                )

                        regulatory_signals = _build_regulatory_signals(
                            compliance_snapshot,
                            workflow_snapshot,
                        )
                        regulatory_metadata: dict[str, object] = {
                            "ingest_success": success,
                            "compliance_status": compliance_snapshot.status.value,
                        }
                        if workflow_snapshot is not None:
                            regulatory_metadata["workflow_status"] = workflow_snapshot.status.value
                        if strategy_registry_summary is not None:
                            regulatory_metadata["strategy_registry"] = strategy_registry_summary

                        regulatory_snapshot = evaluate_regulatory_telemetry(
                            signals=regulatory_signals,
                            metadata=regulatory_metadata,
                        )
                        missing_domains = ",".join(regulatory_snapshot.missing_domains) or "none"
                        logger.info(
                            "🛡️ Regulatory telemetry status=%s coverage=%.2f missing=%s",
                            regulatory_snapshot.status.value.upper(),
                            regulatory_snapshot.coverage_ratio,
                            missing_domains,
                        )
                        publish_regulatory_telemetry(app.event_bus, regulatory_snapshot)
                        record_regulatory = getattr(app, "record_regulatory_snapshot", None)
                        if callable(record_regulatory):
                            try:
                                record_regulatory(regulatory_snapshot)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record regulatory telemetry snapshot on runtime app",
                                    exc_info=True,
                                )

                        strategy_identifier = _resolve_strategy_identifier(extras_mapping)
                        try:
                            audit_evidence = collect_audit_evidence(
                                cfg,
                                strategy_id=strategy_identifier,
                            )
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.exception("Failed to collect audit evidence for governance report")
                            audit_evidence = {"metadata": {"errors": ["audit_collection_failed"]}}

                        governance_metadata: dict[str, object] = {
                            "ingest_success": success,
                            "compliance_status": compliance_snapshot.status.value,
                            "regulatory_status": regulatory_snapshot.status.value,
                            "regulatory_coverage": regulatory_snapshot.coverage_ratio,
                        }
                        if strategy_identifier is not None:
                            governance_metadata["strategy_id"] = strategy_identifier

                        governance_report = generate_governance_report(
                            compliance_readiness=compliance_snapshot,
                            regulatory_snapshot=regulatory_snapshot,
                            audit_evidence=audit_evidence,
                            metadata=governance_metadata,
                        )
                        section_names = ",".join(section.name for section in governance_report.sections)
                        logger.info(
                            "📑 Governance report status=%s sections=%s",
                            governance_report.status.value.upper(),
                            section_names,
                        )
                        publish_governance_report(app.event_bus, governance_report)
                        record_governance = getattr(app, "record_governance_report", None)
                        if callable(record_governance):
                            try:
                                record_governance(governance_report)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record governance report on runtime app",
                                    exc_info=True,
                                )

                    security_policy = SecurityPolicy.from_mapping(extras_mapping)
                    security_state = SecurityState.from_mapping(extras_mapping)
                    security_metadata: dict[str, object] = {
                        "ingest_success": success,
                        "redis": {
                            "configured": redis_configured,
                            "namespace": redis_namespace,
                            "backing": redis_backing,
                        },
                        "kafka": {
                            "configured": kafka_settings.configured,
                            "publishers": kafka_publishers,
                            "topics": list(kafka_topics),
                        },
                        "scheduler": scheduler_metadata,
                        "compliance": {
                            "trade_monitor_active": trade_summary is not None,
                            "kyc_monitor_active": kyc_summary is not None,
                        },
                    }
                    service_name = str(
                        extras_mapping.get("SECURITY_SERVICE_NAME") or "emp_platform"
                    )
                    security_snapshot = evaluate_security_posture(
                        security_policy,
                        security_state,
                        service=service_name,
                        metadata=security_metadata,
                    )
                    logger.info(
                        "🔐 Security posture snapshot:\n%s",
                        security_snapshot.to_markdown(),
                    )
                    publish_security_posture(app.event_bus, security_snapshot)
                    record_security = getattr(app, "record_security_snapshot", None)
                    if callable(record_security):
                        try:
                            record_security(security_snapshot)
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to record security snapshot on runtime app",
                                exc_info=True,
                            )

                    execution_stats: Mapping[str, object] | None = None
                    trading_manager = getattr(app.sensory_organ, "trading_manager", None)
                    get_execution_stats = getattr(trading_manager, "get_execution_stats", None)
                    if callable(get_execution_stats):
                        try:
                            execution_stats = get_execution_stats()
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to capture execution stats from trading manager",
                                exc_info=True,
                            )

                    drop_copy_metrics: Mapping[str, object] | None = None
                    drop_copy_active: bool | None = None
                    trade_app: Any | None = None
                    fix_manager = getattr(app, "fix_connection_manager", None)
                    if fix_manager is not None:
                        get_application = getattr(fix_manager, "get_application", None)
                        if callable(get_application):
                            try:
                                trade_app = get_application("trade")
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to resolve FIX trade application for execution telemetry",
                                    exc_info=True,
                                )
                                trade_app = None
                        else:
                            trade_app = None
                    if trade_app is not None:
                            try:
                                drop_copy_metrics = trade_app.get_queue_metrics()
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to capture drop-copy queue metrics",
                                    exc_info=True,
                                )
                            else:
                                drop_copy_active = True

                    execution_policy = ExecutionPolicy.from_mapping(extras_mapping)
                    execution_state = ExecutionState.from_sources(
                        mapping=extras_mapping,
                        stats=execution_stats,
                        drop_copy_metrics=drop_copy_metrics,
                        drop_copy_active=drop_copy_active,
                    )
                    execution_metadata: dict[str, object] = {
                        "ingest_success": success,
                        "scheduler": scheduler_metadata,
                        "compliance": {
                            "trade_monitor_active": trade_summary is not None,
                            "kyc_monitor_active": kyc_summary is not None,
                        },
                    }
                    if execution_stats:
                        execution_metadata["stats"] = dict(execution_stats)
                    if drop_copy_metrics:
                        execution_metadata["drop_copy"] = dict(drop_copy_metrics)

                    execution_service = str(
                        extras_mapping.get("EXECUTION_SERVICE_NAME") or "execution"
                    )
                    execution_snapshot = evaluate_execution_readiness(
                        execution_policy,
                        execution_state,
                        metadata=execution_metadata,
                        service=execution_service,
                    )
                    logger.info(
                        "🚀 Execution readiness snapshot:\n%s",
                        format_execution_markdown(execution_snapshot),
                    )
                    publish_execution_snapshot(app.event_bus, execution_snapshot)
                    record_execution = getattr(app, "record_execution_snapshot", None)
                    if callable(record_execution):
                        try:
                            record_execution(execution_snapshot)
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to record execution snapshot on runtime app",
                                exc_info=True,
                            )

                    incident_policy = IncidentResponsePolicy.from_mapping(extras_mapping)
                    incident_state = IncidentResponseState.from_mapping(extras_mapping)
                    incident_context: dict[str, object] = {
                        "ingest_success": success,
                        "execution_status": execution_snapshot.status.value,
                        "open_incidents": len(incident_state.open_incidents),
                    }
                    if security_snapshot is not None:
                        incident_context["security_status"] = security_snapshot.status.value
                    if scheduler_metadata:
                        incident_context["scheduler"] = dict(scheduler_metadata)

                    incident_service = str(
                        extras_mapping.get("INCIDENT_SERVICE_NAME") or "incident_response"
                    )
                    incident_snapshot = evaluate_incident_response(
                        incident_policy,
                        incident_state,
                        service=incident_service,
                        metadata=incident_context,
                    )
                    logger.info(
                        "🚨 Incident response snapshot:\n%s",
                        format_incident_response_markdown(incident_snapshot),
                    )
                    publish_incident_response_snapshot(app.event_bus, incident_snapshot)
                    record_incident = getattr(app, "record_incident_response_snapshot", None)
                    if callable(record_incident):
                        try:
                            record_incident(incident_snapshot)
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to record incident response snapshot on runtime app",
                                exc_info=True,
                            )

                    fix_pilot = getattr(app, "fix_pilot", None)
                    if fix_pilot is not None:
                        pilot_state = fix_pilot.snapshot()
                        pilot_policy = FixPilotPolicy.from_mapping(extras_mapping)
                        pilot_metadata: dict[str, object] = {
                            "ingest_success": success,
                            "execution": dict(execution_metadata),
                            "active_orders": pilot_state.active_orders,
                            "dropcopy_running": pilot_state.dropcopy_running,
                            "dropcopy_backlog": pilot_state.dropcopy_backlog,
                        }
                        if pilot_state.queue_metrics:
                            pilot_metadata["queue_metrics"] = {
                                name: dict(metrics)
                                for name, metrics in pilot_state.queue_metrics.items()
                            }
                        if pilot_state.dropcopy_reconciliation:
                            pilot_metadata["dropcopy"] = dict(pilot_state.dropcopy_reconciliation)
                        fix_snapshot = evaluate_fix_pilot(
                            pilot_policy,
                            pilot_state,
                            metadata=pilot_metadata,
                        )
                        logger.info(
                            "📡 FIX pilot snapshot:\n%s",
                            format_fix_pilot_markdown(fix_snapshot),
                        )
                        publish_fix_pilot_snapshot(app.event_bus, fix_snapshot)
                        record_fix = getattr(app, "record_fix_pilot_snapshot", None)
                        if callable(record_fix):
                            try:
                                record_fix(fix_snapshot)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record FIX pilot snapshot on runtime app",
                                    exc_info=True,
                                )

                    experiment_events: Sequence[Mapping[str, Any]] = []
                    experiment_snapshot: EvolutionExperimentSnapshot | None = None
                    strategy_snapshot: StrategyPerformanceSnapshot | None = None
                    roi_snapshot: Any | None = None
                    if trading_manager is not None:
                        get_events = getattr(trading_manager, "get_experiment_events", None)
                        if callable(get_events):
                            try:
                                experiment_events = get_events()
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to gather evolution experiment events from trading manager",
                                    exc_info=True,
                                )
                        get_roi = getattr(trading_manager, "get_last_roi_snapshot", None)
                        if callable(get_roi):
                            try:
                                roi_snapshot = get_roi()
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to gather ROI snapshot for evolution experiments",
                                    exc_info=True,
                                )

                    if experiment_events or roi_snapshot is not None:
                        experiment_metadata: dict[str, object] = {
                            "ingest_success": success,
                            "execution_status": execution_snapshot.status.value,
                        }
                        if scheduler_metadata:
                            experiment_metadata["scheduler"] = dict(scheduler_metadata)
                        if trade_summary is not None:
                            experiment_metadata.setdefault("compliance", {})
                            experiment_metadata["compliance"] = dict(trade_summary)
                        if kyc_summary is not None:
                            experiment_metadata.setdefault("kyc", {})
                            experiment_metadata["kyc"] = dict(kyc_summary)
                        experiment_snapshot = evaluate_evolution_experiments(
                            experiment_events,
                            roi_snapshot=roi_snapshot,
                            metadata=experiment_metadata,
                        )
                        logger.info(
                            "🧬 Evolution experiment snapshot:\n%s",
                            format_evolution_experiment_markdown(experiment_snapshot),
                        )
                        publish_evolution_experiment_snapshot(app.event_bus, experiment_snapshot)
                        record_experiments = getattr(
                            app, "record_evolution_experiment_snapshot", None
                        )
                        if callable(record_experiments):
                            try:
                                record_experiments(experiment_snapshot)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record evolution experiment snapshot on runtime app",
                                    exc_info=True,
                                )

                        strategy_metadata = dict(experiment_metadata)
                        strategy_snapshot = evaluate_strategy_performance(
                            experiment_events,
                            roi_snapshot=roi_snapshot,
                            metadata=strategy_metadata,
                        )
                        logger.info(
                            "🎯 Strategy performance snapshot:\n%s",
                            format_strategy_performance_markdown(strategy_snapshot),
                        )
                        publish_strategy_performance_snapshot(app.event_bus, strategy_snapshot)
                        record_strategy = getattr(app, "record_strategy_performance_snapshot", None)
                        if callable(record_strategy):
                            try:
                                record_strategy(strategy_snapshot)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record strategy performance snapshot on runtime app",
                                    exc_info=True,
                                )

                        tuning_snapshot = evaluate_evolution_tuning(
                            experiment_snapshot,
                            strategy_snapshot,
                            metadata=strategy_metadata,
                        )
                        logger.info(
                            "🧪 Evolution tuning snapshot:\n%s",
                            format_evolution_tuning_markdown(tuning_snapshot),
                        )
                        publish_evolution_tuning_snapshot(app.event_bus, tuning_snapshot)
                        record_tuning = getattr(app, "record_evolution_tuning_snapshot", None)
                        if callable(record_tuning):
                            try:
                                record_tuning(tuning_snapshot)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record evolution tuning snapshot on runtime app",
                                    exc_info=True,
                                )

                    telemetry_metadata: dict[str, object] = {
                        "ingest_success": success,
                        "scheduler": scheduler_metadata,
                    }

                    cache_snapshot = evaluate_cache_health(
                        configured=redis_configured,
                        expected=backbone_context.redis_expected,
                        namespace=redis_namespace,
                        backing=redis_backing,
                        metrics=cache_metrics,
                        policy=cache_policy_details,
                        metadata=telemetry_metadata,
                    )
                    logger.info(
                        "🧰 Cache health snapshot:\n%s",
                        cache_snapshot.to_markdown(),
                    )
                    publish_cache_health(app.event_bus, cache_snapshot)
                    record_cache = getattr(app, "record_cache_snapshot", None)
                    if callable(record_cache):
                        try:
                            record_cache(cache_snapshot)
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to record cache snapshot on runtime app",
                                exc_info=True,
                            )

                    event_bus_snapshot = evaluate_event_bus_health(
                        app.event_bus,
                        expected=True,
                        metadata=telemetry_metadata,
                    )
                    logger.info(
                        "📬 Event bus health snapshot:\n%s",
                        format_event_bus_markdown(event_bus_snapshot),
                    )
                    publish_event_bus_health(app.event_bus, event_bus_snapshot)
                    record_event_bus = getattr(app, "record_event_bus_snapshot", None)
                    if callable(record_event_bus):
                        try:
                            record_event_bus(event_bus_snapshot)
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to record event bus snapshot on runtime app",
                                exc_info=True,
                            )

                    system_validation_snapshot = _resolve_system_validation_snapshot(
                        extras_mapping,
                        telemetry_metadata,
                    )
                    if system_validation_snapshot is not None:
                        logger.info(
                            "🧪 System validation snapshot:\n%s",
                            format_system_validation_markdown(system_validation_snapshot),
                        )
                        publish_system_validation_snapshot(
                            app.event_bus, system_validation_snapshot
                        )
                        record_system_validation = getattr(
                            app, "record_system_validation_snapshot", None
                        )
                        if callable(record_system_validation):
                            try:
                                record_system_validation(system_validation_snapshot)
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to record system validation snapshot on runtime app",
                                    exc_info=True,
                                )

                    sensory_status: Mapping[str, Any] | None = None
                    sensory_component = getattr(app, "sensory_organ", None)
                    if sensory_component is not None:
                        status_method = getattr(sensory_component, "status", None)
                        if callable(status_method):
                            try:
                                sensory_status = status_method()
                            except Exception:  # pragma: no cover - diagnostics only
                                logger.debug(
                                    "Failed to capture sensory status for drift telemetry",
                                    exc_info=True,
                                )
                    sensory_audit_entries = _process_sensory_status(app, sensory_status)

                    if sensory_audit_entries:
                        _evaluate_and_publish_drift(
                            app,
                            sensory_audit_entries,
                            metadata={
                                "ingest_success": success,
                                "scheduler": scheduler_metadata,
                            },
                        )
                    else:
                        logger.debug("Sensory drift evaluation skipped (no audit entries)")

                    try:
                        await publish_scheduler_snapshot(app.event_bus, scheduler_snapshot)
                    except Exception:  # pragma: no cover - diagnostics only
                        logger.debug("Failed to publish scheduler snapshot", exc_info=True)
                    record_scheduler = getattr(app, "record_scheduler_snapshot", None)
                    if callable(record_scheduler):
                        try:
                            record_scheduler(scheduler_snapshot)
                        except Exception:  # pragma: no cover - diagnostics only
                            logger.debug(
                                "Failed to record scheduler snapshot on runtime app",
                                exc_info=True,
                            )
                    return success

                async def _scheduled_ingest() -> bool:
                    nonlocal skip_next_scheduler_run
                    if skip_next_scheduler_run:
                        skip_next_scheduler_run = False
                        return True
                    return await _run_timescale_ingest()

                provisioner = InstitutionalIngestProvisioner(
                    ingest_config,
                    redis_settings=redis_settings,
                    redis_policy=ingest_config.redis_policy,
                    kafka_mapping=extras_mapping,
                )
                services = provisioner.provision(
                    run_ingest=_scheduled_ingest,
                    event_bus=app.event_bus,
                    task_supervisor=app.task_supervisor,
                    redis_client_factory=_redis_client_factory,
                )
                services_holder["services"] = services
                app.add_cleanup_callback(services.stop)

                manifest_snapshot = [snapshot.as_dict() for snapshot in services.managed_manifest()]
                if manifest_snapshot:
                    logger.info("🛰️ Managed ingest connectors provisioned: %s", manifest_snapshot)

                manual_success = await _run_timescale_ingest()
                if manual_success:
                    skip_next_scheduler_run = True

                services.start()
                logger.info("⏱️ Timescale ingest scheduler active under supervisor")
                return

            ingestion = RuntimeWorkload(
                name="timescale-ingest",
                factory=_run_institutional,
                description="Institutional Timescale ingest orchestrator",
                metadata=metadata,
                restart_policy=ingest_restart_policy,
            )
        else:
            reason: str | None = None
            if backbone_mode is DataBackboneMode.institutional and tier is not EmpTier.tier_1:
                reason = f"Tier {tier.value} not ready for institutional ingest"
            ingestion = _build_bootstrap_workload(
                app,
                symbols_csv=symbols_csv,
                db_path=duckdb_path,
                reason=reason,
                restart_policy=bootstrap_ingest_restart_policy,
            )

    trading_metadata: dict[str, object] = {
        "mode": app.config.run_mode.value,
        "protocol": app.config.connection_protocol.value,
    }
    trading_metadata["workload_kind"] = "understanding_loop"
    trading_metadata["supervised_components"] = (
        "understanding_loop",
        "trade_execution",
    )
    risk_startup_callback = _prepare_trading_risk_enforcement(app, trading_metadata)

    trading = RuntimeWorkload(
        name="professional-trading",
        factory=lambda: app.run_forever(),
        description="Professional Predator trading loop",
        metadata=trading_metadata,
        restart_policy=trading_restart_policy,
    )

    runtime_app = RuntimeApplication(
        ingestion=ingestion,
        trading=trading,
        tracer=runtime_tracer,
        task_supervisor=app.task_supervisor,
    )

    attach_runtime = getattr(app, "attach_runtime_application", None)
    if callable(attach_runtime):
        try:
            attach_runtime(runtime_app)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to attach runtime application to app", exc_info=True)

    if risk_startup_callback is not None:
        runtime_app.add_startup_callback(risk_startup_callback)

    extras = app.config.extras or {}
    _configure_drift_monitor(runtime_app, app, runtime_tracer, extras)
    _configure_governance_cadence(runtime_app, app, runtime_tracer, extras)
    health_enabled = _coerce_bool(extras.get("RUNTIME_HEALTHCHECK_ENABLED", True), True)
    if health_enabled:
        host = str(extras.get("RUNTIME_HEALTHCHECK_HOST", "0.0.0.0"))
        port = _coerce_int(extras.get("RUNTIME_HEALTHCHECK_PORT"), 8080)
        path = str(extras.get("RUNTIME_HEALTHCHECK_PATH", "/health"))
        health_server = RuntimeHealthServer(
            app,
            host=host,
            port=port,
            path=path,
            ingest_warn_after=_coerce_float(
                extras.get("RUNTIME_HEALTHCHECK_INGEST_WARN_SECONDS"), 900.0
            ),
            ingest_fail_after=_coerce_float(
                extras.get("RUNTIME_HEALTHCHECK_INGEST_FAIL_SECONDS"), 1800.0
            ),
            decision_warn_after=_coerce_float(
                extras.get("RUNTIME_HEALTHCHECK_DECISION_WARN_SECONDS"), 180.0
            ),
            decision_fail_after=_coerce_float(
                extras.get("RUNTIME_HEALTHCHECK_DECISION_FAIL_SECONDS"), 600.0
            ),
        )

        async def _start_health() -> None:
            await health_server.start()

        async def _stop_health() -> None:
            await health_server.stop()

        runtime_app.add_startup_callback(_start_health)
        runtime_app.add_shutdown_callback(_stop_health)

    return runtime_app


__all__ = [
    "RuntimeApplication",
    "RuntimeWorkload",
    "WorkloadRestartPolicy",
    "build_professional_runtime_application",
    "_normalise_ingest_plan_metadata",
    "_execute_timescale_ingest",
]
