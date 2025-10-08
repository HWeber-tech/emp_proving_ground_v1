"""Composable runtime assembly for the EMP Professional Predator."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
from collections import ChainMap
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from statistics import fmean, pstdev
from types import MappingProxyType, TracebackType
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    cast,
)

from src.compliance.kyc import KycAmlMonitor
from src.compliance.trade_compliance import TradeComplianceMonitor, TradeCompliancePolicy
from src.config.risk.risk_config import RiskConfig
from src.core.evolution.engine import EvolutionConfig, EvolutionEngine
from src.core.event_bus import EventBus
from src.core.interfaces import DecisionGenome
from src.data_foundation.batch.spark_export import (
    SparkExportSnapshot,
    format_spark_export_markdown,
)
from src.data_foundation.cache import (
    ManagedRedisCache,
    RedisCachePolicy,
    RedisConnectionSettings,
    TimescaleQueryCache,
    configure_redis_client,
    wrap_managed_cache,
)
from src.data_foundation.fabric.market_data_fabric import MarketDataConnector
from src.data_foundation.fabric.timescale_connector import (
    TimescaleDailyBarConnector,
    TimescaleIntradayTradeConnector,
)
from src.data_foundation.persist.timescale import (
    TimescaleComplianceJournal,
    TimescaleConnectionSettings,
    TimescaleExecutionJournal,
    TimescaleIngestJournal,
    TimescaleKycJournal,
    TimescaleMigrator,
)
from src.data_foundation.persist.timescale_reader import TimescaleReader
from src.data_foundation.services.macro_events import TimescaleMacroEventService
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaIngestEventConsumer,
    create_ingest_event_consumer,
)
from src.data_foundation.ingest.scheduler_telemetry import (
    IngestSchedulerSnapshot,
    format_scheduler_markdown,
)
from src.operations.ingest_trends import (
    IngestTrendSnapshot,
    format_ingest_trends_markdown,
)
from src.governance.audit_logger import AuditLogger
from src.governance.safety_manager import SafetyManager
from src.governance.strategy_registry import StrategyRegistry
from src.evolution.feature_flags import ADAPTIVE_RUNS_FLAG, EvolutionFeatureFlags
from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.understanding import DecisionDiaryStore, ProbeRegistry
from src.governance.system_config import ConnectionProtocol, DataBackboneMode, EmpTier, SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager, SystemConfigProtocol
from src.operations.backup import BackupReadinessSnapshot, format_backup_markdown
from src.compliance.workflow import ComplianceWorkflowSnapshot
from src.operations.compliance_readiness import ComplianceReadinessSnapshot
from src.operations.data_backbone import (
    DataBackboneReadinessSnapshot,
    DataBackboneValidationSnapshot,
)
from src.operations.failover_drill import (
    FailoverDrillSnapshot,
    format_failover_drill_markdown,
)
from src.operations.spark_stress import (
    SparkStressSnapshot,
    format_spark_stress_markdown,
)
from src.operations.slo import OperationalSLOSnapshot
from src.operations.professional_readiness import ProfessionalReadinessSnapshot
from src.operations.security import SecurityPostureSnapshot
from src.operations.cache_health import CacheHealthSnapshot
from src.operations.event_bus_health import (
    EventBusHealthSnapshot,
    format_event_bus_markdown as format_event_bus_health_markdown,
)
from src.operations.configuration_audit import (
    ConfigurationAuditSnapshot,
    format_configuration_audit_markdown,
)
from src.operations.kafka_readiness import (
    KafkaReadinessSnapshot,
    format_kafka_readiness_markdown,
)
from src.operations.cross_region_failover import (
    CrossRegionFailoverSnapshot,
    format_cross_region_markdown,
)
from src.operations.system_validation import (
    SystemValidationSnapshot,
    format_system_validation_markdown,
)
from src.operations.regulatory_telemetry import RegulatoryTelemetrySnapshot
from src.operations.governance_reporting import GovernanceReport
from src.operations.operational_readiness import (
    evaluate_operational_readiness,
    format_operational_readiness_markdown,
)
from src.operations.evolution_experiments import (
    EvolutionExperimentSnapshot,
    format_evolution_experiment_markdown,
)
from src.operations.evolution_tuning import (
    EvolutionTuningSnapshot,
    format_evolution_tuning_markdown,
)
from src.operations.strategy_performance import (
    StrategyPerformanceSnapshot,
    format_strategy_performance_markdown,
)
from src.operations.incident_response import (
    IncidentResponseSnapshot,
    format_incident_response_markdown,
)
from src.operations.roi import RoiCostModel, format_roi_markdown as format_roi_summary
from src.operations.execution import (
    ExecutionReadinessSnapshot,
    format_execution_markdown,
)
from src.operations.sensory_drift import SensoryDriftSnapshot
from src.operations.sensory_metrics import SensoryMetrics
from src.operations.sensory_summary import SensorySummary
from src.observability.tracing import (
    NullRuntimeTracer,
    RuntimeTracer,
    configure_event_bus_tracer,
    configure_runtime_tracer,
    parse_opentelemetry_settings,
)
from src.operations.retention import (
    DataRetentionSnapshot,
    format_data_retention_markdown,
)
from src.sensory.anomaly import AnomalySensor
from src.sensory.how.how_sensor import HowSensor
from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan
from src.sensory.what.what_sensor import WhatSensor
from src.sensory.when.when_sensor import WhenSensor
from src.sensory.why.why_sensor import WhySensor
from src.trading.integration.fix_broker_interface import FIXBrokerInterface
from src.trading.risk.policy_telemetry import format_policy_markdown
from src.trading.risk.risk_api import (
    RISK_API_RUNBOOK,
    RiskApiError,
    build_runtime_risk_metadata,
    resolve_trading_risk_interface,
)
from src.runtime.bootstrap_runtime import BootstrapRuntime
from src.orchestration.evolution_cycle import EvolutionCycleOrchestrator
from src.runtime.fix_dropcopy import FixDropcopyReconciler
from src.runtime.fix_pilot import FixIntegrationPilot
from src.runtime.task_supervisor import TaskSupervisor
from src.risk.telemetry import format_risk_markdown

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    from src.sensory.signals import SensorSignal
    from src.operations.fix_pilot import FixPilotSnapshot


logger = logging.getLogger(__name__)


def _call_manager_method(manager: Any, method_name: str) -> Any:
    target = getattr(manager, method_name, None)
    if not callable(target):
        return None
    try:
        return target()
    except Exception:  # pragma: no cover - diagnostics only
        return None


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any] | None:
    if snapshot is None:
        return None

    as_dict = getattr(snapshot, "as_dict", None)
    if callable(as_dict):
        try:
            payload = as_dict()
        except Exception:  # pragma: no cover - diagnostics only
            payload = None
        if isinstance(payload, Mapping):
            return dict(payload)

    if isinstance(snapshot, Mapping):
        return dict(snapshot)

    return None


def _format_snapshot_markdown(
    formatter: Callable[[Any], str | None], snapshot: Any
) -> str | None:
    try:
        return formatter(snapshot)
    except Exception:  # pragma: no cover - diagnostics only
        return None


CleanupCallback = Callable[[], Awaitable[None] | None]


class MarketDataSensor(Protocol):
    """Protocol describing the narrow interface used by Tier-0 ingestion."""

    def process(self, df: "pd.DataFrame") -> Sequence["SensorSignal"]: ...


SensoryRuntime = FIXSensoryOrgan | BootstrapRuntime


class ProfessionalPredatorApp:
    """High-level application wrapper with explicit lifecycle management."""

    def __init__(
        self,
        *,
        config: SystemConfig,
        event_bus: EventBus,
        sensory_organ: SensoryRuntime | None,
        broker_interface: Optional[FIXBrokerInterface],
        fix_connection_manager: Optional[FIXConnectionManager],
        sensors: Mapping[str, MarketDataSensor],
        compliance_monitor: Optional[TradeComplianceMonitor] = None,
        kyc_monitor: Optional[KycAmlMonitor] = None,
        redis_client: ManagedRedisCache | None = None,
        execution_journal: TimescaleExecutionJournal | None = None,
        strategy_registry: StrategyRegistry | None = None,
        task_supervisor: TaskSupervisor | None = None,
        runtime_tracer: RuntimeTracer | None = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.sensory_organ: SensoryRuntime | None = sensory_organ
        self.broker_interface = broker_interface
        self.fix_connection_manager = fix_connection_manager
        self._sensors = dict(sensors)
        self.compliance_monitor = compliance_monitor
        self.kyc_monitor = kyc_monitor
        self.redis_client = redis_client
        self.execution_journal = execution_journal
        self.strategy_registry = strategy_registry
        self._runtime_tracer: RuntimeTracer = runtime_tracer or NullRuntimeTracer()

        self._logger = logging.getLogger(__name__)
        self._cleanup_callbacks: list[CleanupCallback] = []
        self._task_supervisor = task_supervisor or TaskSupervisor(
            namespace="professional-runtime", logger=self._logger
        )
        self.fix_pilot: FixIntegrationPilot | None = None
        self._stop_event = asyncio.Event()
        self._started = False
        self._closed = False
        self._start_time: datetime | None = None
        self._shutdown_time: datetime | None = None
        self._last_backup_snapshot: BackupReadinessSnapshot | None = None
        self._last_backbone_snapshot: DataBackboneReadinessSnapshot | None = None
        self._last_data_retention_snapshot: DataRetentionSnapshot | None = None
        self._last_backbone_validation_snapshot: DataBackboneValidationSnapshot | None = None
        self._last_professional_snapshot: ProfessionalReadinessSnapshot | None = None
        self._last_operational_slo_snapshot: OperationalSLOSnapshot | None = None
        self._last_compliance_snapshot: ComplianceReadinessSnapshot | None = None
        self._last_compliance_workflow_snapshot: ComplianceWorkflowSnapshot | None = None
        self._last_regulatory_snapshot: RegulatoryTelemetrySnapshot | None = None
        self._last_governance_report: GovernanceReport | None = None
        self._last_security_snapshot: SecurityPostureSnapshot | None = None
        self._last_incident_response_snapshot: IncidentResponseSnapshot | None = None
        self._last_cache_snapshot: CacheHealthSnapshot | None = None
        self._last_kafka_readiness_snapshot: KafkaReadinessSnapshot | None = None
        self._last_cross_region_snapshot: CrossRegionFailoverSnapshot | None = None
        self._last_event_bus_snapshot: EventBusHealthSnapshot | None = None
        self._last_scheduler_snapshot: IngestSchedulerSnapshot | None = None
        self._last_spark_export_snapshot: SparkExportSnapshot | None = None
        self._last_spark_stress_snapshot: SparkStressSnapshot | None = None
        self._last_failover_drill_snapshot: FailoverDrillSnapshot | None = None
        self._last_execution_snapshot: ExecutionReadinessSnapshot | None = None
        self._last_sensory_drift_snapshot: SensoryDriftSnapshot | None = None
        self._last_sensory_summary: SensorySummary | None = None
        self._last_sensory_metrics: SensoryMetrics | None = None
        self._last_system_validation_snapshot: SystemValidationSnapshot | None = None
        self._last_evolution_experiment_snapshot: EvolutionExperimentSnapshot | None = None
        self._last_evolution_tuning_snapshot: EvolutionTuningSnapshot | None = None
        self._last_strategy_performance_snapshot: StrategyPerformanceSnapshot | None = None
        self._last_fix_pilot_snapshot: "FixPilotSnapshot" | None = None
        self._last_ingest_trend_snapshot: IngestTrendSnapshot | None = None
        self._last_configuration_snapshot: ConfigurationAuditSnapshot | None = None
        self._last_risk_configuration: dict[str, Any] | None = None
        self._kafka_bridge_metadata: dict[str, object] | None = None

    async def __aenter__(self) -> "ProfessionalPredatorApp":
        await self.start()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        await self.shutdown()

    @property
    def sensors(self) -> Mapping[str, MarketDataSensor]:
        """Read-only view of configured sensors."""

        return MappingProxyType(self._sensors)

    def add_cleanup_callback(self, callback: CleanupCallback) -> None:
        self._cleanup_callbacks.append(callback)

    def create_background_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: Optional[str] = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> asyncio.Task[Any]:
        return self._task_supervisor.create(coro, name=name, metadata=metadata)

    def register_background_task(self, task: asyncio.Task[Any]) -> None:
        """Track an externally-created background task for managed shutdown."""

        if not isinstance(task, asyncio.Task):
            raise TypeError("register_background_task expects an asyncio.Task")
        self._task_supervisor.track(task)

    @property
    def runtime_tracer(self) -> RuntimeTracer:
        """Tracer used for runtime workload instrumentation."""

        return self._runtime_tracer

    def get_kafka_bridge_metadata(self) -> Mapping[str, object] | None:
        """Expose the resolved Kafka ingest bridge metadata for diagnostics."""

        if self._kafka_bridge_metadata is None:
            return None
        return dict(self._kafka_bridge_metadata)

    def record_backup_snapshot(self, snapshot: BackupReadinessSnapshot) -> None:
        """Store the most recent backup readiness snapshot for summaries."""

        self._last_backup_snapshot = snapshot

    def record_data_backbone_snapshot(self, snapshot: DataBackboneReadinessSnapshot) -> None:
        """Record the latest data backbone readiness snapshot."""

        self._last_backbone_snapshot = snapshot

    def record_data_retention_snapshot(self, snapshot: DataRetentionSnapshot) -> None:
        """Store the most recent data retention snapshot."""

        self._last_data_retention_snapshot = snapshot

    def record_data_backbone_validation_snapshot(
        self, snapshot: DataBackboneValidationSnapshot
    ) -> None:
        """Record the latest data backbone validation snapshot."""

        self._last_backbone_validation_snapshot = snapshot

    def record_professional_readiness_snapshot(
        self, snapshot: ProfessionalReadinessSnapshot
    ) -> None:
        """Store the latest professional readiness snapshot."""

        self._last_professional_snapshot = snapshot

    def record_operational_slo_snapshot(
        self, snapshot: OperationalSLOSnapshot
    ) -> None:
        """Store the latest operational SLO snapshot."""

        self._last_operational_slo_snapshot = snapshot

    def get_last_operational_slo_snapshot(self) -> OperationalSLOSnapshot | None:
        """Return the most recent operational SLO snapshot, if any."""

        return self._last_operational_slo_snapshot

    def record_spark_export_snapshot(self, snapshot: SparkExportSnapshot) -> None:
        """Record the most recent Spark export snapshot for summaries."""

        self._last_spark_export_snapshot = snapshot

    def record_spark_stress_snapshot(self, snapshot: SparkStressSnapshot) -> None:
        """Record the latest Spark stress drill snapshot."""

        self._last_spark_stress_snapshot = snapshot

    def record_failover_drill_snapshot(self, snapshot: FailoverDrillSnapshot) -> None:
        """Store the latest failover drill outcome for runtime summaries."""

        self._last_failover_drill_snapshot = snapshot

    def record_compliance_readiness_snapshot(self, snapshot: ComplianceReadinessSnapshot) -> None:
        """Store the latest compliance readiness snapshot."""

        self._last_compliance_snapshot = snapshot

    def record_compliance_workflow_snapshot(self, snapshot: ComplianceWorkflowSnapshot) -> None:
        """Store the most recent compliance workflow checklist."""

        self._last_compliance_workflow_snapshot = snapshot

    def record_regulatory_snapshot(self, snapshot: RegulatoryTelemetrySnapshot) -> None:
        """Store the latest regulatory telemetry snapshot."""

        self._last_regulatory_snapshot = snapshot

    def record_governance_report(self, report: GovernanceReport) -> None:
        """Store the latest governance compliance report."""

        self._last_governance_report = report

    def record_security_snapshot(self, snapshot: SecurityPostureSnapshot) -> None:
        """Store the most recent security posture snapshot."""

        self._last_security_snapshot = snapshot

    def record_incident_response_snapshot(self, snapshot: IncidentResponseSnapshot) -> None:
        """Store the latest incident response telemetry."""

        self._last_incident_response_snapshot = snapshot

    def record_cache_snapshot(self, snapshot: CacheHealthSnapshot) -> None:
        """Store the most recent cache health snapshot."""

        self._last_cache_snapshot = snapshot

    def record_kafka_readiness_snapshot(self, snapshot: KafkaReadinessSnapshot) -> None:
        """Store the latest Kafka readiness snapshot."""

        self._last_kafka_readiness_snapshot = snapshot

    def record_cross_region_snapshot(self, snapshot: CrossRegionFailoverSnapshot) -> None:
        """Store the latest cross-region failover readiness snapshot."""

        self._last_cross_region_snapshot = snapshot

    def record_event_bus_snapshot(self, snapshot: EventBusHealthSnapshot) -> None:
        """Store the most recent event bus health snapshot."""

        self._last_event_bus_snapshot = snapshot

    def record_scheduler_snapshot(self, snapshot: IngestSchedulerSnapshot) -> None:
        """Store the latest ingest scheduler telemetry snapshot."""

        self._last_scheduler_snapshot = snapshot

    def record_ingest_trend_snapshot(self, snapshot: IngestTrendSnapshot) -> None:
        """Store the latest ingest trend snapshot for summaries."""

        self._last_ingest_trend_snapshot = snapshot

    def record_configuration_snapshot(self, snapshot: ConfigurationAuditSnapshot) -> None:
        """Store the latest configuration audit snapshot."""

        self._last_configuration_snapshot = snapshot

    def record_risk_configuration(self, payload: Mapping[str, object]) -> None:
        """Store the latest risk configuration payload for runtime summaries."""

        self._last_risk_configuration = dict(payload)

    def record_execution_snapshot(self, snapshot: ExecutionReadinessSnapshot) -> None:
        """Store the latest execution readiness snapshot."""

        self._last_execution_snapshot = snapshot
        journal: TimescaleExecutionJournal | None = getattr(self, "execution_journal", None)
        if journal is not None:
            try:
                journal.record_snapshot(snapshot, strategy_id=self._resolve_strategy_id())
            except Exception:  # pragma: no cover - diagnostics only
                self._logger.debug("Failed to persist execution readiness snapshot", exc_info=True)

    def record_fix_pilot_snapshot(self, snapshot: "FixPilotSnapshot") -> None:
        """Store the latest FIX pilot snapshot for runtime summaries."""

        self._last_fix_pilot_snapshot = snapshot

    def get_last_backup_snapshot(self) -> BackupReadinessSnapshot | None:
        """Return the last recorded backup readiness snapshot, if any."""

        return self._last_backup_snapshot

    def get_last_data_backbone_snapshot(
        self,
    ) -> DataBackboneReadinessSnapshot | None:
        """Return the last recorded data backbone readiness snapshot."""

        return self._last_backbone_snapshot

    def get_last_data_retention_snapshot(self) -> DataRetentionSnapshot | None:
        """Return the latest data retention snapshot, if any."""

        return self._last_data_retention_snapshot

    def get_last_data_backbone_validation_snapshot(
        self,
    ) -> DataBackboneValidationSnapshot | None:
        """Return the last recorded data backbone validation snapshot, if any."""

        return self._last_backbone_validation_snapshot

    def get_last_professional_readiness_snapshot(
        self,
    ) -> ProfessionalReadinessSnapshot | None:
        """Return the last recorded professional readiness snapshot, if any."""

        return self._last_professional_snapshot

    def get_last_compliance_readiness_snapshot(
        self,
    ) -> ComplianceReadinessSnapshot | None:
        """Return the last compliance readiness snapshot surfaced by the runtime."""

        return self._last_compliance_snapshot

    def get_last_compliance_workflow_snapshot(
        self,
    ) -> ComplianceWorkflowSnapshot | None:
        """Return the last compliance workflow snapshot captured by the runtime."""

        return self._last_compliance_workflow_snapshot

    def get_last_regulatory_snapshot(self) -> RegulatoryTelemetrySnapshot | None:
        """Return the last regulatory telemetry snapshot captured by the runtime."""

        return self._last_regulatory_snapshot

    def get_last_governance_report(self) -> GovernanceReport | None:
        """Return the last governance report assembled by the runtime."""

        return self._last_governance_report

    def get_last_security_snapshot(self) -> SecurityPostureSnapshot | None:
        """Return the most recent security posture snapshot, if any."""

        return self._last_security_snapshot

    def get_last_incident_response_snapshot(
        self,
    ) -> IncidentResponseSnapshot | None:
        """Return the last recorded incident response snapshot, if any."""

        return self._last_incident_response_snapshot

    def get_last_cache_snapshot(self) -> CacheHealthSnapshot | None:
        """Return the last recorded cache health snapshot, if any."""

        return self._last_cache_snapshot

    def get_last_kafka_readiness_snapshot(self) -> KafkaReadinessSnapshot | None:
        """Return the last recorded Kafka readiness snapshot, if any."""

        return self._last_kafka_readiness_snapshot

    def get_last_cross_region_snapshot(self) -> CrossRegionFailoverSnapshot | None:
        """Return the most recent cross-region failover snapshot, if any."""

        return self._last_cross_region_snapshot

    def get_last_event_bus_snapshot(self) -> EventBusHealthSnapshot | None:
        """Return the most recent event bus health snapshot, if any."""

        return self._last_event_bus_snapshot

    def record_system_validation_snapshot(self, snapshot: SystemValidationSnapshot) -> None:
        """Store the most recent system validation snapshot."""

        self._last_system_validation_snapshot = snapshot

    def get_last_system_validation_snapshot(
        self,
    ) -> SystemValidationSnapshot | None:
        """Return the most recent system validation snapshot, if any."""

        return self._last_system_validation_snapshot

    def record_evolution_experiment_snapshot(self, snapshot: EvolutionExperimentSnapshot) -> None:
        """Store the most recent evolution experiment snapshot."""

        self._last_evolution_experiment_snapshot = snapshot

    def get_last_evolution_experiment_snapshot(
        self,
    ) -> EvolutionExperimentSnapshot | None:
        """Return the most recent evolution experiment snapshot, if any."""

        return self._last_evolution_experiment_snapshot

    def record_evolution_tuning_snapshot(self, snapshot: EvolutionTuningSnapshot) -> None:
        """Store the most recent evolution tuning snapshot."""

        self._last_evolution_tuning_snapshot = snapshot

    def get_last_evolution_tuning_snapshot(
        self,
    ) -> EvolutionTuningSnapshot | None:
        """Return the most recent evolution tuning snapshot, if any."""

        return self._last_evolution_tuning_snapshot

    def record_strategy_performance_snapshot(self, snapshot: StrategyPerformanceSnapshot) -> None:
        """Store the latest strategy performance telemetry."""

        self._last_strategy_performance_snapshot = snapshot

    def get_last_strategy_performance_snapshot(
        self,
    ) -> StrategyPerformanceSnapshot | None:
        """Expose the last recorded strategy performance snapshot, if any."""

        return self._last_strategy_performance_snapshot

    def get_last_scheduler_snapshot(self) -> IngestSchedulerSnapshot | None:
        """Return the last recorded ingest scheduler snapshot, if any."""

        return self._last_scheduler_snapshot

    def get_last_ingest_trend_snapshot(self) -> IngestTrendSnapshot | None:
        """Return the most recent ingest trend snapshot."""

        return self._last_ingest_trend_snapshot

    def get_last_execution_snapshot(self) -> ExecutionReadinessSnapshot | None:
        """Return the last recorded execution readiness snapshot, if any."""

        return self._last_execution_snapshot

    def record_sensory_summary(self, summary: SensorySummary) -> None:
        """Store the most recent sensory summary payload."""

        self._last_sensory_summary = summary

    def get_last_sensory_summary(self) -> SensorySummary | None:
        """Return the most recent sensory summary, if any."""

        return self._last_sensory_summary

    def record_sensory_metrics(self, metrics: SensoryMetrics) -> None:
        """Store the most recent sensory metrics payload."""

        self._last_sensory_metrics = metrics

    def get_last_sensory_metrics(self) -> SensoryMetrics | None:
        """Return the most recent sensory metrics, if any."""

        return self._last_sensory_metrics

    def record_sensory_drift_snapshot(self, snapshot: SensoryDriftSnapshot) -> None:
        """Store the most recent sensory drift snapshot."""

        self._last_sensory_drift_snapshot = snapshot
        trading_manager = getattr(self.sensory_organ, "trading_manager", None)
        update_method = getattr(trading_manager, "update_drift_sentry_snapshot", None)
        if callable(update_method):
            try:
                update_method(snapshot)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to propagate sensory drift snapshot to trading manager",
                    exc_info=True,
                )

    def get_last_sensory_drift_snapshot(self) -> SensoryDriftSnapshot | None:
        """Return the most recent sensory drift snapshot, if any."""

        return self._last_sensory_drift_snapshot

    @property
    def task_supervisor(self) -> TaskSupervisor:
        """Expose the runtime task supervisor for orchestration helpers."""

        return self._task_supervisor

    def attach_fix_pilot(self, pilot: FixIntegrationPilot) -> None:
        """Attach a FIX integration pilot responsible for supervised lifecycles."""

        self.fix_pilot = pilot

    def _resolve_strategy_id(self) -> str:
        extras = self.config.extras or {}
        strategy_hint = (
            extras.get("EXECUTION_STRATEGY_ID")
            or extras.get("TRADE_STRATEGY_ID")
            or extras.get("COMPLIANCE_STRATEGY_ID")
            or extras.get("STRATEGY_ID")
            or extras.get("BOOTSTRAP_STRATEGY_ID")
        )
        if strategy_hint:
            return str(strategy_hint)
        trading_stack = getattr(self.sensory_organ, "trading_stack", None)
        if trading_stack is not None:
            strategy_value = getattr(trading_stack, "strategy_id", None)
            if strategy_value:
                return str(strategy_value)
        return "bootstrap-strategy"

    @property
    def active_background_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        """Return the active background tasks supervised by the runtime."""

        return self._task_supervisor.active_tasks

    def _build_kafka_bridge_metadata(
        self, kafka_bridge: KafkaIngestEventConsumer
    ) -> dict[str, object]:
        metadata: dict[str, object] = {"component": "kafka_ingest_bridge"}

        topics = getattr(kafka_bridge, "topics", ())
        if topics:
            metadata["topics"] = [str(topic) for topic in topics if str(topic).strip()]

        default_event_type = getattr(kafka_bridge, "default_event_type", None)
        if default_event_type:
            metadata["event_type"] = str(default_event_type)

        topic_event_types = getattr(kafka_bridge, "topic_event_types", None)
        if topic_event_types:
            metadata["topic_event_types"] = {
                str(topic): str(event_type)
                for topic, event_type in dict(topic_event_types).items()
                if str(topic).strip() and str(event_type).strip()
            }

        event_source = getattr(kafka_bridge, "event_source", None)
        if event_source:
            metadata["event_source"] = str(event_source)

        consumer_group = getattr(kafka_bridge, "consumer_group", None)
        if consumer_group:
            metadata["consumer_group"] = str(consumer_group)

        poll_timeout = getattr(kafka_bridge, "poll_timeout", None)
        if poll_timeout is not None:
            metadata["poll_timeout_seconds"] = float(poll_timeout)

        idle_sleep = getattr(kafka_bridge, "idle_sleep", None)
        if idle_sleep is not None:
            metadata["idle_sleep_seconds"] = float(idle_sleep)

        commit_offsets = getattr(kafka_bridge, "commit_offsets_enabled", None)
        if commit_offsets is not None:
            metadata["commit_offsets"] = bool(commit_offsets)

        commit_async = getattr(kafka_bridge, "commit_asynchronously", None)
        if commit_async is not None:
            metadata["commit_async"] = bool(commit_async)

        publish_consumer_lag = getattr(kafka_bridge, "publish_consumer_lag_enabled", None)
        if publish_consumer_lag is not None:
            metadata["publish_consumer_lag"] = bool(publish_consumer_lag)

        consumer_lag_event_type = getattr(kafka_bridge, "consumer_lag_event_type", None)
        if consumer_lag_event_type:
            metadata["consumer_lag_event_type"] = str(consumer_lag_event_type)

        consumer_lag_source = getattr(kafka_bridge, "consumer_lag_source", None)
        if consumer_lag_source:
            metadata["consumer_lag_source"] = str(consumer_lag_source)

        consumer_lag_interval = getattr(kafka_bridge, "consumer_lag_interval", None)
        if consumer_lag_interval is not None:
            metadata["consumer_lag_interval_seconds"] = float(consumer_lag_interval)

        return metadata

    def _build_kafka_bridge_summary(
        self, task_details: Sequence[Mapping[str, Any]]
    ) -> dict[str, object] | None:
        if self._kafka_bridge_metadata is None:
            return None

        payload = dict(self._kafka_bridge_metadata)
        payload["active"] = False

        for entry in task_details:
            if entry.get("name") == "kafka-ingest-bridge":
                payload["active"] = True
                payload["state"] = entry.get("state")
                payload["task_created_at"] = entry.get("created_at")
                break

        return payload

    def _register_component_tasks(self, component: Any) -> None:
        """Capture known background tasks emitted by runtime components."""

        for attr in ("_price_task", "_trade_task"):
            task = getattr(component, attr, None)
            if isinstance(task, asyncio.Task):
                self.register_background_task(task)

    async def _start_component(self, component: Any) -> None:
        if component is None:
            return

        start_method = getattr(component, "start", None)
        if start_method is None:
            return

        try:
            result = start_method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._logger.exception("Error starting component %s", component.__class__.__name__)
            raise
        else:
            self._register_component_tasks(component)

    async def _stop_component(self, component: Any) -> None:
        if component is None:
            return

        stop_method = getattr(component, "stop", None)
        if stop_method is None:
            return

        try:
            result = stop_method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self._logger.exception("Error stopping component %s", component.__class__.__name__)

    async def _activate_components(self) -> None:
        if self.fix_pilot is not None:
            await self.fix_pilot.start()
            return
        await self._start_component(self.sensory_organ)
        await self._start_component(self.broker_interface)

    async def _deactivate_components(self) -> None:
        if self.fix_pilot is not None:
            await self.fix_pilot.stop()
            return
        await self._stop_component(self.broker_interface)
        await self._stop_component(self.sensory_organ)

    async def start(self) -> None:
        if self._started:
            return

        self._logger.info("🚀 Initializing EMP v4.0 Professional Predator")
        self._logger.info("✅ Configuration loaded: %s", self.config.to_dict())
        self._logger.info("🔧 Protocol: %s", self.config.connection_protocol.value)
        self._logger.info("🧰 Run mode: %s", self.config.run_mode.value)
        self._logger.info("🏷️ Tier selected: %s", self.config.tier.value)
        self._logger.info("🧵 Data backbone mode: %s", self.config.data_backbone_mode.value)
        if self.sensory_organ:
            self._logger.info("✅ %s ready", self.sensory_organ.__class__.__name__)
        if self.broker_interface:
            self._logger.info("✅ %s ready", self.broker_interface.__class__.__name__)

        self._stop_event = asyncio.Event()
        self._started = True
        self._closed = False
        self._start_time = datetime.now()
        self._shutdown_time = None

        await self._activate_components()
        self._logger.info("🎉 Professional Predator initialization complete")

    async def run_forever(self, heartbeat_seconds: float = 60.0) -> None:
        """Block until shutdown, emitting debug heartbeats."""

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                self._logger.debug("Heartbeat check - system alive")

    def request_shutdown(self) -> None:
        self._stop_event.set()

    async def shutdown(self) -> None:
        if self._closed:
            return

        self.request_shutdown()

        await self._deactivate_components()

        for callback in reversed(self._cleanup_callbacks):
            try:
                outcome = callback()
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("Error during shutdown callback %s", callback)

        await self._task_supervisor.cancel_all()
        self._shutdown_time = datetime.now()
        self._closed = True
        self._started = False
        self._logger.info("✅ Professional Predator shutdown complete")

    def summary(self) -> Dict[str, Any]:
        status: str
        if self._closed:
            status = "STOPPED"
        elif self._started:
            status = "RUNNING"
        else:
            status = "INITIALIZED"

        uptime_seconds: float = 0.0
        if self._start_time:
            end_time = self._shutdown_time or datetime.now()
            uptime_seconds = max((end_time - self._start_time).total_seconds(), 0.0)

        ingest_journal_summary: Dict[str, Any] | None = None
        if self.config.data_backbone_mode is DataBackboneMode.institutional:
            extras = self.config.extras or {}
            try:
                settings = TimescaleConnectionSettings.from_mapping(extras)
                engine = settings.create_engine()
            except Exception:  # pragma: no cover - diagnostics should not fail summary
                self._logger.debug(
                    "Unable to create Timescale engine for ingest journal summary",
                    exc_info=True,
                )
            else:
                try:
                    ingest_journal = TimescaleIngestJournal(engine)
                    recent = ingest_journal.fetch_recent(limit=5)
                    latest = ingest_journal.fetch_latest_by_dimension()
                    ingest_journal_summary = {
                        "recent": [record.as_dict() for record in recent],
                        "latest_status": {
                            dimension: record.as_dict() for dimension, record in latest.items()
                        },
                    }
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug(
                        "Failed to read ingest journal for runtime summary", exc_info=True
                    )
                finally:
                    try:
                        engine.dispose()
                    except Exception:  # pragma: no cover - best effort cleanup
                        self._logger.debug("Timescale engine dispose failed", exc_info=True)

        components: Dict[str, Any] = {
            "sensory_organ": self.sensory_organ.__class__.__name__ if self.sensory_organ else None,
            "broker_interface": self.broker_interface.__class__.__name__
            if self.broker_interface
            else None,
            "fix_manager": "FIXConnectionManager" if self.fix_connection_manager else None,
            "sensors": sorted(self._sensors.keys()),
        }

        sensor_audit: list[Mapping[str, Any]] = []

        if self.sensory_organ and hasattr(self.sensory_organ, "running"):
            components["sensory_running"] = bool(getattr(self.sensory_organ, "running"))
        if self.broker_interface and hasattr(self.broker_interface, "running"):
            components["broker_running"] = bool(getattr(self.broker_interface, "running"))

        if self._last_fix_pilot_snapshot is not None:
            components["fix_pilot"] = {
                "status": self._last_fix_pilot_snapshot.status.value,
                "components": [
                    {
                        "name": comp.name,
                        "status": comp.status.value,
                        "details": dict(comp.details),
                    }
                    for comp in self._last_fix_pilot_snapshot.components
                ],
            }

        if self.sensory_organ and hasattr(self.sensory_organ, "status"):
            try:
                status_payload = getattr(self.sensory_organ, "status")()
            except Exception:  # pragma: no cover - diagnostics should not fail summary
                status_payload = None
            if status_payload is not None:
                if isinstance(status_payload, Mapping):
                    payload_dict = dict(status_payload)
                    components["sensory_status"] = payload_dict
                    audit_payload = payload_dict.get("sensor_audit")
                    if isinstance(audit_payload, list):
                        sensor_audit = [
                            item if isinstance(item, Mapping) else dict(item)
                            for item in audit_payload
                        ]
                else:
                    components["sensory_status"] = status_payload

        queue_metrics: Dict[str, Dict[str, int]] = {}
        if self.fix_connection_manager:
            price_app = self.fix_connection_manager.get_application("price")
            trade_app = self.fix_connection_manager.get_application("trade")
            dropcopy_app = self.fix_connection_manager.get_application("dropcopy")
            if price_app and hasattr(price_app, "get_queue_metrics"):
                queue_metrics["price"] = dict(price_app.get_queue_metrics())
            if trade_app and hasattr(trade_app, "get_queue_metrics"):
                queue_metrics["trade"] = dict(trade_app.get_queue_metrics())
            if dropcopy_app and hasattr(dropcopy_app, "get_queue_metrics"):
                queue_metrics["dropcopy"] = dict(dropcopy_app.get_queue_metrics())
        if queue_metrics:
            components["queue_metrics"] = queue_metrics

        summary_payload: Dict[str, Any] = {
            "version": "4.0",
            "protocol": self.config.connection_protocol.value,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime_seconds,
            "background_tasks": self._task_supervisor.active_count,
            "cleanup_callbacks": len(self._cleanup_callbacks),
            "components": components,
            "sensor_audit": sensor_audit,
        }

        task_details = self._task_supervisor.describe()
        if task_details:
            summary_payload["background_task_details"] = task_details

        kafka_bridge_summary = self._build_kafka_bridge_summary(task_details)
        if kafka_bridge_summary:
            summary_payload["kafka_ingest_bridge"] = kafka_bridge_summary

        if ingest_journal_summary is not None:
            summary_payload["ingest_journal"] = ingest_journal_summary

        if self.strategy_registry is not None:
            try:
                registry_summary = self.strategy_registry.get_registry_summary()
            except Exception:  # pragma: no cover - diagnostics only
                self._logger.debug("Failed to summarise strategy registry", exc_info=True)
            else:
                summary_payload["strategy_registry"] = registry_summary

        if self._last_scheduler_snapshot is not None:
            scheduler_snapshot = self._last_scheduler_snapshot
            scheduler_payload: dict[str, object] = {"snapshot": scheduler_snapshot.as_dict()}
            markdown = format_scheduler_markdown(scheduler_snapshot)
            if markdown:
                scheduler_payload["markdown"] = markdown
            summary_payload["ingest_scheduler"] = scheduler_payload

        if self._last_ingest_trend_snapshot is not None:
            trend_snapshot = self._last_ingest_trend_snapshot
            trend_payload: dict[str, object] = {
                "snapshot": trend_snapshot.as_dict(),
                "markdown": format_ingest_trends_markdown(trend_snapshot),
            }
            summary_payload["ingest_trends"] = trend_payload

        if self._last_configuration_snapshot is not None:
            configuration_snapshot = self._last_configuration_snapshot
            summary_payload["configuration_audit"] = {
                "snapshot": configuration_snapshot.as_dict(),
                "markdown": format_configuration_audit_markdown(configuration_snapshot),
            }

        if self.compliance_monitor is not None:
            try:
                summary_payload["compliance"] = self.compliance_monitor.summary()
            except Exception:  # pragma: no cover - diagnostics only
                self._logger.debug("Failed to build compliance summary", exc_info=True)

        kyc_monitor = getattr(self, "kyc_monitor", None)
        if kyc_monitor is not None:
            try:
                summary_payload["kyc"] = kyc_monitor.summary()
            except Exception:  # pragma: no cover - diagnostics only
                self._logger.debug("Failed to build KYC summary", exc_info=True)

        trading_manager = getattr(self.sensory_organ, "trading_manager", None)
        if trading_manager is not None:
            risk_section = summary_payload.setdefault("risk", {})
            risk_section.setdefault("runbook", RISK_API_RUNBOOK)
            if self._last_risk_configuration is not None:
                risk_section.setdefault(
                    "configuration_event",
                    dict(self._last_risk_configuration),
                )

            snapshot_obj = _call_manager_method(trading_manager, "get_last_risk_snapshot")
            snapshot_dict = _snapshot_to_dict(snapshot_obj)
            risk_markdown = (
                _format_snapshot_markdown(format_risk_markdown, snapshot_obj)
                if snapshot_obj is not None
                else None
            )
            if snapshot_dict is not None:
                risk_section["snapshot"] = snapshot_dict
                if risk_markdown:
                    risk_section["markdown"] = risk_markdown

            try:
                runtime_metadata = build_runtime_risk_metadata(trading_manager)
            except RiskApiError as exc:
                errors = risk_section.setdefault("errors", {})
                errors["runtime"] = exc.to_metadata()
            else:
                risk_section["runtime"] = runtime_metadata

            extra_interface_payload: dict[str, Any] | None = None
            describe_interface = getattr(trading_manager, "describe_risk_interface", None)
            if callable(describe_interface):
                try:
                    interface_candidate = describe_interface()
                except RiskApiError as exc:
                    errors = risk_section.setdefault("errors", {})
                    errors["interface"] = exc.to_metadata()
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug(
                        "Failed to resolve trading risk interface for runtime summary",
                        exc_info=True,
                    )
                else:
                    if interface_candidate is not None:
                        if isinstance(interface_candidate, Mapping):
                            extra_interface_payload = dict(interface_candidate)
                        else:
                            extra_interface_payload = {"value": interface_candidate}

            try:
                interface = resolve_trading_risk_interface(trading_manager)
            except RiskApiError as exc:
                errors = risk_section.setdefault("errors", {})
                errors.setdefault("interface", exc.to_metadata())
                if extra_interface_payload is not None:
                    risk_section["interface"] = extra_interface_payload
            else:
                interface_payload: dict[str, Any] = {
                    "summary": interface.summary(),
                    "config": interface.config.dict(),
                }
                if interface.status is not None:
                    interface_payload["status"] = dict(interface.status)
                if extra_interface_payload is None:
                    risk_section["interface"] = interface_payload
                else:
                    merged = dict(extra_interface_payload)
                    merged.setdefault("summary", interface_payload["summary"])
                    merged.setdefault("config", interface_payload["config"])
                    if "status" not in merged and "status" in interface_payload:
                        merged["status"] = interface_payload["status"]
                    risk_section["interface"] = merged

            policy_obj = _call_manager_method(trading_manager, "get_last_policy_snapshot")
            policy_snapshot = _snapshot_to_dict(policy_obj)
            policy_markdown = (
                _format_snapshot_markdown(format_policy_markdown, policy_obj)
                if policy_obj is not None
                else None
            )
            if policy_snapshot is not None:
                policy_block: dict[str, Any] = {"snapshot": policy_snapshot}
                if policy_markdown:
                    policy_block["markdown"] = policy_markdown
                risk_section["policy"] = policy_block

            roi_obj = _call_manager_method(trading_manager, "get_last_roi_snapshot")
            roi_dict = _snapshot_to_dict(roi_obj)
            roi_markdown = (
                _format_snapshot_markdown(format_roi_summary, roi_obj)
                if roi_obj is not None
                else None
            )
            if roi_dict is not None:
                summary_payload["roi"] = {"snapshot": roi_dict}
                if roi_markdown:
                    summary_payload["roi"]["markdown"] = roi_markdown

            describe_gate = getattr(trading_manager, "describe_drift_gate", None)
            if callable(describe_gate):
                try:
                    gate_payload = describe_gate()
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug(
                        "Failed to capture drift gate summary from trading manager",
                        exc_info=True,
                    )
                else:
                    if gate_payload:
                        risk_section["drift_gate"] = gate_payload

            describe_release = getattr(trading_manager, "describe_release_posture", None)
            if callable(describe_release):
                try:
                    release_payload = describe_release(self._resolve_strategy_id())
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug(
                        "Failed to resolve policy release posture for runtime summary",
                        exc_info=True,
                    )
                else:
                    if release_payload:
                        risk_section["release"] = dict(release_payload)

            describe_release_execution = getattr(
                trading_manager, "describe_release_execution", None
            )
            if callable(describe_release_execution):
                try:
                    release_execution_payload = describe_release_execution()
                except Exception:  # pragma: no cover - diagnostics only
                    self._logger.debug(
                        "Failed to resolve release execution routing for runtime summary",
                        exc_info=True,
                    )
                else:
                    if isinstance(release_execution_payload, Mapping):
                        risk_section["release_execution"] = dict(release_execution_payload)
                    elif release_execution_payload is not None:
                        risk_section["release_execution"] = release_execution_payload

        if self._last_execution_snapshot is not None:
            execution_snapshot = self._last_execution_snapshot
            execution_payload: dict[str, object] = {"snapshot": execution_snapshot.as_dict()}
            markdown = format_execution_markdown(execution_snapshot)
            if markdown:
                execution_payload["markdown"] = markdown
            summary_payload["execution"] = execution_payload

        journal: TimescaleExecutionJournal | None = getattr(self, "execution_journal", None)
        if journal is not None:
            try:
                service_filter = None
                if self._last_execution_snapshot is not None:
                    service_filter = getattr(self._last_execution_snapshot, "service", None)
                strategy_filter = self._resolve_strategy_id()
                journal_payload: dict[str, object] = {}
                recent_records = journal.fetch_recent(
                    limit=5,
                    service=service_filter,
                    strategy_id=strategy_filter,
                )
                if recent_records:
                    journal_payload["recent"] = [record.as_dict() for record in recent_records]
                latest_record = journal.fetch_latest(
                    service=service_filter,
                    strategy_id=strategy_filter,
                )
                if latest_record is not None:
                    journal_payload["latest"] = latest_record.as_dict()
                if journal_payload:
                    summary_payload["execution_journal"] = journal_payload
            except Exception:  # pragma: no cover - diagnostics only
                self._logger.debug("Failed to summarise execution journal", exc_info=True)

        if self._last_backup_snapshot is not None:
            backup_snapshot = self._last_backup_snapshot
            backup_payload: dict[str, object] = {"snapshot": backup_snapshot.as_dict()}
            markdown = format_backup_markdown(backup_snapshot)
            if markdown:
                backup_payload["markdown"] = markdown
            summary_payload["backups"] = backup_payload

        if self._last_backbone_validation_snapshot is not None:
            backbone_validation_snapshot = self._last_backbone_validation_snapshot
            validation_payload: dict[str, object] = {
                "snapshot": backbone_validation_snapshot.as_dict()
            }
            markdown = backbone_validation_snapshot.to_markdown()
            if markdown:
                validation_payload["markdown"] = markdown
            summary_payload["data_backbone_validation"] = validation_payload

        if self._last_backbone_snapshot is not None:
            backbone_snapshot = self._last_backbone_snapshot
            backbone_payload: dict[str, object] = {"snapshot": backbone_snapshot.as_dict()}
            markdown = backbone_snapshot.to_markdown()
            if markdown:
                backbone_payload["markdown"] = markdown
            summary_payload["data_backbone"] = backbone_payload

        if self._last_data_retention_snapshot is not None:
            retention_snapshot = self._last_data_retention_snapshot
            retention_payload: dict[str, object] = {"snapshot": retention_snapshot.as_dict()}
            markdown = format_data_retention_markdown(retention_snapshot)
            if markdown:
                retention_payload["markdown"] = markdown
            summary_payload["data_retention"] = retention_payload

        if self._last_compliance_snapshot is not None:
            compliance_snapshot = self._last_compliance_snapshot
            compliance_payload: dict[str, object] = {"snapshot": compliance_snapshot.as_dict()}
            markdown = compliance_snapshot.to_markdown()
            if markdown:
                compliance_payload["markdown"] = markdown
            summary_payload["compliance_readiness"] = compliance_payload

        if self._last_regulatory_snapshot is not None:
            regulatory_snapshot = self._last_regulatory_snapshot
            regulatory_payload: dict[str, object] = {
                "snapshot": regulatory_snapshot.as_dict(),
                "coverage_ratio": regulatory_snapshot.coverage_ratio,
                "missing_domains": list(regulatory_snapshot.missing_domains),
                "signals": [
                    {
                        "name": signal.name,
                        "status": signal.status.value,
                        "summary": signal.summary,
                        "metadata": dict(signal.metadata),
                        "observed_at": signal.observed_at.isoformat()
                        if signal.observed_at is not None
                        else None,
                    }
                    for signal in regulatory_snapshot.signals
                ],
            }
            if regulatory_snapshot.metadata:
                regulatory_payload["metadata"] = dict(regulatory_snapshot.metadata)
            summary_payload["regulatory_telemetry"] = regulatory_payload

        if self._last_governance_report is not None:
            report = self._last_governance_report
            governance_payload: dict[str, object] = {
                "status": report.status.value,
                "generated_at": report.generated_at.isoformat(),
                "period": {
                    "start": report.period_start.isoformat(),
                    "end": report.period_end.isoformat(),
                },
                "sections": [
                    {
                        "name": section.name,
                        "status": section.status.value,
                        "summary": section.summary,
                        "metadata": dict(section.metadata),
                    }
                    for section in report.sections
                ],
            }
            if report.metadata:
                governance_payload["metadata"] = dict(report.metadata)
            markdown = report.to_markdown()
            if markdown:
                governance_payload["markdown"] = markdown
            summary_payload["governance_report"] = governance_payload

        if self._last_security_snapshot is not None:
            security_snapshot = self._last_security_snapshot
            security_payload: dict[str, object] = {"snapshot": security_snapshot.as_dict()}
            markdown = security_snapshot.to_markdown()
            if markdown:
                security_payload["markdown"] = markdown
            summary_payload["security"] = security_payload

        if self._last_incident_response_snapshot is not None:
            incident_snapshot = self._last_incident_response_snapshot
            incident_payload: dict[str, object] = {"snapshot": incident_snapshot.as_dict()}
            markdown = format_incident_response_markdown(incident_snapshot)
            if markdown:
                incident_payload["markdown"] = markdown
            summary_payload["incident_response"] = incident_payload

        if self._last_cache_snapshot is not None:
            cache_snapshot = self._last_cache_snapshot
            cache_payload: dict[str, object] = {"snapshot": cache_snapshot.as_dict()}
            markdown = cache_snapshot.to_markdown()
            if markdown:
                cache_payload["markdown"] = markdown
            summary_payload["cache"] = cache_payload

        if self._last_kafka_readiness_snapshot is not None:
            kafka_snapshot = self._last_kafka_readiness_snapshot
            kafka_payload: dict[str, object] = {"snapshot": kafka_snapshot.as_dict()}
            markdown = format_kafka_readiness_markdown(kafka_snapshot)
            if markdown:
                kafka_payload["markdown"] = markdown
            summary_payload["kafka_readiness"] = kafka_payload

        if self._last_event_bus_snapshot is not None:
            event_bus_snapshot = self._last_event_bus_snapshot
            event_bus_payload: dict[str, object] = {"snapshot": event_bus_snapshot.as_dict()}
            markdown = format_event_bus_health_markdown(event_bus_snapshot)
            if markdown:
                event_bus_payload["markdown"] = markdown
            summary_payload["event_bus"] = event_bus_payload

        if self._last_system_validation_snapshot is not None:
            system_validation_snapshot = self._last_system_validation_snapshot
            system_validation_payload: dict[str, object] = {
                "snapshot": system_validation_snapshot.as_dict()
            }
            markdown = format_system_validation_markdown(system_validation_snapshot)
            if markdown:
                system_validation_payload["markdown"] = markdown
            summary_payload["system_validation"] = system_validation_payload

        if self._last_sensory_summary is not None:
            sensory_summary = self._last_sensory_summary
            summary_payload["sensory_summary"] = {
                "snapshot": sensory_summary.as_dict(),
                "markdown": sensory_summary.to_markdown(),
            }

        if self._last_sensory_metrics is not None:
            sensory_metrics = self._last_sensory_metrics
            summary_payload["sensory_metrics"] = {
                "snapshot": sensory_metrics.as_dict(),
            }

        if self._last_sensory_drift_snapshot is not None:
            drift_snapshot = self._last_sensory_drift_snapshot
            drift_payload: dict[str, object] = {"snapshot": drift_snapshot.as_dict()}
            markdown = drift_snapshot.to_markdown()
            if markdown:
                drift_payload["markdown"] = markdown
            summary_payload["sensory_drift"] = drift_payload

        if self._last_evolution_experiment_snapshot is not None:
            experiment_snapshot = self._last_evolution_experiment_snapshot
            experiment_payload: dict[str, object] = {"snapshot": experiment_snapshot.as_dict()}
            markdown = format_evolution_experiment_markdown(experiment_snapshot)
            if markdown:
                experiment_payload["markdown"] = markdown
            summary_payload["evolution_experiments"] = experiment_payload

        if self._last_evolution_tuning_snapshot is not None:
            tuning_snapshot = self._last_evolution_tuning_snapshot
            tuning_payload: dict[str, object] = {"snapshot": tuning_snapshot.as_dict()}
            markdown = format_evolution_tuning_markdown(tuning_snapshot)
            if markdown:
                tuning_payload["markdown"] = markdown
            summary_payload["evolution_tuning"] = tuning_payload

        if self._last_strategy_performance_snapshot is not None:
            strategy_snapshot = self._last_strategy_performance_snapshot
            strategy_payload: dict[str, object] = {"snapshot": strategy_snapshot.as_dict()}
            markdown = format_strategy_performance_markdown(strategy_snapshot)
            if markdown:
                strategy_payload["markdown"] = markdown
            summary_payload["strategy_performance"] = strategy_payload

        if self._last_spark_export_snapshot is not None:
            spark_snapshot = self._last_spark_export_snapshot
            spark_payload: dict[str, object] = {"snapshot": spark_snapshot.as_dict()}
            markdown = format_spark_export_markdown(spark_snapshot)
            if markdown:
                spark_payload["markdown"] = markdown
            summary_payload["spark_exports"] = spark_payload

        if self._last_spark_stress_snapshot is not None:
            stress_snapshot = self._last_spark_stress_snapshot
            stress_payload: dict[str, object] = {"snapshot": stress_snapshot.as_dict()}
            markdown = format_spark_stress_markdown(stress_snapshot)
            if markdown:
                stress_payload["markdown"] = markdown
            summary_payload["spark_stress"] = stress_payload

        if self._last_failover_drill_snapshot is not None:
            drill_snapshot = self._last_failover_drill_snapshot
            drill_payload: dict[str, object] = {"snapshot": drill_snapshot.as_dict()}
            markdown = format_failover_drill_markdown(drill_snapshot)
            if markdown:
                drill_payload["markdown"] = markdown
            summary_payload["failover_drill"] = drill_payload

        if self._last_cross_region_snapshot is not None:
            cross_snapshot = self._last_cross_region_snapshot
            cross_payload: dict[str, object] = {"snapshot": cross_snapshot.as_dict()}
            markdown = format_cross_region_markdown(cross_snapshot)
            if markdown:
                cross_payload["markdown"] = markdown
            summary_payload["cross_region_failover"] = cross_payload

        if self._last_compliance_workflow_snapshot is not None:
            workflow_snapshot = self._last_compliance_workflow_snapshot
            workflow_payload: dict[str, object] = {"snapshot": workflow_snapshot.as_dict()}
            markdown = workflow_snapshot.to_markdown()
            if markdown:
                workflow_payload["markdown"] = markdown
            summary_payload["compliance_workflows"] = workflow_payload

        if self._last_professional_snapshot is not None:
            professional_snapshot = self._last_professional_snapshot
            professional_payload: dict[str, object] = {"snapshot": professional_snapshot.as_dict()}
            markdown = professional_snapshot.to_markdown()
            if markdown:
                professional_payload["markdown"] = markdown
            summary_payload["professional_readiness"] = professional_payload

        if self._last_operational_slo_snapshot is not None:
            slo_snapshot = self._last_operational_slo_snapshot
            slo_payload: dict[str, object] = {"snapshot": slo_snapshot.as_dict()}
            markdown = slo_snapshot.to_markdown()
            if markdown:
                slo_payload["markdown"] = markdown
            summary_payload["operational_slos"] = slo_payload

        if (
            self._last_system_validation_snapshot is not None
            or self._last_incident_response_snapshot is not None
            or self._last_operational_slo_snapshot is not None
        ):
            readiness_snapshot = evaluate_operational_readiness(
                system_validation=self._last_system_validation_snapshot,
                incident_response=self._last_incident_response_snapshot,
                drift_snapshot=self._last_sensory_drift_snapshot,
                slo_snapshot=self._last_operational_slo_snapshot,
                metadata={"protocol": self.config.connection_protocol.value},
            )
            summary_payload["operational_readiness"] = {
                "snapshot": readiness_snapshot.as_dict(),
                "markdown": format_operational_readiness_markdown(readiness_snapshot),
            }

        return summary_payload


def _ensure_fix_components(
    config: SystemConfig,
    event_bus: EventBus,
    *,
    task_factory: Callable[[Coroutine[Any, Any, Any], Optional[str]], asyncio.Task[Any]] | None = None,
) -> tuple[FIXConnectionManager, FIXSensoryOrgan, FIXBrokerInterface]:
    logger = logging.getLogger(__name__)
    logger.info(
        "🔧 Setting up LIVE components using '%s' protocol", config.connection_protocol.value
    )
    logger.info("🎯 Configuring FIX protocol components")

    fix_config = cast(SystemConfigProtocol, config)
    fix_connection_manager = FIXConnectionManager(fix_config)

    price_queue: asyncio.Queue[Any] = asyncio.Queue()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()

    price_app = fix_connection_manager.get_application("price")
    if price_app:
        price_app.set_message_queue(price_queue)
    trade_app = fix_connection_manager.get_application("trade")
    if trade_app:
        trade_app.set_message_queue(trade_queue)

    sensory_organ = FIXSensoryOrgan(
        event_bus,
        price_queue,
        config.to_dict(),
        task_factory=task_factory,
    )
    broker_interface = FIXBrokerInterface(
        event_bus,
        trade_queue,
        fix_connection_manager.get_initiator("trade"),
        task_factory=task_factory,
    )

    logger.info("✅ FIX components configured successfully")
    return fix_connection_manager, sensory_organ, broker_interface


def _default_sensors() -> Dict[str, MarketDataSensor]:
    return {
        "why": WhySensor(),
        "what": WhatSensor(),
        "when": WhenSensor(),
        "how": HowSensor(),
        "anomaly": AnomalySensor(),
    }


def _extra_float(extras: Mapping[str, str], key: str, default: float) -> float:
    raw = extras.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _extra_int(extras: Mapping[str, str], key: str, default: int | None) -> int | None:
    raw = extras.get(key)
    if raw is None:
        return default
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return default


def _extra_decimal(extras: Mapping[str, str], key: str, default: Decimal) -> Decimal:
    raw = extras.get(key)
    if raw is None:
        return default
    try:
        return Decimal(str(raw))
    except Exception:
        return default


def _parse_optional_bool_flag(value: str | None) -> bool | None:
    if value is None:
        return None
    normalised = str(value).strip().lower()
    if not normalised or normalised in {"auto", "default"}:
        return None
    if normalised in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if normalised in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return None


def _collect_numeric_parameters(genome: DecisionGenome) -> list[float]:
    params = getattr(genome, "parameters", {}) or {}
    if isinstance(params, Mapping):
        candidate_values = params.values()
    elif hasattr(params, "__dict__"):
        candidate_values = vars(params).values()
    else:
        candidate_values = []

    values: list[float] = []
    for item in candidate_values:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        values.append(number)
    return values


def _bootstrap_evolution_evaluator() -> Callable[[DecisionGenome], Mapping[str, Any]]:
    def _evaluate(genome: DecisionGenome) -> Mapping[str, Any]:
        values = _collect_numeric_parameters(genome)
        if values:
            average = fmean(values)
            volatility = pstdev(values) if len(values) > 1 else 0.0
            total_return = average / len(values)
            max_drawdown = max(0.0, volatility * 0.1)
            fitness = average - max_drawdown
        else:
            average = 0.0
            volatility = 0.0
            total_return = 0.0
            max_drawdown = 0.0
            fitness = 0.0

        metadata = {
            "evaluated": True,
            "evaluation": "bootstrap_average_score",
            "parameter_count": len(values),
            "parameter_mean": float(average),
            "volatility_estimate": float(volatility),
        }

        return {
            "fitness_score": float(fitness),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(average),
            "total_return": float(total_return),
            "volatility": float(volatility),
            "metadata": metadata,
        }

    return _evaluate


def _build_evolution_config(extras: Mapping[str, str]) -> EvolutionConfig:
    population_default = 12
    population = _extra_int(extras, "EVOLUTION_POPULATION_SIZE", population_default) or population_default
    population = max(2, population)

    elite_baseline = max(1, min(population // 4 or 1, population - 1))
    elite = _extra_int(extras, "EVOLUTION_ELITE_COUNT", elite_baseline) or elite_baseline
    elite = max(1, min(elite, population - 1))

    crossover = _extra_float(extras, "EVOLUTION_CROSSOVER_RATE", 0.6)
    mutation = _extra_float(extras, "EVOLUTION_MUTATION_RATE", 0.2)
    crossover = min(max(crossover, 0.0), 1.0)
    mutation = min(max(mutation, 0.0), 1.0)

    max_generations = _extra_int(extras, "EVOLUTION_MAX_GENERATIONS", 100) or 100
    max_generations = max(1, max_generations)

    use_catalogue_hint = _parse_optional_bool_flag(extras.get("EVOLUTION_USE_CATALOGUE"))

    config_kwargs: dict[str, Any] = {
        "population_size": population,
        "elite_count": elite,
        "crossover_rate": crossover,
        "mutation_rate": mutation,
        "max_generations": max_generations,
    }
    if use_catalogue_hint is not None:
        config_kwargs["use_catalogue"] = use_catalogue_hint

    return EvolutionConfig(**config_kwargs)


def _build_evolution_feature_flags(extras: Mapping[str, str]) -> EvolutionFeatureFlags:
    flag_value = extras.get(ADAPTIVE_RUNS_FLAG)
    if flag_value is None:
        return EvolutionFeatureFlags()
    overlay = ChainMap({ADAPTIVE_RUNS_FLAG: flag_value}, os.environ)
    return EvolutionFeatureFlags(env=overlay)


def _build_evolution_orchestrator(
    config: SystemConfig,
    *,
    event_bus: EventBus,
    strategy_registry: StrategyRegistry | None,
) -> EvolutionCycleOrchestrator | None:
    extras = config.extras or {}
    enabled_hint = _parse_optional_bool_flag(extras.get("EVOLUTION_ORCHESTRATOR_ENABLED"))
    if enabled_hint is False:
        return None

    evolution_config = _build_evolution_config(extras)
    engine = EvolutionEngine(config=evolution_config)
    try:
        engine.ensure_population()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Failed to pre-seed evolution population", exc_info=True)

    feature_flags = _build_evolution_feature_flags(extras)
    adaptive_override = _parse_optional_bool_flag(extras.get("EVOLUTION_ADAPTIVE_RUNS_OVERRIDE"))
    evaluator = _bootstrap_evolution_evaluator()

    return EvolutionCycleOrchestrator(
        engine,
        evaluator,
        strategy_registry=strategy_registry,
        event_bus=event_bus,
        adaptive_runs_enabled=adaptive_override,
        feature_flags=feature_flags,
    )


def _extra_symbols(extras: Mapping[str, str], key: str) -> list[str]:
    raw = extras.get(key)
    if not raw:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _extra_bool(extras: Mapping[str, str], key: str, default: bool) -> bool:
    raw = extras.get(key)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _resolve_roi_cost_model(config: SystemConfig, *, initial_equity: float) -> RoiCostModel:
    extras = config.extras or {}
    if config.tier is EmpTier.tier_0:
        base = RoiCostModel.bootstrap_defaults(initial_equity)
    else:
        base = RoiCostModel.institutional_defaults(initial_equity)

    initial_capital = _extra_float(extras, "ROI_INITIAL_CAPITAL", base.initial_capital)
    target_annual = _extra_float(extras, "ROI_TARGET_ANNUAL_ROI", base.target_annual_roi)
    monthly_cost = _extra_float(
        extras,
        "ROI_MONTHLY_OPERATING_COST",
        base.infrastructure_daily_cost * 30.0,
    )
    broker_fee_flat = _extra_float(extras, "ROI_BROKER_FEE_FLAT", base.broker_fee_flat)
    broker_fee_bps = _extra_float(extras, "ROI_BROKER_FEE_BPS", base.broker_fee_bps)

    return RoiCostModel(
        initial_capital=max(0.0, initial_capital),
        target_annual_roi=max(0.0, target_annual),
        infrastructure_daily_cost=max(0.0, monthly_cost) / 30.0,
        broker_fee_flat=max(0.0, broker_fee_flat),
        broker_fee_bps=max(0.0, broker_fee_bps),
    )


def _configure_execution_journal(
    config: SystemConfig,
) -> tuple[TimescaleExecutionJournal | None, CleanupCallback | None]:
    extras = config.extras or {}
    enabled = _extra_bool(
        extras,
        "EXECUTION_JOURNAL_ENABLED",
        config.data_backbone_mode is DataBackboneMode.institutional,
    )
    if not enabled:
        return None, None

    engine = None
    try:
        settings = TimescaleConnectionSettings.from_mapping(extras)
        engine = settings.create_engine()
        TimescaleMigrator(engine).ensure_execution_tables()
        journal = TimescaleExecutionJournal(engine)
    except Exception:
        logger.exception("Failed to configure Timescale execution journal")
        if engine is not None:
            try:
                engine.dispose()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug(
                    "Failed to dispose execution journal engine after init error",
                    exc_info=True,
                )
        return None, None

    def _cleanup() -> None:
        try:
            engine.dispose()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to dispose execution journal engine", exc_info=True)

    return journal, _cleanup


def _resolve_risk_config(
    extras: Mapping[str, str], *, risk_per_trade: float, max_drawdown: float
) -> RiskConfig:
    risk_pct = max(0.0, risk_per_trade)
    drawdown_pct = max(0.0, max_drawdown)
    max_total_default = Decimal("0.5")
    max_leverage_default = Decimal("10.0")
    max_total = _extra_decimal(extras, "RISK_MAX_TOTAL_EXPOSURE_PCT", max_total_default)
    max_leverage = _extra_decimal(extras, "RISK_MAX_LEVERAGE", max_leverage_default)
    min_position = _extra_int(extras, "RISK_MIN_POSITION_SIZE", 1) or 1
    max_position = _extra_int(extras, "RISK_MAX_POSITION_SIZE", 1_000_000) or 1_000_000
    mandatory_stop = _extra_bool(extras, "RISK_MANDATORY_STOP_LOSS", True)
    research_mode = _extra_bool(extras, "RISK_RESEARCH_MODE", False)

    return RiskConfig(
        max_risk_per_trade_pct=Decimal(str(risk_pct)),
        max_total_exposure_pct=max_total,
        max_leverage=max_leverage,
        max_drawdown_pct=Decimal(str(drawdown_pct)),
        min_position_size=int(min_position),
        max_position_size=int(max_position),
        mandatory_stop_loss=mandatory_stop,
        research_mode=research_mode,
    )


def _maybe_configure_redis_client(config: SystemConfig) -> ManagedRedisCache | None:
    if config.data_backbone_mode is not DataBackboneMode.institutional:
        logger.debug(
            "Redis configuration skipped for backbone mode %s",
            config.data_backbone_mode.value,
        )
        return None

    extras = config.extras or {}
    policy = RedisCachePolicy.from_mapping(extras)
    settings = RedisConnectionSettings.from_mapping(extras)
    if not settings.configured:
        logger.debug("No Redis credentials found in extras; using managed in-memory cache")
        return wrap_managed_cache(None, policy=policy, bootstrap=True)

    client = configure_redis_client(settings)
    if client is None:
        logger.warning(
            "Redis configuration present but client unavailable; falling back to in-memory cache (%s)",
            settings.summary(),
        )
        return wrap_managed_cache(None, policy=policy, bootstrap=True)

    return wrap_managed_cache(client, policy=policy)


def _build_trade_compliance_monitor(
    config: SystemConfig, event_bus: EventBus
) -> TradeComplianceMonitor | None:
    extras = config.extras or {}
    enabled = _extra_bool(extras, "COMPLIANCE_MONITOR_ENABLED", True)
    if not enabled:
        logger.info("⚖️ Trade compliance monitor disabled via extras")
        return None

    policy = TradeCompliancePolicy.from_mapping(extras)
    audit_log_path = extras.get("COMPLIANCE_AUDIT_LOG_PATH")
    audit_logger = AuditLogger(log_file=str(audit_log_path)) if audit_log_path else AuditLogger()

    strategy_hint = extras.get("COMPLIANCE_STRATEGY_ID") or extras.get("STRATEGY_ID")
    strategy_id = str(strategy_hint) if strategy_hint else config.run_mode.value

    journal: TimescaleComplianceJournal | None = None
    journal_engine = None
    journal_enabled = _extra_bool(
        extras,
        "COMPLIANCE_JOURNAL_ENABLED",
        config.data_backbone_mode is DataBackboneMode.institutional,
    )
    if journal_enabled:
        try:
            timescale_settings = TimescaleConnectionSettings.from_mapping(extras)
            journal_engine = timescale_settings.create_engine()
            TimescaleMigrator(journal_engine).ensure_compliance_tables()
            journal = TimescaleComplianceJournal(journal_engine)
        except Exception:
            logger.exception("Failed to configure Timescale compliance journal")
            if journal_engine is not None:
                try:
                    journal_engine.dispose()
                except Exception:  # pragma: no cover - defensive cleanup
                    logger.debug(
                        "Failed to dispose compliance journal engine after init error",
                        exc_info=True,
                    )
            journal = None

    monitor = TradeComplianceMonitor(
        event_bus=event_bus,
        policy=policy,
        audit_logger=audit_logger,
        strategy_id=strategy_id,
        snapshot_journal=journal,
    )
    logger.info(
        "⚖️ Trade compliance monitor active: policy=%s channel=%s",
        policy.policy_name,
        policy.report_channel,
    )
    if journal is not None:
        logger.info("🗃️ Timescale compliance journal enabled (telemetry.compliance_audit)")
    return monitor


def _build_kyc_monitor(config: SystemConfig, event_bus: EventBus) -> KycAmlMonitor | None:
    extras = config.extras or {}
    enabled = _extra_bool(extras, "KYC_MONITOR_ENABLED", False)
    if not enabled:
        logger.info("🛡️ KYC/AML monitor disabled via extras")
        return None

    report_channel = str(extras.get("KYC_REPORT_CHANNEL") or "telemetry.compliance.kyc")
    history_limit = _extra_int(extras, "KYC_HISTORY_LIMIT", 20) or 20
    audit_log_path = extras.get("KYC_AUDIT_LOG_PATH")
    audit_logger = AuditLogger(log_file=str(audit_log_path)) if audit_log_path else AuditLogger()

    strategy_hint = (
        extras.get("KYC_STRATEGY_ID")
        or extras.get("COMPLIANCE_STRATEGY_ID")
        or extras.get("STRATEGY_ID")
    )
    strategy_id = str(strategy_hint) if strategy_hint else config.run_mode.value

    journal: TimescaleKycJournal | None = None
    journal_engine = None
    journal_enabled = _extra_bool(
        extras,
        "KYC_JOURNAL_ENABLED",
        config.data_backbone_mode is DataBackboneMode.institutional,
    )
    if journal_enabled:
        try:
            timescale_settings = TimescaleConnectionSettings.from_mapping(extras)
            journal_engine = timescale_settings.create_engine()
            TimescaleMigrator(journal_engine).ensure_compliance_tables()
            journal = TimescaleKycJournal(journal_engine)
        except Exception:
            logger.exception("Failed to configure Timescale KYC journal")
            if journal_engine is not None:
                try:
                    journal_engine.dispose()
                except Exception:  # pragma: no cover - defensive cleanup
                    logger.debug(
                        "Failed to dispose KYC journal engine after init error",
                        exc_info=True,
                    )
            journal = None

    monitor = KycAmlMonitor(
        event_bus=event_bus,
        report_channel=report_channel,
        audit_logger=audit_logger,
        strategy_id=strategy_id,
        snapshot_journal=journal,
        history_limit=history_limit,
    )
    logger.info(
        "🛡️ KYC/AML monitor active: channel=%s history=%d",
        report_channel,
        history_limit,
    )
    if journal is not None:
        logger.info("🗃️ Timescale KYC journal enabled (telemetry.compliance_kyc)")
    return monitor


def _configure_institutional_connectors(
    config: SystemConfig,
    *,
    redis_client: ManagedRedisCache | None = None,
) -> tuple[dict[str, MarketDataConnector], list[CleanupCallback]]:
    """Build Timescale-backed connectors when institutional mode is active."""

    if config.data_backbone_mode is not DataBackboneMode.institutional:
        return {}, []

    extras = config.extras or {}
    settings = TimescaleConnectionSettings.from_mapping(extras)

    try:
        engine = settings.create_engine()
    except Exception:
        logger.exception(
            "Failed to create Timescale engine for institutional connectors (url=%s)",
            settings.url,
        )
        return {}, []

    reader = TimescaleReader(engine)
    query_cache = TimescaleQueryCache(reader, redis_client) if redis_client else None
    macro_service: TimescaleMacroEventService | None = None
    if query_cache is not None and query_cache.enabled and redis_client is not None:
        logger.info(
            "🧠 Timescale query cache enabled via Redis namespace %s (ttl=%s)",
            redis_client.policy.namespace,
            redis_client.policy.ttl_seconds,
        )
    try:
        macro_service = TimescaleMacroEventService(reader, query_cache)
    except Exception:  # pragma: no cover - enrichment should not block runtime assembly
        logger.exception("Failed to initialise Timescale macro event service")
        macro_service = None
    else:
        logger.info(
            "🌐 Timescale macro enrichment enabled (lookback=%sh, lookahead=%sh)",
            round(macro_service.lookback.total_seconds() / 3600.0, 2),
            round(macro_service.lookahead.total_seconds() / 3600.0, 2),
        )

    connectors: dict[str, MarketDataConnector] = {
        "timescale_intraday": TimescaleIntradayTradeConnector(
            reader, cache=query_cache, macro_service=macro_service
        ),
        "timescale_daily": TimescaleDailyBarConnector(
            reader, cache=query_cache, macro_service=macro_service
        ),
    }

    logger.info("🗄️ Timescale market data connectors ready: %s", list(connectors.keys()))

    def _dispose() -> None:
        engine.dispose()

    return connectors, [_dispose]


def _build_bootstrap_runtime(
    config: SystemConfig,
    bus: EventBus,
    *,
    redis_client: Any | None = None,
    connectors: Mapping[str, MarketDataConnector] | None = None,
    strategy_registry: StrategyRegistry | None = None,
) -> BootstrapRuntime:
    extras = config.extras or {}
    symbols = _extra_symbols(extras, "BOOTSTRAP_SYMBOLS") or ["EURUSD"]
    tick_interval = _extra_float(extras, "BOOTSTRAP_TICK_INTERVAL", 2.5)
    max_ticks = _extra_int(extras, "BOOTSTRAP_MAX_TICKS", None)
    buy_threshold = _extra_float(extras, "BOOTSTRAP_BUY_THRESHOLD", 0.25)
    sell_threshold = _extra_float(extras, "BOOTSTRAP_SELL_THRESHOLD", 0.25)
    quantity = _extra_decimal(extras, "BOOTSTRAP_ORDER_SIZE", Decimal("1"))
    stop_loss_pct = _extra_float(extras, "BOOTSTRAP_STOP_LOSS_PCT", 0.01)
    risk_per_trade = _extra_float(extras, "BOOTSTRAP_RISK_PER_TRADE", 0.02)
    max_positions = _extra_int(extras, "BOOTSTRAP_MAX_POSITIONS", 5) or 5
    max_drawdown = _extra_float(extras, "BOOTSTRAP_MAX_DRAWDOWN", 0.1)
    initial_equity = _extra_float(extras, "BOOTSTRAP_INITIAL_EQUITY", 100_000.0)
    min_confidence = _extra_float(extras, "BOOTSTRAP_MIN_CONFIDENCE", 0.05)
    min_liq_conf = _extra_float(extras, "BOOTSTRAP_MIN_LIQ_CONF", 0.25)
    strategy_id = extras.get("BOOTSTRAP_STRATEGY_ID", "bootstrap-strategy")
    roi_cost_model = _resolve_roi_cost_model(config, initial_equity=initial_equity)
    risk_config = _resolve_risk_config(
        extras, risk_per_trade=risk_per_trade, max_drawdown=max_drawdown
    )

    release_manager: LedgerReleaseManager | None = None
    diary_store: DecisionDiaryStore | None = None
    diary_path_resolved: Path | None = None
    diary_path_hint = extras.get("DECISION_DIARY_PATH") or extras.get("DECISION_DIARY_ARTIFACT")
    probe_registry_hint = extras.get("PROBE_REGISTRY_PATH")
    if diary_path_hint:
        diary_path = Path(str(diary_path_hint)).expanduser()
        diary_path_resolved = diary_path
        probe_registry: ProbeRegistry | None = None
        if probe_registry_hint:
            registry_path = Path(str(probe_registry_hint)).expanduser()
            try:
                probe_registry = ProbeRegistry.from_file(registry_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load probe registry",
                    exc_info=exc,
                    extra={"probe_registry_path": str(registry_path)},
                )
        try:
            diary_store = DecisionDiaryStore(
                diary_path,
                probe_registry=probe_registry,
                event_bus=bus,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialise decision diary store",
                exc_info=exc,
                extra={"decision_diary_path": str(diary_path)},
            )
        else:
            logger.info(
                "📝 Decision diary store ready",
                extra={"decision_diary_path": str(diary_path)},
            )
    ledger_path_hint = extras.get("POLICY_LEDGER_PATH") or extras.get("POLICY_LEDGER_ARTIFACT")
    if ledger_path_hint:
        ledger_path = Path(str(ledger_path_hint)).expanduser()
        try:
            store = PolicyLedgerStore(ledger_path)
        except ValueError as exc:
            logger.warning(
                "Failed to initialise policy ledger store: %s",
                exc,
                extra={"policy_ledger_path": str(ledger_path)},
            )
        else:
            default_stage = PolicyLedgerStage.EXPERIMENT
            stage_hint = extras.get("POLICY_LEDGER_DEFAULT_STAGE")
            if stage_hint:
                try:
                    default_stage = PolicyLedgerStage.from_value(stage_hint)
                except ValueError:
                    logger.warning(
                        "Unknown policy ledger default stage %r; using %s",
                        stage_hint,
                        default_stage.value,
                        extra={"policy_ledger_path": str(ledger_path)},
                    )
            evidence_resolver = diary_store.exists if diary_store else None
            release_manager = LedgerReleaseManager(
                store,
                default_stage=default_stage,
                evidence_resolver=evidence_resolver,
            )
            logger.info(
                "📘 Policy ledger release manager enabled",
                extra={
                    "policy_ledger_path": str(ledger_path),
                    "default_stage": default_stage.value,
                    "decision_diary_path": str(diary_path_resolved) if diary_path_resolved else None,
                },
            )

    orchestrator = _build_evolution_orchestrator(
        config,
        event_bus=bus,
        strategy_registry=strategy_registry,
    )

    evolution_interval = _extra_int(extras, "EVOLUTION_CYCLE_INTERVAL", 5) or 5
    if evolution_interval <= 0:
        evolution_interval = 1

    return BootstrapRuntime(
        event_bus=bus,
        symbols=symbols,
        connectors=connectors,
        tick_interval=tick_interval,
        max_ticks=max_ticks,
        strategy_id=strategy_id,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        requested_quantity=quantity,
        stop_loss_pct=stop_loss_pct,
        risk_per_trade=None,
        max_open_positions=max_positions,
        max_daily_drawdown=None,
        initial_equity=initial_equity,
        min_intent_confidence=min_confidence,
        min_liquidity_confidence=min_liq_conf,
        redis_client=redis_client,
        roi_cost_model=roi_cost_model,
        risk_config=risk_config,
        release_manager=release_manager,
        evolution_orchestrator=orchestrator,
        evolution_cycle_interval=evolution_interval,
    )


async def build_professional_predator_app(
    *,
    config: Optional[SystemConfig] = None,
    event_bus: Optional[EventBus] = None,
) -> ProfessionalPredatorApp:
    """Assemble a ProfessionalPredatorApp with all mandatory dependencies."""

    cfg = config or SystemConfig.from_env()
    extras_mapping = cfg.extras or {}
    tracing_settings = parse_opentelemetry_settings(extras_mapping)
    event_bus_tracer = configure_event_bus_tracer(tracing_settings)
    runtime_tracer = configure_runtime_tracer(tracing_settings)

    task_supervisor = TaskSupervisor(namespace="professional-runtime")

    def _task_factory(
        coro: Coroutine[Any, Any, Any],
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> asyncio.Task[Any]:
        return task_supervisor.create(coro, name=name, metadata=metadata)

    if event_bus is None:
        bus = EventBus(task_factory=_task_factory, tracer=event_bus_tracer)
    else:
        bus = event_bus
        set_factory = getattr(bus, "set_task_factory", None)
        if callable(set_factory):
            try:
                set_factory(_task_factory)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Failed to update event bus task factory", exc_info=True)
        set_tracer = getattr(bus, "set_tracer", None)
        if callable(set_tracer):
            try:
                set_tracer(event_bus_tracer)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug("Failed to update event bus tracer", exc_info=True)

    # Enforce guardrails before instantiating live components
    SafetyManager.from_config(cfg.to_dict()).enforce()

    sensors = _default_sensors()
    compliance_monitor = _build_trade_compliance_monitor(cfg, bus)
    kyc_monitor = _build_kyc_monitor(cfg, bus)
    execution_journal, execution_cleanup = _configure_execution_journal(cfg)

    registry: StrategyRegistry | None = None
    registry_path = (cfg.extras or {}).get("STRATEGY_REGISTRY_PATH") if cfg.extras else None
    if registry_path:
        try:
            registry = StrategyRegistry(db_path=str(registry_path))
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to initialise strategy registry", exc_info=True)
            registry = None

    if cfg.connection_protocol is ConnectionProtocol.fix:
        fix_manager, sensory_organ, broker_interface = _ensure_fix_components(
            cfg, bus, task_factory=_task_factory
        )
        app = ProfessionalPredatorApp(
            config=cfg,
            event_bus=bus,
            sensory_organ=sensory_organ,
            broker_interface=broker_interface,
            fix_connection_manager=fix_manager,
            sensors=sensors,
            compliance_monitor=compliance_monitor,
            kyc_monitor=kyc_monitor,
            execution_journal=execution_journal,
            strategy_registry=registry,
            task_supervisor=task_supervisor,
            runtime_tracer=runtime_tracer,
        )
        dropcopy_listener = FixDropcopyReconciler(
            event_bus=bus,
            broker_order_lookup=broker_interface.get_all_orders,
            task_factory=_task_factory,
        )
        pilot = FixIntegrationPilot(
            connection_manager=fix_manager,
            sensory_organ=sensory_organ,
            broker_interface=broker_interface,
            task_supervisor=app.task_supervisor,
            event_bus=bus,
            compliance_monitor=compliance_monitor,
            trading_manager=getattr(sensory_organ, "trading_manager", None),
            dropcopy_listener=dropcopy_listener,
            logger=logger,
        )
        app.attach_fix_pilot(pilot)
        app.add_cleanup_callback(fix_manager.stop_sessions)
        if compliance_monitor is not None:
            app.add_cleanup_callback(compliance_monitor.close)
        if kyc_monitor is not None:
            app.add_cleanup_callback(kyc_monitor.close)
        if execution_cleanup is not None:
            app.add_cleanup_callback(execution_cleanup)
        if registry is not None:
            app.add_cleanup_callback(registry.close)

        async def _stop_dropcopy() -> None:
            await dropcopy_listener.stop()

        app.add_cleanup_callback(_stop_dropcopy)
        return app

    if cfg.connection_protocol in (ConnectionProtocol.bootstrap, ConnectionProtocol.paper):
        redis_client = _maybe_configure_redis_client(cfg)
        connector_map, connector_cleanups = _configure_institutional_connectors(
            cfg, redis_client=redis_client
        )
        kafka_bridge: KafkaIngestEventConsumer | None = None
        if cfg.data_backbone_mode is DataBackboneMode.institutional:
            kafka_settings = KafkaConnectionSettings.from_mapping(cfg.extras or {})
            kafka_bridge = create_ingest_event_consumer(
                kafka_settings,
                cfg.extras or {},
                event_bus=bus,
            )
        runtime = _build_bootstrap_runtime(
            cfg,
            bus,
            redis_client=redis_client,
            connectors=connector_map or None,
            strategy_registry=registry,
        )
        app = ProfessionalPredatorApp(
            config=cfg,
            event_bus=bus,
            sensory_organ=runtime,
            broker_interface=None,
            fix_connection_manager=None,
            sensors=sensors,
            compliance_monitor=compliance_monitor,
            kyc_monitor=kyc_monitor,
            redis_client=redis_client,
            execution_journal=execution_journal,
            strategy_registry=registry,
            task_supervisor=task_supervisor,
            runtime_tracer=runtime_tracer,
        )
        for cleanup in connector_cleanups:
            app.add_cleanup_callback(cleanup)
        if compliance_monitor is not None:
            app.add_cleanup_callback(compliance_monitor.close)
        if kyc_monitor is not None:
            app.add_cleanup_callback(kyc_monitor.close)
        if execution_cleanup is not None:
            app.add_cleanup_callback(execution_cleanup)
        if registry is not None:
            app.add_cleanup_callback(registry.close)
        if kafka_bridge is not None:
            stop_event = asyncio.Event()
            bridge_metadata = app._build_kafka_bridge_metadata(kafka_bridge)
            app._kafka_bridge_metadata = bridge_metadata

            async def _consume_kafka() -> None:
                try:
                    await kafka_bridge.run_forever(stop_event)
                finally:
                    kafka_bridge.close()

            app.create_background_task(
                _consume_kafka(),
                name="kafka-ingest-bridge",
                metadata=bridge_metadata,
            )

            async def _shutdown_kafka() -> None:
                stop_event.set()
                kafka_bridge.close()
                if app._kafka_bridge_metadata is not None:
                    updated = dict(app._kafka_bridge_metadata)
                    updated.setdefault("stopped_at", datetime.now().isoformat())
                    app._kafka_bridge_metadata = updated

            app.add_cleanup_callback(_shutdown_kafka)
            logger.info("📡 Kafka ingest consumer bridge active: %s", kafka_bridge.summary())
        return app

    raise ValueError(
        f"Unsupported connection protocol: {cfg.connection_protocol.value}. "
        "Supported protocols: bootstrap, paper, fix."
    )
