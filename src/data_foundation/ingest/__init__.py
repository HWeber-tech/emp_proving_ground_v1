"""Lazy re-exports for ingest helpers to avoid import cycles during bootstrap."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "InstitutionalIngestConfig",
    "TimescaleIngestRecoverySettings",
    "TimescaleCrossRegionSettings",
    "build_institutional_ingest_config",
    "DEFAULT_FRESHNESS_SLA_SECONDS",
    "DEFAULT_MIN_ROWS",
    "IngestHealthCheck",
    "IngestHealthReport",
    "IngestHealthStatus",
    "evaluate_ingest_health",
    "IngestFailoverPolicy",
    "IngestFailoverDecision",
    "decide_ingest_failover",
    "IngestRecoveryRecommendation",
    "plan_ingest_recovery",
    "CompositeIngestPublisher",
    "EventBusIngestPublisher",
    "combine_ingest_publishers",
    "IngestDimensionMetrics",
    "IngestMetricsSnapshot",
    "summarise_ingest_metrics",
    "IngestObservabilityDimension",
    "IngestObservabilitySnapshot",
    "build_ingest_observability_snapshot",
    "FeedAnomaly",
    "FeedAnomalySeverity",
    "IngestQualityCheck",
    "IngestQualityReport",
    "IngestQualityStatus",
    "evaluate_ingest_quality",
    "FalseTickAnomaly",
    "FalseTickSeverity",
    "detect_feed_anomalies",
    "detect_false_ticks",
    "IngestSchedulerSnapshot",
    "IngestSchedulerStatus",
    "build_scheduler_snapshot",
    "format_scheduler_markdown",
    "publish_scheduler_snapshot",
    "AggregationMetadata",
    "AggregationResult",
    "CoverageValidator",
    "CrossSourceDriftValidator",
    "DataQualityFinding",
    "DataQualitySeverity",
    "MultiSourceAggregator",
    "ProviderContribution",
    "ProviderSnapshot",
    "ProviderSpec",
    "StalenessValidator",
    "ProductionIngestSlice",
]


_MODULE_MAP = {
    "InstitutionalIngestConfig": "src.data_foundation.ingest.configuration",
    "TimescaleIngestRecoverySettings": "src.data_foundation.ingest.configuration",
    "TimescaleCrossRegionSettings": "src.data_foundation.ingest.configuration",
    "build_institutional_ingest_config": "src.data_foundation.ingest.configuration",
    "DEFAULT_FRESHNESS_SLA_SECONDS": "src.data_foundation.ingest.health",
    "DEFAULT_MIN_ROWS": "src.data_foundation.ingest.health",
    "IngestHealthCheck": "src.data_foundation.ingest.health",
    "IngestHealthReport": "src.data_foundation.ingest.health",
    "IngestHealthStatus": "src.data_foundation.ingest.health",
    "evaluate_ingest_health": "src.data_foundation.ingest.health",
    "IngestFailoverPolicy": "src.data_foundation.ingest.failover",
    "IngestFailoverDecision": "src.data_foundation.ingest.failover",
    "decide_ingest_failover": "src.data_foundation.ingest.failover",
    "IngestRecoveryRecommendation": "src.data_foundation.ingest.recovery",
    "plan_ingest_recovery": "src.data_foundation.ingest.recovery",
    "CompositeIngestPublisher": "src.data_foundation.ingest.telemetry",
    "EventBusIngestPublisher": "src.data_foundation.ingest.telemetry",
    "combine_ingest_publishers": "src.data_foundation.ingest.telemetry",
    "IngestDimensionMetrics": "src.data_foundation.ingest.metrics",
    "IngestMetricsSnapshot": "src.data_foundation.ingest.metrics",
    "summarise_ingest_metrics": "src.data_foundation.ingest.metrics",
    "IngestObservabilityDimension": "src.data_foundation.ingest.observability",
    "IngestObservabilitySnapshot": "src.data_foundation.ingest.observability",
    "build_ingest_observability_snapshot": "src.data_foundation.ingest.observability",
    "FeedAnomaly": "src.data_foundation.ingest.anomaly_detection",
    "FeedAnomalySeverity": "src.data_foundation.ingest.anomaly_detection",
    "IngestQualityCheck": "src.data_foundation.ingest.quality",
    "IngestQualityReport": "src.data_foundation.ingest.quality",
    "IngestQualityStatus": "src.data_foundation.ingest.quality",
    "evaluate_ingest_quality": "src.data_foundation.ingest.quality",
    "FalseTickAnomaly": "src.data_foundation.ingest.anomaly_detection",
    "FalseTickSeverity": "src.data_foundation.ingest.anomaly_detection",
    "detect_feed_anomalies": "src.data_foundation.ingest.anomaly_detection",
    "detect_false_ticks": "src.data_foundation.ingest.anomaly_detection",
    "IngestSchedulerSnapshot": "src.data_foundation.ingest.scheduler_telemetry",
    "IngestSchedulerStatus": "src.data_foundation.ingest.scheduler_telemetry",
    "build_scheduler_snapshot": "src.data_foundation.ingest.scheduler_telemetry",
    "format_scheduler_markdown": "src.data_foundation.ingest.scheduler_telemetry",
    "publish_scheduler_snapshot": "src.data_foundation.ingest.scheduler_telemetry",
    "AggregationMetadata": "src.data_foundation.ingest.multi_source",
    "AggregationResult": "src.data_foundation.ingest.multi_source",
    "CoverageValidator": "src.data_foundation.ingest.multi_source",
    "CrossSourceDriftValidator": "src.data_foundation.ingest.multi_source",
    "DataQualityFinding": "src.data_foundation.ingest.multi_source",
    "DataQualitySeverity": "src.data_foundation.ingest.multi_source",
    "MultiSourceAggregator": "src.data_foundation.ingest.multi_source",
    "ProviderContribution": "src.data_foundation.ingest.multi_source",
    "ProviderSnapshot": "src.data_foundation.ingest.multi_source",
    "ProviderSpec": "src.data_foundation.ingest.multi_source",
    "StalenessValidator": "src.data_foundation.ingest.multi_source",
    "ProductionIngestSlice": "src.data_foundation.ingest.production_slice",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__ + list(globals().keys())))
