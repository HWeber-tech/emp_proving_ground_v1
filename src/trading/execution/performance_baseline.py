"""Helpers for capturing trading performance baselines."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, MutableMapping, Sequence, TYPE_CHECKING

from .performance_report import (
    build_execution_performance_report,
    build_performance_health_report,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking only
    from src.trading.trading_manager import TradingManager

UTC = timezone.utc


def collect_performance_baseline(
    manager: "TradingManager",
    *,
    max_processing_ms: float = 250.0,
    max_lag_ms: float = 250.0,
    backlog_threshold_ms: float | None = None,
    max_cpu_percent: float | None = None,
    max_memory_mb: float | None = None,
    max_memory_percent: float | None = None,
) -> Mapping[str, Any]:
    """Capture execution, throughput, and resource snapshots for baselining."""

    execution_stats_raw = manager.get_execution_stats()
    execution_stats = _normalise_payload(execution_stats_raw)

    throughput_health_raw = manager.assess_throughput_health(
        max_processing_ms=max_processing_ms,
        max_lag_ms=max_lag_ms,
    )
    throughput_health = _normalise_payload(throughput_health_raw)

    performance_health_raw = manager.assess_performance_health(
        max_processing_ms=max_processing_ms,
        max_lag_ms=max_lag_ms,
        backlog_threshold_ms=backlog_threshold_ms,
        max_cpu_percent=max_cpu_percent,
        max_memory_mb=max_memory_mb,
        max_memory_percent=max_memory_percent,
    )
    performance_health = _normalise_payload(performance_health_raw)

    throttle_summary = performance_health.get("throttle")
    resource_snapshot = performance_health.get("resource")
    backlog_snapshot = performance_health.get("backlog")
    throttle_scopes = manager.get_trade_throttle_scope_snapshots()

    reports: MutableMapping[str, str] = {}
    reports["execution"] = build_execution_performance_report(execution_stats)
    reports["performance"] = build_performance_health_report(performance_health)

    baseline: MutableMapping[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "healthy": bool(performance_health.get("healthy")),
        "execution": execution_stats,
        "throughput": throughput_health,
        "performance": performance_health,
        "reports": reports,
    }
    baseline["backlog"] = backlog_snapshot
    baseline["resource"] = resource_snapshot
    baseline["throttle"] = throttle_summary
    baseline["throttle_scopes"] = list(throttle_scopes)

    return baseline


def _normalise_payload(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _normalise_payload(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_payload(item) for item in value]
    return value
