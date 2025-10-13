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
    trade_throttle_snapshot = manager.get_trade_throttle_snapshot()

    reports: MutableMapping[str, str] = {}
    reports["execution"] = build_execution_performance_report(execution_stats)
    reports["performance"] = build_performance_health_report(performance_health)

    options = _build_options_payload(
        max_processing_ms=max_processing_ms,
        max_lag_ms=max_lag_ms,
        backlog_threshold_ms=backlog_threshold_ms,
        max_cpu_percent=max_cpu_percent,
        max_memory_mb=max_memory_mb,
        max_memory_percent=max_memory_percent,
        trade_throttle_snapshot=trade_throttle_snapshot,
    )

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
    baseline["options"] = options

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


def _build_options_payload(
    *,
    max_processing_ms: float,
    max_lag_ms: float,
    backlog_threshold_ms: float | None,
    max_cpu_percent: float | None,
    max_memory_mb: float | None,
    max_memory_percent: float | None,
    trade_throttle_snapshot: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    options: MutableMapping[str, Any] = {
        "max_processing_ms": float(max_processing_ms),
        "max_lag_ms": float(max_lag_ms),
        "backlog_threshold_ms": (
            float(backlog_threshold_ms)
            if backlog_threshold_ms is not None
            else None
        ),
        "max_cpu_percent": (
            float(max_cpu_percent) if max_cpu_percent is not None else None
        ),
        "max_memory_mb": (
            float(max_memory_mb) if max_memory_mb is not None else None
        ),
        "max_memory_percent": (
            float(max_memory_percent) if max_memory_percent is not None else None
        ),
    }

    throttle_options = _coerce_throttle_options(trade_throttle_snapshot)
    options["trade_throttle"] = throttle_options
    return options


def _coerce_throttle_options(
    snapshot: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(snapshot, Mapping):
        return None

    metadata = snapshot.get("metadata")
    metadata_mapping: Mapping[str, Any] | None
    if isinstance(metadata, Mapping):
        metadata_mapping = metadata
    else:
        metadata_mapping = None

    config: MutableMapping[str, Any] = {}

    name = snapshot.get("name")
    if isinstance(name, str) and name.strip():
        config["name"] = name

    if metadata_mapping is not None:
        max_trades = metadata_mapping.get("max_trades")
        if isinstance(max_trades, (int, float)):
            config["max_trades"] = int(max_trades)

        window_seconds = metadata_mapping.get("window_seconds")
        if isinstance(window_seconds, (int, float)):
            config["window_seconds"] = float(window_seconds)

        cooldown_seconds = metadata_mapping.get("cooldown_seconds")
        if isinstance(cooldown_seconds, (int, float)) and cooldown_seconds > 0:
            config["cooldown_seconds"] = float(cooldown_seconds)

        min_spacing_seconds = metadata_mapping.get("min_spacing_seconds")
        if isinstance(min_spacing_seconds, (int, float)) and min_spacing_seconds > 0:
            config["min_spacing_seconds"] = float(min_spacing_seconds)

        max_notional = metadata_mapping.get("max_notional")
        if isinstance(max_notional, (int, float)):
            config["max_notional"] = float(max_notional)

        multiplier = snapshot.get("multiplier")
        if multiplier is None and metadata_mapping.get("multiplier") is not None:
            multiplier = metadata_mapping.get("multiplier")
        if isinstance(multiplier, (int, float)):
            config["multiplier"] = float(multiplier)

        scope = metadata_mapping.get("scope")
        if isinstance(scope, Mapping):
            config["scope"] = {str(key): value for key, value in scope.items()}

    return dict(config) if config else None
