"""Helpers for executing end-to-end paper trading simulations.

This module wires the Bootstrap runtime into the paper trading adapter so that
roadmap acceptance tests can execute a real HTTP round-trip against a broker
simulation.  The helper is intentionally lightweight – callers provide a
``SystemConfig`` populated with the desired extras and the function waits until
the runtime submits at least one order (or a timeout is reached).

The resulting report captures broker orders, failure telemetry, decision diary
entry counts, and release posture summaries so that higher-level automation can
persist evidence for sign-off reviews.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)

import inspect
import math

from src.core.event_bus import EventBus
from src.governance.system_config import SystemConfig
from src.operations.incident_response import (
    IncidentResponsePolicy,
    IncidentResponseState,
    evaluate_incident_response,
)
from src.runtime.predator_app import _build_bootstrap_runtime
from src.trading.execution.paper_broker_adapter import PaperBrokerExecutionAdapter
from src.trading.execution.release_router import ReleaseAwareExecutionRouter

logger = logging.getLogger(__name__)

__all__ = [
    "PaperTradingSimulationReport",
    "PaperTradingSimulationProgress",
    "run_paper_trading_simulation",
]


@dataclass(slots=True)
class PaperTradingSimulationReport:
    """Summary of an executed paper trading simulation."""

    orders: list[Mapping[str, Any]] = field(default_factory=list)
    errors: list[Mapping[str, Any]] = field(default_factory=list)
    decisions: int = 0
    diary_entries: int = 0
    runtime_seconds: float = 0.0
    order_summary: Mapping[str, Any] | None = None
    paper_broker: Mapping[str, Any] | None = None
    paper_metrics: Mapping[str, Any] | None = None
    paper_failover: Mapping[str, Any] | None = None
    portfolio_state: Mapping[str, Any] | None = None
    performance: Mapping[str, Any] | None = None
    execution_stats: Mapping[str, Any] | None = None
    performance_health: Mapping[str, Any] | None = None
    strategy_summary: Mapping[str, Any] | None = None
    release: Mapping[str, Any] | None = None
    trade_throttle: Mapping[str, Any] | None = None
    trade_throttle_scopes: Sequence[Mapping[str, Any]] | None = None
    trade_throttle_events: Sequence[Mapping[str, Any]] = field(default_factory=list)
    incident_response: Mapping[str, Any] | None = None

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable representation of the report."""

        payload: MutableMapping[str, Any] = {
            "orders": [dict(order) for order in self.orders],
            "errors": [dict(error) for error in self.errors],
            "decisions": int(self.decisions),
            "diary_entries": int(self.diary_entries),
            "runtime_seconds": float(self.runtime_seconds),
        }
        if self.paper_broker is not None:
            payload["paper_broker"] = dict(self.paper_broker)
        if self.paper_metrics is not None:
            payload["paper_metrics"] = dict(self.paper_metrics)
        if self.paper_failover is not None:
            payload["paper_failover"] = dict(self.paper_failover)
        if self.order_summary is not None:
            payload["order_summary"] = _serialise_runtime_value(self.order_summary)
        if self.portfolio_state is not None:
            payload["portfolio_state"] = dict(self.portfolio_state)
        if self.performance is not None:
            payload["performance"] = dict(self.performance)
        if self.execution_stats is not None:
            payload["execution_stats"] = dict(self.execution_stats)
        if self.performance_health is not None:
            payload["performance_health"] = dict(self.performance_health)
        if self.strategy_summary is not None:
            payload["strategy_summary"] = dict(self.strategy_summary)
        if self.release is not None:
            payload["release"] = dict(self.release)
        if self.trade_throttle is not None:
            payload["trade_throttle"] = dict(self.trade_throttle)
        if self.trade_throttle_scopes is not None:
            payload["trade_throttle_scopes"] = [
                dict(scope) for scope in self.trade_throttle_scopes
            ]
        if self.trade_throttle_events:
            payload["trade_throttle_events"] = [
                dict(event) for event in self.trade_throttle_events
            ]
        if self.incident_response is not None:
            payload["incident_response"] = dict(self.incident_response)
        return payload


@dataclass(slots=True)
class PaperTradingSimulationProgress:
    """Incremental snapshot emitted while running a paper trading simulation."""

    timestamp: datetime
    runtime_seconds: float
    orders_observed: int
    errors_observed: int
    decisions_observed: int
    paper_metrics: Mapping[str, Any] | None = None
    failover: Mapping[str, Any] | None = None
    last_order: Mapping[str, Any] | None = None
    last_error: Mapping[str, Any] | None = None


async def run_paper_trading_simulation(
    config: SystemConfig,
    *,
    min_orders: int = 1,
    max_runtime: float | None = 60.0,
    poll_interval: float = 0.5,
    stop_when_complete: bool = True,
    report_path: str | Path | None = None,
    stop_event: asyncio.Event | None = None,
    progress_callback: Callable[[PaperTradingSimulationProgress], Awaitable[None] | None] | None = None,
    progress_interval: float | None = None,
) -> PaperTradingSimulationReport:
    """Execute the bootstrap runtime until paper orders are observed.

    Parameters
    ----------
    config:
        Fully resolved ``SystemConfig`` describing the runtime.  The caller must
        ensure paper-trading extras are supplied (API URL, ledger path, diary
        path, etc.).
    min_orders:
        Stop the simulation once at least this many paper orders have been
        observed.  Set to ``0`` to run purely on time-based criteria.
    max_runtime:
        Optional guard to stop the simulation after the specified number of
        seconds even if the order quota was not achieved.  ``None`` disables the
        timeout.
    poll_interval:
        Sampling cadence (in seconds) for checking the broker adapter telemetry.
    stop_when_complete:
        When ``True`` (default) the runtime is shut down immediately after the
        order quota is satisfied.
    report_path:
        Optional filesystem path where a JSON serialisation of the simulation
        report should be written.  Parent directories are created automatically.
    stop_event:
        Optional asyncio event that, when set, triggers a graceful shutdown of
        the simulation loop before ``max_runtime`` elapses.  Useful for
        long-running simulations that must respond to OS signals.
    progress_callback:
        Optional callable invoked periodically with a
        :class:`PaperTradingSimulationProgress` snapshot while the simulation
        runs.  The callback may be synchronous or async; any errors are logged
        and ignored so execution continues.
    progress_interval:
        Minimum number of seconds between progress callback invocations.  When
        omitted or non-positive a default interval of five seconds is used.

    Returns
    -------
    PaperTradingSimulationReport
        A structured payload describing orders, errors, decision counts, and
        captured paper broker metadata.
    """

    bus = EventBus()
    runtime, cleanups = _build_bootstrap_runtime(config, bus)

    paper_engine = getattr(runtime.trading_manager, "_live_engine", None)
    if paper_engine is None or not isinstance(paper_engine, PaperBrokerExecutionAdapter):
        raise RuntimeError(
            "Paper broker adapter is not configured; ensure paper trading extras "
            "and ledger configuration promote strategies into limited_live."
        )

    router = getattr(runtime.trading_manager, "execution_engine", None)
    if isinstance(router, ReleaseAwareExecutionRouter):
        try:
            router.configure_engines(
                paper_engine=paper_engine,
                pilot_engine=paper_engine,
                live_engine=paper_engine,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to configure release router for paper simulation", exc_info=True)

    seen_orders: set[str] = set()
    orders: list[Mapping[str, Any]] = []
    seen_errors: set[tuple[Any, ...]] = set()
    errors: list[Mapping[str, Any]] = []

    start_time = monotonic()
    stop_requested = False

    progress_interval_seconds: float | None = None
    next_progress_at: float | None = None
    if progress_callback is not None:
        interval_value = 5.0 if progress_interval is None else progress_interval
        try:
            interval_value = float(interval_value)
        except (TypeError, ValueError):
            interval_value = 5.0
        if not math.isfinite(interval_value) or interval_value <= 0.0:
            interval_value = 5.0
        progress_interval_seconds = max(0.05, interval_value)
        next_progress_at = start_time + progress_interval_seconds

    try:
        await runtime.start()

        while True:
            if stop_event is not None and stop_event.is_set():
                stop_requested = True
                break
            if stop_when_complete and min_orders > 0 and len(orders) >= min_orders:
                stop_requested = True
                break
            if not runtime.running:
                break
            if max_runtime is not None and monotonic() - start_time >= max_runtime:
                stop_requested = True
                break

            interval = max(poll_interval, 0.01)
            if stop_event is None:
                await asyncio.sleep(interval)
            else:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
                else:
                    stop_requested = True
                    break

            _capture_order_history(paper_engine, orders, seen_orders)
            _capture_error_history(paper_engine, errors, seen_errors)

            if next_progress_at is not None and progress_interval_seconds is not None:
                now = monotonic()
                if now >= next_progress_at:
                    await _emit_progress_update(
                        progress_callback,
                        runtime,
                        paper_engine,
                        orders,
                        errors,
                        start_time,
                        now,
                    )
                    next_progress_at = now + progress_interval_seconds

        # Capture one final snapshot after the loop exits in case the broker
        # updated its telemetry between the final poll and the runtime stop.
        _capture_order_history(paper_engine, orders, seen_orders)
        _capture_error_history(paper_engine, errors, seen_errors)
        if progress_callback is not None:
            await _emit_progress_update(
                progress_callback,
                runtime,
                paper_engine,
                orders,
                errors,
                start_time,
                monotonic(),
            )
    finally:
        if stop_requested or runtime.running:
            await runtime.stop()

        for cleanup in cleanups:
            try:
                outcome = cleanup()
                if asyncio.iscoroutine(outcome):
                    await outcome
            except Exception:  # pragma: no cover - diagnostics for optional cleanups
                logger.debug("Paper trading simulation cleanup failed", exc_info=True)

    runtime_seconds = monotonic() - start_time
    describe_last_error = getattr(paper_engine, "describe_last_error", None)
    if callable(describe_last_error):
        try:
            residual_error = describe_last_error()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to capture fallback paper broker error", exc_info=True)
        else:
            _append_error_metadata(residual_error, errors, seen_errors)
    portfolio_snapshot = _resolve_portfolio_snapshot(paper_engine)
    release_summary = _resolve_release_summary(runtime)

    execution_stats: Mapping[str, Any] | None = None
    try:
        stats_payload = runtime.trading_manager.get_execution_stats()
    except Exception:  # pragma: no cover - diagnostic guardrail
        logger.debug("Failed to capture execution stats from trading manager", exc_info=True)
    else:
        if isinstance(stats_payload, Mapping):
            execution_stats = _serialise_runtime_value(stats_payload)

    performance_health: Mapping[str, Any] | None = None
    try:
        health_payload = runtime.trading_manager.assess_performance_health()
    except Exception:  # pragma: no cover - diagnostic guardrail
        logger.debug("Failed to assess performance health from trading manager", exc_info=True)
    else:
        if isinstance(health_payload, Mapping):
            performance_health = _serialise_runtime_value(health_payload)

    throttle_events: Sequence[Mapping[str, Any]] = []
    try:
        history_records = runtime.trading_manager.get_trade_throttle_history()
    except Exception:  # pragma: no cover - diagnostic guardrail
        logger.debug("Failed to capture trade throttle history", exc_info=True)
    else:
        if isinstance(history_records, Sequence):
            throttle_events = [
                _serialise_runtime_value(record)
                if isinstance(record, Mapping)
                else record
                for record in history_records
            ]

    paper_metrics_snapshot = _maybe_describe_mapping(paper_engine, "describe_metrics")

    order_summary = _summarise_orders(orders)

    strategy_summary = _resolve_strategy_summary(runtime)
    strategy_summary = _enrich_strategy_summary(strategy_summary, orders, config)

    report = PaperTradingSimulationReport(
        orders=list(orders),
        errors=list(errors),
        decisions=len(runtime.decisions),
        diary_entries=_resolve_diary_count(config, runtime),
        runtime_seconds=runtime_seconds,
        order_summary=order_summary,
        paper_broker=_resolve_paper_broker_snapshot(runtime),
        paper_metrics=paper_metrics_snapshot,
        paper_failover=_resolve_paper_failover_snapshot(
            paper_engine, paper_metrics_snapshot
        ),
        portfolio_state=portfolio_snapshot,
        performance=_build_performance_summary(portfolio_snapshot),
        execution_stats=execution_stats,
        performance_health=performance_health,
        strategy_summary=strategy_summary,
        release=release_summary,
        trade_throttle=_resolve_trade_throttle_snapshot(runtime),
        trade_throttle_scopes=_resolve_trade_throttle_scopes(runtime),
        trade_throttle_events=tuple(throttle_events),
        incident_response=_resolve_incident_response(
            runtime,
            config,
            release_summary=release_summary,
            performance_health=performance_health,
            execution_stats=execution_stats,
        ),
    )
    if report_path is not None:
        path = Path(report_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to create report directory %s", path.parent, exc_info=True)
        try:
            payload = json.dumps(report.to_dict(), indent=2, default=_json_default)
            path.write_text(payload, encoding="utf-8")
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to persist paper trading simulation report", exc_info=True)

    return report


async def _emit_progress_update(
    callback: Callable[[PaperTradingSimulationProgress], Awaitable[None] | None] | None,
    runtime: Any,
    paper_engine: PaperBrokerExecutionAdapter,
    orders: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    start_monotonic: float,
    now_monotonic: float,
) -> None:
    if callback is None:
        return

    decisions = getattr(runtime, "decisions", [])
    decisions_observed = len(decisions) if isinstance(decisions, Sequence) else 0

    snapshot = PaperTradingSimulationProgress(
        timestamp=datetime.now(timezone.utc),
        runtime_seconds=max(0.0, now_monotonic - start_monotonic),
        orders_observed=len(orders),
        errors_observed=len(errors),
        decisions_observed=decisions_observed,
        paper_metrics=_maybe_describe_mapping(paper_engine, "describe_metrics"),
        failover=_maybe_describe_mapping(paper_engine, "describe_failover"),
        last_order=_maybe_describe_mapping(paper_engine, "describe_last_order"),
        last_error=_maybe_describe_mapping(paper_engine, "describe_last_error"),
    )

    try:
        outcome = callback(snapshot)
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("Paper trading simulation progress callback failed", exc_info=True)


def _capture_order_history(
    paper_engine: PaperBrokerExecutionAdapter,
    orders: list[Mapping[str, Any]],
    seen_orders: set[str],
) -> None:
    supported, records = _consume_adapter_history(paper_engine, "consume_order_history")
    if supported:
        if not records:
            return
        for metadata in records:
            _append_order_metadata(metadata, orders, seen_orders)
        return

    metadata = paper_engine.describe_last_order()
    _append_order_metadata(metadata, orders, seen_orders)


def _capture_error_history(
    paper_engine: PaperBrokerExecutionAdapter,
    errors: list[Mapping[str, Any]],
    seen_errors: set[tuple[Any, ...]],
) -> None:
    supported, records = _consume_adapter_history(paper_engine, "consume_error_history")
    if supported:
        if records:
            for payload in records:
                _append_error_metadata(payload, errors, seen_errors)
            return
        payload = paper_engine.describe_last_error()
        _append_error_metadata(payload, errors, seen_errors)
        return

    payload = paper_engine.describe_last_error()
    _append_error_metadata(payload, errors, seen_errors)


def _consume_adapter_history(
    adapter: PaperBrokerExecutionAdapter,
    method_name: str,
) -> tuple[bool, list[Mapping[str, Any]]]:
    method = getattr(adapter, method_name, None)
    if not callable(method):
        return False, []
    try:
        records = method()
    except Exception:  # pragma: no cover - defensive guard for adapter hooks
        logger.debug(
            "Failed to consume %s from paper broker adapter", method_name,
            exc_info=True,
        )
        return True, []
    if records is None:
        return True, []
    payloads: list[Mapping[str, Any]] = []
    if isinstance(records, Iterable):
        for entry in records:
            if isinstance(entry, Mapping):
                payloads.append(dict(entry))
    else:  # pragma: no cover - diagnostics for unexpected payloads
        logger.debug(
            "Paper broker adapter returned unsupported history payload: %r",
            records,
        )
    return True, payloads


def _append_order_metadata(
    metadata: Mapping[str, Any] | None,
    orders: list[Mapping[str, Any]],
    seen_orders: set[str],
) -> None:
    if not metadata:
        return
    order_id = str(metadata.get("order_id", "")).strip()
    if order_id and order_id in seen_orders:
        return
    if order_id:
        seen_orders.add(order_id)
    orders.append(dict(metadata))


def _append_error_metadata(
    payload: Mapping[str, Any] | None,
    errors: list[Mapping[str, Any]],
    seen_errors: set[tuple[Any, ...]],
) -> None:
    if not payload:
        return
    signature = (
        payload.get("stage"),
        payload.get("message"),
        payload.get("exception"),
    )
    if signature in seen_errors:
        return
    seen_errors.add(signature)
    errors.append(dict(payload))


def _resolve_paper_broker_snapshot(runtime: Any) -> Mapping[str, Any] | None:
    try:
        summary = runtime.describe_paper_broker()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Failed to resolve paper broker summary", exc_info=True)
        return None
    if not summary:
        return None
    return dict(summary)


def _resolve_paper_failover_snapshot(
    paper_engine: PaperBrokerExecutionAdapter,
    metrics_snapshot: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    snapshot = _maybe_describe_mapping(paper_engine, "describe_failover")
    if snapshot is not None:
        return dict(snapshot)

    if isinstance(metrics_snapshot, Mapping):
        failover_payload = metrics_snapshot.get("failover")
        if isinstance(failover_payload, Mapping):
            return _serialise_runtime_value(failover_payload)
    return None


def _resolve_strategy_summary(runtime: Any) -> Mapping[str, Any] | None:
    manager = getattr(runtime, "trading_manager", None)
    if manager is None:
        return None
    summary_fn = getattr(manager, "get_strategy_execution_summary", None)
    if not callable(summary_fn):
        return None
    try:
        summary = summary_fn()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Failed to summarise strategy execution", exc_info=True)
        return None
    if isinstance(summary, Mapping):
        return {
            str(strategy_id): dict(payload)
            if isinstance(payload, Mapping)
            else payload
            for strategy_id, payload in summary.items()
        }
    return None


def _enrich_strategy_summary(
    summary: Mapping[str, Any] | None,
    orders: Sequence[Mapping[str, Any]],
    config: SystemConfig,
) -> Mapping[str, Any] | None:
    if not orders:
        return summary

    enriched: dict[str, Any] = {}
    if isinstance(summary, Mapping):
        enriched.update({str(key): dict(value) if isinstance(value, Mapping) else value for key, value in summary.items()})

    extras = config.extras or {}
    fallback_strategy = str(extras.get("BOOTSTRAP_STRATEGY_ID") or extras.get("POLICY_ID") or "bootstrap-strategy")

    counts: dict[str, int] = {}
    for order in orders:
        strategy_id: str | None = None
        metadata = order.get("metadata")
        if isinstance(metadata, Mapping):
            strategy_id = metadata.get("strategy_id") or metadata.get("policy_id")
        if not strategy_id:
            strategy_id = order.get("strategy_id") or order.get("policy_id")
        strategy_key = str(strategy_id or fallback_strategy)
        counts[strategy_key] = counts.get(strategy_key, 0) + 1

    for strategy_key, executed_count in counts.items():
        stats = dict(enriched.get(strategy_key, {})) if isinstance(enriched.get(strategy_key), Mapping) else {}
        current_executed = stats.get("executed")
        if not isinstance(current_executed, int):
            current_executed = 0
        stats["executed"] = max(current_executed, executed_count)
        enriched[strategy_key] = stats

    return enriched


def _resolve_release_summary(runtime: Any) -> Mapping[str, Any] | None:
    manager = getattr(runtime, "trading_manager", None)
    if manager is None:
        return None

    summary: MutableMapping[str, Any] = {}
    stack = getattr(runtime, "trading_stack", None)
    strategy_id = getattr(stack, "strategy_id", None) if stack is not None else None

    describe_posture = getattr(manager, "describe_release_posture", None)
    if callable(describe_posture):
        try:
            posture = describe_posture(strategy_id)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to resolve release posture", exc_info=True)
        else:
            if isinstance(posture, Mapping):
                summary["posture"] = _serialise_runtime_value(posture)

    describe_execution = getattr(manager, "describe_release_execution", None)
    if callable(describe_execution):
        try:
            execution = describe_execution()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to resolve release execution summary", exc_info=True)
        else:
            if isinstance(execution, Mapping):
                summary["execution"] = _serialise_runtime_value(execution)

    get_last_route = getattr(manager, "get_last_release_route", None)
    if callable(get_last_route):
        try:
            last_route = get_last_route()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to resolve last release route", exc_info=True)
        else:
            if isinstance(last_route, Mapping) and last_route:
                summary.setdefault("last_route", _serialise_runtime_value(last_route))

    if not summary:
        return None
    return dict(summary)


def _resolve_trade_throttle_snapshot(runtime: Any) -> Mapping[str, Any] | None:
    manager = getattr(runtime, "trading_manager", None)
    if manager is None:
        return None

    snapshot_fn = getattr(manager, "get_trade_throttle_snapshot", None)
    if not callable(snapshot_fn):
        return None

    try:
        snapshot = snapshot_fn()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Failed to resolve trade throttle snapshot", exc_info=True)
        return None

    if isinstance(snapshot, Mapping):
        return _serialise_runtime_value(snapshot)
    return None


def _resolve_trade_throttle_scopes(
    runtime: Any,
) -> Sequence[Mapping[str, Any]] | None:
    manager = getattr(runtime, "trading_manager", None)
    if manager is None:
        return None

    scopes_fn = getattr(manager, "get_trade_throttle_scope_snapshots", None)
    if not callable(scopes_fn):
        return None

    try:
        snapshots = scopes_fn()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Failed to resolve trade throttle scopes", exc_info=True)
        return None

    if not isinstance(snapshots, Sequence):
        return None

    serialised: list[Mapping[str, Any]] = []
    for snapshot in snapshots:
        if isinstance(snapshot, Mapping):
            serialised.append(_serialise_runtime_value(snapshot))
    if not serialised:
        return None
    return serialised


def _summarise_orders(
    orders: Sequence[Mapping[str, Any]] | Sequence[Any],
) -> Mapping[str, Any] | None:
    total_orders = 0
    total_quantity = 0.0
    quantity_observed = False
    total_notional = 0.0
    notional_observed = False
    first_order: datetime | None = None
    last_order: datetime | None = None

    side_stats: dict[str, dict[str, Any]] = {}
    symbol_stats: dict[str, dict[str, Any]] = {}

    for order in orders:
        if not isinstance(order, Mapping):
            continue

        total_orders += 1

        raw_symbol = order.get("symbol")
        symbol = str(raw_symbol).upper() if raw_symbol is not None else "UNKNOWN"
        raw_side = order.get("side")
        side = str(raw_side).upper() if raw_side is not None else "UNKNOWN"

        quantity = _coerce_float(order.get("quantity"))
        if quantity is not None:
            quantity_observed = True
            total_quantity += quantity

        notional = _extract_notional(order)
        if notional is not None:
            notional_observed = True
            total_notional += notional

        side_entry = side_stats.setdefault(side, {"count": 0})
        side_entry["count"] += 1
        if quantity is not None:
            side_entry["quantity"] = side_entry.get("quantity", 0.0) + quantity
        if notional is not None:
            side_entry["notional"] = side_entry.get("notional", 0.0) + notional

        symbol_entry = symbol_stats.setdefault(symbol, {"count": 0, "sides": {}})
        symbol_entry["count"] += 1
        if quantity is not None:
            symbol_entry["quantity"] = symbol_entry.get("quantity", 0.0) + quantity
        if notional is not None:
            symbol_entry["notional"] = symbol_entry.get("notional", 0.0) + notional

        symbol_side_entry = symbol_entry["sides"].setdefault(side, {"count": 0})
        symbol_side_entry["count"] += 1
        if quantity is not None:
            symbol_side_entry["quantity"] = symbol_side_entry.get("quantity", 0.0) + quantity
        if notional is not None:
            symbol_side_entry["notional"] = symbol_side_entry.get("notional", 0.0) + notional

        timestamp = order.get("placed_at") or order.get("timestamp")
        parsed_ts = _parse_order_timestamp(timestamp)
        if parsed_ts is not None:
            if first_order is None or parsed_ts < first_order:
                first_order = parsed_ts
            if last_order is None or parsed_ts > last_order:
                last_order = parsed_ts

    if total_orders == 0:
        return None

    summary: dict[str, Any] = {
        "total_orders": total_orders,
        "unique_symbols": len(symbol_stats),
    }

    if quantity_observed:
        summary["total_quantity"] = total_quantity
    if notional_observed:
        summary["total_notional"] = total_notional
    if first_order is not None:
        summary["first_order_at"] = first_order.astimezone(timezone.utc).isoformat()
    if last_order is not None:
        summary["last_order_at"] = last_order.astimezone(timezone.utc).isoformat()

    sides_payload: dict[str, Any] = {}
    for side, values in side_stats.items():
        entry: dict[str, Any] = {"count": values["count"]}
        if quantity_observed and "quantity" in values:
            entry["quantity"] = values["quantity"]
        if notional_observed and "notional" in values:
            entry["notional"] = values["notional"]
        sides_payload[side] = entry
    summary["sides"] = sides_payload

    symbols_payload: dict[str, Any] = {}
    for symbol, value in symbol_stats.items():
        payload: dict[str, Any] = {"count": value["count"]}
        if quantity_observed and "quantity" in value:
            payload["quantity"] = value["quantity"]
        if notional_observed and "notional" in value:
            payload["notional"] = value["notional"]
        side_breakdown: dict[str, Any] = {}
        for side, side_value in value["sides"].items():
            side_entry: dict[str, Any] = {"count": side_value["count"]}
            if quantity_observed and "quantity" in side_value:
                side_entry["quantity"] = side_value["quantity"]
            if notional_observed and "notional" in side_value:
                side_entry["notional"] = side_value["notional"]
            side_breakdown[side] = side_entry
        payload["sides"] = side_breakdown
        symbols_payload[symbol] = payload

    summary["symbols"] = symbols_payload

    return summary


def _coerce_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


def _extract_notional(order: Mapping[str, Any]) -> float | None:
    candidates: list[Any] = []
    for key in ("notional", "order_notional", "notional_value", "trade_notional"):
        if key in order:
            candidates.append(order.get(key))

    metadata = order.get("metadata")
    if isinstance(metadata, Mapping):
        for key in (
            "notional",
            "order_notional",
            "notional_value",
            "trade_notional",
            "applied_notional",
        ):
            if key in metadata:
                candidates.append(metadata.get(key))

    for candidate in candidates:
        value = _coerce_float(candidate)
        if value is not None:
            return value
    return None


def _parse_order_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _resolve_portfolio_snapshot(
    paper_engine: PaperBrokerExecutionAdapter,
) -> Mapping[str, Any] | None:
    monitor = getattr(paper_engine, "portfolio_monitor", None)
    if monitor is None:
        return None
    try:
        snapshot = monitor.get_state()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Failed to capture portfolio snapshot from paper engine", exc_info=True)
        return None
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    return None


def _build_performance_summary(
    snapshot: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not snapshot:
        return None

    summary: MutableMapping[str, Any] = {}

    try:
        equity = float(snapshot.get("equity", 0.0))
    except (TypeError, ValueError):
        equity = 0.0
    try:
        total_pnl = float(snapshot.get("total_pnl", 0.0))
    except (TypeError, ValueError):
        total_pnl = 0.0

    initial_equity = equity - total_pnl
    roi: float | None = None
    if initial_equity:
        roi = total_pnl / initial_equity

    summary["equity"] = equity
    summary["total_pnl"] = total_pnl
    summary["initial_equity_estimate"] = initial_equity
    if roi is not None:
        summary["roi"] = roi

    realized = snapshot.get("realized_pnl")
    unrealized = snapshot.get("unrealized_pnl")
    if isinstance(realized, (int, float)):
        summary["realized_pnl"] = float(realized)
    if isinstance(unrealized, (int, float)):
        summary["unrealized_pnl"] = float(unrealized)

    drawdown = snapshot.get("current_daily_drawdown")
    if isinstance(drawdown, (int, float)):
        summary["current_daily_drawdown"] = float(drawdown)

    return summary


def _resolve_diary_count(config: SystemConfig, runtime: Any) -> int:
    store = getattr(getattr(runtime, "trading_stack", None), "_diary_store", None)
    if store is not None:
        try:
            entries: Sequence[Any] = getattr(store, "entries")()
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to read decision diary entries from runtime", exc_info=True)
        else:
            return len(entries)

    path_raw = (config.extras or {}).get("DECISION_DIARY_PATH")
    if not path_raw:
        return 0

    try:
        payload = json.loads(Path(path_raw).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0
    except json.JSONDecodeError:  # pragma: no cover - diagnostics only
        logger.debug("Decision diary payload is not valid JSON", exc_info=True)
        return 0

    entries = payload.get("entries")
    if isinstance(entries, Sequence):
        return len(entries)
    return 0


def _resolve_incident_response(
    runtime: Any,
    config: SystemConfig,
    *,
    release_summary: Mapping[str, Any] | None,
    performance_health: Mapping[str, Any] | None,
    execution_stats: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    getter = getattr(runtime, "get_last_incident_response_snapshot", None)
    if not callable(getter):
        snapshot = None
    else:
        try:
            snapshot = getter()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to capture incident response snapshot", exc_info=True)
            snapshot = None

    if snapshot is None:
        extras = config.extras or {}
        if not any(key.startswith("INCIDENT_") for key in extras):
            return None
        policy = IncidentResponsePolicy.from_mapping(extras)
        state = IncidentResponseState.from_mapping(extras)
        incident_context: MutableMapping[str, Any] = {}
        if release_summary:
            posture = release_summary.get("posture")
            if isinstance(posture, Mapping):
                status = posture.get("status") or posture.get("state")
                if status is not None:
                    incident_context["execution_status"] = status
                stage = posture.get("stage")
                if stage is not None:
                    incident_context.setdefault("execution_stage", stage)
        if execution_stats:
            orders_submitted = execution_stats.get("orders_submitted")
            if orders_submitted is not None:
                incident_context["orders_submitted"] = orders_submitted
            last_error = execution_stats.get("last_error")
            if last_error:
                incident_context["last_execution_error"] = last_error
        if performance_health:
            throughput = performance_health.get("throughput")
            if isinstance(throughput, Mapping):
                incident_context["throughput_healthy"] = bool(throughput.get("healthy"))
        incident_service = str(
            extras.get("INCIDENT_SERVICE_NAME") or "incident_response"
        )
        try:
            snapshot = evaluate_incident_response(
                policy,
                state,
                service=incident_service,
                metadata=incident_context,
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to synthesise incident response snapshot", exc_info=True)
            snapshot = None

    if snapshot is None:
        return None

    payload: MutableMapping[str, Any] = {"snapshot": snapshot.as_dict()}
    to_markdown = getattr(snapshot, "to_markdown", None)
    if callable(to_markdown):
        try:
            markdown = to_markdown()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to render incident response markdown", exc_info=True)
        else:
            if isinstance(markdown, str) and markdown.strip():
                payload["markdown"] = markdown
    return dict(payload)


def _serialise_runtime_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serialise_runtime_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialise_runtime_value(item) for item in value]
    if isinstance(value, (datetime, Decimal, Path, set, frozenset, bytes)):
        return _json_default(value)
    return value


def _maybe_describe_mapping(obj: Any, attr: str) -> Mapping[str, Any] | None:
    candidate = getattr(obj, attr, None)
    if candidate is None:
        return None
    try:
        result = candidate() if callable(candidate) else candidate
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("Progress snapshot descriptor '%s' failed", attr, exc_info=True)
        return None
    if isinstance(result, Mapping):
        return _serialise_runtime_value(result)
    return None


def _json_default(value: Any) -> Any:
    """Best-effort conversion of runtime objects into JSON-safe primitives."""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
