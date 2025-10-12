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
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.core.event_bus import EventBus
from src.governance.system_config import SystemConfig
from src.runtime.predator_app import _build_bootstrap_runtime
from src.trading.execution.paper_broker_adapter import PaperBrokerExecutionAdapter

logger = logging.getLogger(__name__)

__all__ = ["PaperTradingSimulationReport", "run_paper_trading_simulation"]


@dataclass(slots=True)
class PaperTradingSimulationReport:
    """Summary of an executed paper trading simulation."""

    orders: list[Mapping[str, Any]] = field(default_factory=list)
    errors: list[Mapping[str, Any]] = field(default_factory=list)
    decisions: int = 0
    diary_entries: int = 0
    runtime_seconds: float = 0.0
    paper_broker: Mapping[str, Any] | None = None
    paper_metrics: Mapping[str, Any] | None = None
    portfolio_state: Mapping[str, Any] | None = None
    performance: Mapping[str, Any] | None = None
    execution_stats: Mapping[str, Any] | None = None
    performance_health: Mapping[str, Any] | None = None
    strategy_summary: Mapping[str, Any] | None = None
    release: Mapping[str, Any] | None = None

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
        return payload


async def run_paper_trading_simulation(
    config: SystemConfig,
    *,
    min_orders: int = 1,
    max_runtime: float | None = 60.0,
    poll_interval: float = 0.5,
    stop_when_complete: bool = True,
    report_path: str | Path | None = None,
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

    seen_orders: set[str] = set()
    orders: list[Mapping[str, Any]] = []
    seen_errors: set[tuple[Any, ...]] = set()
    errors: list[Mapping[str, Any]] = []

    start_time = monotonic()
    stop_requested = False

    try:
        await runtime.start()

        while True:
            if stop_when_complete and min_orders > 0 and len(orders) >= min_orders:
                stop_requested = True
                break
            if not runtime.running:
                break
            if max_runtime is not None and monotonic() - start_time >= max_runtime:
                stop_requested = True
                break

            await asyncio.sleep(max(poll_interval, 0.01))

            _capture_order_history(paper_engine, orders, seen_orders)
            _capture_error_history(paper_engine, errors, seen_errors)

        # Capture one final snapshot after the loop exits in case the broker
        # updated its telemetry between the final poll and the runtime stop.
        _capture_order_history(paper_engine, orders, seen_orders)
        _capture_error_history(paper_engine, errors, seen_errors)
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

    report = PaperTradingSimulationReport(
        orders=list(orders),
        errors=list(errors),
        decisions=len(runtime.decisions),
        diary_entries=_resolve_diary_count(config, runtime),
        runtime_seconds=runtime_seconds,
        paper_broker=_resolve_paper_broker_snapshot(runtime),
        paper_metrics=paper_engine.describe_metrics(),
        portfolio_state=portfolio_snapshot,
        performance=_build_performance_summary(portfolio_snapshot),
        execution_stats=execution_stats,
        performance_health=performance_health,
        strategy_summary=_resolve_strategy_summary(runtime),
        release=release_summary,
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


def _serialise_runtime_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serialise_runtime_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialise_runtime_value(item) for item in value]
    if isinstance(value, (datetime, Decimal, Path, set, frozenset, bytes)):
        return _json_default(value)
    return value


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
