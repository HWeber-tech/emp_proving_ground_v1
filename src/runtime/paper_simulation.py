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
from typing import Any, Mapping, MutableMapping, Sequence

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
    portfolio_state: Mapping[str, Any] | None = None
    performance: Mapping[str, Any] | None = None

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
        if self.portfolio_state is not None:
            payload["portfolio_state"] = dict(self.portfolio_state)
        if self.performance is not None:
            payload["performance"] = dict(self.performance)
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

            _capture_last_order(paper_engine, orders, seen_orders)
            _capture_last_error(paper_engine, errors, seen_errors)

        # Capture one final snapshot after the loop exits in case the broker
        # updated its telemetry between the final poll and the runtime stop.
        _capture_last_order(paper_engine, orders, seen_orders)
        _capture_last_error(paper_engine, errors, seen_errors)
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

    report = PaperTradingSimulationReport(
        orders=list(orders),
        errors=list(errors),
        decisions=len(runtime.decisions),
        diary_entries=_resolve_diary_count(config, runtime),
        runtime_seconds=runtime_seconds,
        paper_broker=_resolve_paper_broker_snapshot(runtime),
        portfolio_state=portfolio_snapshot,
        performance=_build_performance_summary(portfolio_snapshot),
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


def _capture_last_order(
    paper_engine: PaperBrokerExecutionAdapter,
    orders: list[Mapping[str, Any]],
    seen_orders: set[str],
) -> None:
    metadata = paper_engine.describe_last_order()
    if not metadata:
        return
    order_id = str(metadata.get("order_id", "")).strip()
    if order_id and order_id in seen_orders:
        return
    if order_id:
        seen_orders.add(order_id)
    orders.append(dict(metadata))


def _capture_last_error(
    paper_engine: PaperBrokerExecutionAdapter,
    errors: list[Mapping[str, Any]],
    seen_errors: set[tuple[Any, ...]],
) -> None:
    payload = paper_engine.describe_last_error()
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
