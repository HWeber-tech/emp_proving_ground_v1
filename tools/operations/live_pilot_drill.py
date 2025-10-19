"""CLI to exercise the live-pilot readiness drill (tiny capital, kill-switch, rollback, reconciliation)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence

from src.governance.safety_manager import SafetyManager
from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
    EmpEnvironment,
    EmpTier,
    RunMode,
    SystemConfig,
)
from src.runtime import predator_app
from src.trading.execution.trade_throttle import (
    TradeThrottle,
    TradeThrottleConfig,
)
from src.trading.order_management import PositionTracker, report_to_dict

logger = logging.getLogger("tools.operations.live_pilot_drill")

UTC = timezone.utc


def _resolve_run_timestamp(timestamp: str | None) -> str:
    if timestamp:
        return timestamp
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_tiny_capital_drill() -> Dict[str, Any]:
    """Enable tiny-capital mode and surface the resolved ROI cost model."""

    baseline = SystemConfig(
        run_mode=RunMode.paper,
        environment=EmpEnvironment.demo,
        tier=EmpTier.tier_0,
        confirm_live=False,
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
    )

    extras = dict(baseline.extras)
    extras.update(
        {
            "ROI_INITIAL_CAPITAL": "5000",
            "ROI_TARGET_ANNUAL_ROI": "0.10",
            "ROI_BROKER_FEE_BPS": "0.5",
        }
    )

    tiny_config = baseline.with_updated(
        run_mode=RunMode.live,
        confirm_live=True,
        environment=EmpEnvironment.production,
        connection_protocol=ConnectionProtocol.fix,
        data_backbone_mode=DataBackboneMode.institutional,
        extras=extras,
    )

    initial_equity = 125_000.0
    roi_cost_model = predator_app._resolve_roi_cost_model(tiny_config, initial_equity=initial_equity)

    if roi_cost_model.initial_capital >= 50_000.0:
        raise RuntimeError(
            "Tiny capital override failed: expected initial capital to be below 50k, "
            f"got {roi_cost_model.initial_capital:.2f}"
        )

    return {
        "run_mode": tiny_config.run_mode.value,
        "environment": tiny_config.environment.value,
        "confirm_live": tiny_config.confirm_live,
        "connection_protocol": tiny_config.connection_protocol.value,
        "data_backbone_mode": tiny_config.data_backbone_mode.value,
        "initial_equity": initial_equity,
        "resolved_initial_capital": roi_cost_model.initial_capital,
        "target_annual_roi": roi_cost_model.target_annual_roi,
        "broker_fee_bps": roi_cost_model.broker_fee_bps,
        "extras": dict(tiny_config.extras),
    }


def _run_kill_switch_drill(run_dir: Path) -> Dict[str, Any]:
    """Exercise the kill-switch enforcement path."""

    kill_switch_path = (run_dir / "kill_switch.flag").resolve()
    logger.debug("Using kill-switch sentinel at %s", kill_switch_path)

    manager = SafetyManager("paper", confirm_live=False, kill_switch_path=kill_switch_path)
    manager.enforce()

    kill_switch_path.write_text("halt", encoding="utf-8")

    engaged = False
    error_message: str | None = None
    try:
        armed_manager = SafetyManager("paper", confirm_live=False, kill_switch_path=kill_switch_path)
        try:
            armed_manager.enforce()
        except RuntimeError as exc:
            engaged = True
            error_message = str(exc)
    finally:
        try:
            kill_switch_path.unlink()
        except FileNotFoundError:
            pass

    if not engaged:
        raise RuntimeError("Kill-switch drill failed: enforcement did not abort when sentinel file was present.")

    return {
        "kill_switch_path": str(kill_switch_path),
        "message": error_message,
    }


def _run_trade_rollback_drill() -> Dict[str, Any]:
    """Validate trade throttle rollback restores capacity for the same scope."""

    config = TradeThrottleConfig(max_trades=2, window_seconds=60.0, scope_fields=("symbol",))
    throttle = TradeThrottle(config)
    symbol_metadata: Mapping[str, Any] = {"symbol": "EURUSD", "strategy_id": "live-pilot-drill"}

    start = datetime.now(tz=UTC).replace(microsecond=0)
    first = throttle.evaluate(now=start, metadata=symbol_metadata)
    if not first.allowed:
        raise RuntimeError("First trade should be allowed during rollback drill")

    second_time = start + timedelta(seconds=1)
    second = throttle.evaluate(now=second_time, metadata=symbol_metadata)
    if not second.allowed:
        raise RuntimeError("Second trade should be allowed before rollback")

    third_time = start + timedelta(seconds=2)
    blocked = throttle.evaluate(now=third_time, metadata=symbol_metadata)
    if blocked.allowed:
        raise RuntimeError("Throttle should block third trade before rollback")

    snapshot = throttle.rollback(second)
    if snapshot is None:
        raise RuntimeError("Rollback returned no snapshot for allowed trade")

    retry_time = third_time + timedelta(seconds=1)
    retried = throttle.evaluate(now=retry_time, metadata=symbol_metadata)
    if not retried.allowed:
        raise RuntimeError("Throttle should allow trade after rollback")

    return {
        "initial_allowed": 2,
        "blocked_reason": blocked.reason,
        "rollback_scope": snapshot.get("scope_key"),
        "retry_allowed": True,
    }


def _run_reconciliation_drill() -> Dict[str, Any]:
    """Replay sample fills and reconcile against broker balances."""

    tracker = PositionTracker(pnl_mode="fifo", default_account="LIVE-PILOT")

    tracker.record_fill("EURUSD", 100_000.0, 1.1000, account="LIVE-PILOT")
    tracker.record_fill("EURUSD", -100_000.0, 1.1002, account="LIVE-PILOT")
    tracker.update_mark_price("EURUSD", 1.1001)

    tracker.record_fill("USDJPY", -50_000.0, 147.25, account="LIVE-PILOT")
    tracker.record_fill("USDJPY", 50_000.0, 147.20, account="LIVE-PILOT")
    tracker.update_mark_price("USDJPY", 147.22)

    broker_positions = {
        "EURUSD": 0.0,
        "USDJPY": 0.0,
    }

    report = tracker.generate_reconciliation_report(
        broker_positions,
        account="LIVE-PILOT",
        tolerance=1e-6,
    )

    if report.differences:
        raise RuntimeError("Reconciliation drill detected differences")

    payload = report_to_dict(report)
    payload["broker_positions"] = broker_positions
    payload["tracked_symbols"] = sorted(broker_positions.keys())

    return payload


def _execute_drill(
    name: str,
    handler: Callable[[], Dict[str, Any]],
    artifact_path: Path,
) -> Dict[str, Any]:
    try:
        payload = handler()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Live pilot drill '%s' failed", name)
        result: Dict[str, Any] = {
            "status": "failed",
            "error": str(exc),
        }
    else:
        result = {
            "status": "passed",
            **payload,
        }

    _write_json(artifact_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/live_pilot"),
        help="Directory where live pilot drill artifacts are stored (default: artifacts/live_pilot)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional ISO-like timestamp (YYYYMMDDTHHMMSSZ) used for the run directory name.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level)

    timestamp = _resolve_run_timestamp(args.timestamp)
    run_root = args.run_root.resolve()
    run_dir = (run_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting live pilot drill run_id=%s", timestamp)

    results: MutableMapping[str, Dict[str, Any]] = {}

    results["tiny_capital"] = _execute_drill(
        "tiny_capital",
        _run_tiny_capital_drill,
        run_dir / "tiny_capital.json",
    )

    results["kill_switch"] = _execute_drill(
        "kill_switch",
        lambda: _run_kill_switch_drill(run_dir),
        run_dir / "kill_switch.json",
    )

    results["trade_rollback"] = _execute_drill(
        "trade_rollback",
        _run_trade_rollback_drill,
        run_dir / "trade_rollback.json",
    )

    results["reconciliation"] = _execute_drill(
        "reconciliation",
        _run_reconciliation_drill,
        run_dir / "reconciliation.json",
    )

    summary = {
        "run_id": f"live-pilot-drill-{timestamp}",
        "timestamp": timestamp,
        "results": results,
    }
    _write_json(run_dir / "live_pilot_drill_summary.json", summary)

    overall_status = all(entry.get("status") == "passed" for entry in results.values())
    if overall_status:
        logger.info("Live pilot drill completed successfully")
        return 0

    logger.error("Live pilot drill encountered failures")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI gateway
    raise SystemExit(main())
