"""CLI to validate the incident playbook drills (kill-switch, replay, rollback)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence

from src.governance.safety_manager import SafetyManager
from src.trading.execution.trade_throttle import (
    TradeThrottle,
    TradeThrottleConfig,
)
from tools.operations import nightly_replay_job

logger = logging.getLogger("tools.operations.incident_playbook_validation")

UTC = timezone.utc


def _resolve_run_timestamp(timestamp: str | None) -> str:
    if timestamp:
        return timestamp
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def _normalise_replay_timestamp(timestamp: str) -> str:
    """Convert run identifiers to ISO-8601 strings understood by replay CLI."""

    try:
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return timestamp
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_kill_switch_drill(run_dir: Path) -> Dict[str, Any]:
    """Exercise the kill-switch enforcement path."""

    kill_switch_path = (run_dir / "kill_switch.flag").resolve()
    logger.debug("Using kill-switch sentinel at %s", kill_switch_path)

    manager = SafetyManager("paper", confirm_live=False, kill_switch_path=kill_switch_path)
    manager.enforce()  # should not raise when the sentinel is absent

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


def _run_replay_drill(run_dir: Path, timestamp: str, log_level: str) -> Dict[str, Any]:
    """Execute the nightly replay job and capture key artifacts."""

    replay_root = (run_dir / "nightly_replay").resolve()
    replay_timestamp = _normalise_replay_timestamp(timestamp)

    replay_args = [
        "--run-root",
        str(replay_root),
        "--timestamp",
        replay_timestamp,
        "--log-level",
        log_level,
    ]

    logger.debug("Invoking nightly replay job with args: %s", replay_args)
    exit_code = nightly_replay_job.main(replay_args)
    if exit_code != 0:
        raise RuntimeError(f"Nightly replay job exited with status {exit_code}")

    job_dir = replay_root / timestamp
    evaluation_path = job_dir / "replay_evaluation.json"
    drift_path = job_dir / "sensor_drift_summary.json"
    diary_path = job_dir / "decision_diary.json"
    ledger_path = job_dir / "policy_ledger.json"

    if not evaluation_path.exists():
        raise FileNotFoundError(f"Replay evaluation summary missing at {evaluation_path}")

    summary_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    drift_payload = json.loads(drift_path.read_text(encoding="utf-8")) if drift_path.exists() else {}
    diary_payload = json.loads(diary_path.read_text(encoding="utf-8")) if diary_path.exists() else {}
    ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}

    return {
        "run_id": summary_payload.get("run_id"),
        "artifact_dir": str(job_dir),
        "tactic_count": len(summary_payload.get("results", [])),
        "drift_observations": drift_payload.get("total_observations"),
        "diary_entry_count": len(diary_payload.get("entries", [])),
        "ledger_record_count": len(ledger_payload.get("records", [])),
    }


def _run_rollback_drill(run_dir: Path) -> Dict[str, Any]:
    """Validate trade throttle rollback restores capacity for the same scope."""

    config = TradeThrottleConfig(max_trades=2, window_seconds=60.0, scope_fields=("symbol",))
    throttle = TradeThrottle(config)
    symbol_metadata: Mapping[str, Any] = {"symbol": "EURUSD", "strategy_id": "incident-drill"}

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


def _execute_drill(
    name: str,
    handler: Callable[[], Dict[str, Any]],
    artifact_path: Path,
) -> Dict[str, Any]:
    try:
        payload = handler()
    except Exception as exc:  # pragma: no cover - exercised in failure scenarios
        logger.exception("Incident drill '%s' failed", name)
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
        default=Path("artifacts/incident_playbook"),
        help="Directory where incident validation artifacts are stored (default: artifacts/incident_playbook)",
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

    logger.info("Starting incident playbook validation run_id=%s", timestamp)

    results: MutableMapping[str, Dict[str, Any]] = {}

    results["kill_switch"] = _execute_drill(
        "kill_switch",
        lambda: _run_kill_switch_drill(run_dir),
        run_dir / "kill_switch.json",
    )

    results["nightly_replay"] = _execute_drill(
        "nightly_replay",
        lambda: _run_replay_drill(run_dir, timestamp, args.log_level.upper()),
        run_dir / "nightly_replay.json",
    )

    results["trade_rollback"] = _execute_drill(
        "trade_rollback",
        lambda: _run_rollback_drill(run_dir),
        run_dir / "trade_rollback.json",
    )

    summary = {
        "run_id": f"incident-playbook-{timestamp}",
        "timestamp": timestamp,
        "results": results,
    }
    _write_json(run_dir / "incident_playbook_summary.json", summary)

    overall_status = all(entry.get("status") == "passed" for entry in results.values())
    if overall_status:
        logger.info("Incident playbook validation completed successfully")
        return 0

    logger.error("Incident playbook validation encountered failures")
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI gateway
    raise SystemExit(main())
