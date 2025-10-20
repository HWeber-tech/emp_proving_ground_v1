#!/usr/bin/env python3
"""Execute a deterministic bootstrap simulation and persist a run summary."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_foundation.duckdb_security import (
    resolve_encrypted_duckdb_path,
    verify_encrypted_duckdb_path,
)
from src.governance.system_config import SystemConfig
from src.runtime.predator_app import build_professional_predator_app
from src.runtime.runtime_builder import build_professional_runtime_application
from src.runtime.runtime_runner import run_runtime_application


logger = logging.getLogger("tools.run_simulation")


_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _json_ready(value: Any) -> Any:
    """Convert runtime payloads into JSON-serialisable objects."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        try:
            return _json_ready(value.as_dict())
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bootstrap runtime in deterministic simulation mode and write "
            "a JSON summary for roadmap acceptance checks."
        )
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("artifacts/sim/summary.json"),
        help="File path where the summary JSON should be written.",
    )
    parser.add_argument(
        "--diary-path",
        type=Path,
        default=None,
        help=(
            "Decision diary path for the run. Defaults to artifacts/diaries/sim.jsonl "
            "when not provided."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Maximum runtime duration in seconds before a graceful shutdown is requested.",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=0.5,
        help="Tick interval (seconds) applied to the bootstrap runtime (BOOTSTRAP_TICK_INTERVAL).",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=120,
        help="Optional cap on bootstrap ticks (BOOTSTRAP_MAX_TICKS). Use 0 to disable the cap.",
    )
    parser.add_argument(
        "--symbols",
        default="EURUSD",
        help="Comma-separated symbols fed into the bootstrap runtime (BOOTSTRAP_SYMBOLS).",
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=Path("data/tier0.duckdb"),
        help="DuckDB file used for bootstrap ingest fallbacks (passed to the runtime builder).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=sorted(_LOG_LEVELS),
        help="Logging level for the simulation wrapper.",
    )
    parser.add_argument(
        "--enable-trading",
        action="store_true",
        help="Run with trading workloads enabled (disabled by default for determinism).",
    )
    parser.add_argument(
        "--include-ingest",
        dest="skip_ingest",
        action="store_false",
        help="Include ingest workloads. Defaults to skipping ingest for faster runs.",
    )
    parser.set_defaults(skip_ingest=True)
    return parser.parse_args()


def _apply_environment(args: argparse.Namespace) -> Mapping[str, str]:
    """Set deterministic environment defaults and return the values applied."""

    applied: dict[str, str] = {}

    def _set(name: str, value: str) -> None:
        os.environ[name] = value
        applied[name] = value

    _set("RUN_MODE", os.environ.get("RUN_MODE", "mock"))
    _set("EMP_ENVIRONMENT", os.environ.get("EMP_ENVIRONMENT", "demo"))
    _set("EMP_TIER", os.environ.get("EMP_TIER", "tier_0"))
    _set("DATA_BACKBONE_MODE", os.environ.get("DATA_BACKBONE_MODE", "bootstrap"))

    diary_path = args.diary_path or Path("artifacts/diaries/sim.jsonl")
    diary_path = diary_path.expanduser().resolve()
    diary_path.parent.mkdir(parents=True, exist_ok=True)
    _set("DECISION_DIARY_PATH", str(diary_path))

    if args.symbols:
        _set("BOOTSTRAP_SYMBOLS", args.symbols)
    _set("BOOTSTRAP_TICK_INTERVAL", str(args.tick_interval))
    if args.max_ticks and args.max_ticks > 0:
        _set("BOOTSTRAP_MAX_TICKS", str(args.max_ticks))
    else:
        os.environ.pop("BOOTSTRAP_MAX_TICKS", None)

    return applied


async def _run_simulation(args: argparse.Namespace) -> Mapping[str, Any]:
    config = SystemConfig.from_env()
    app = await build_professional_predator_app(config=config)
    duckdb_destination = resolve_encrypted_duckdb_path(args.duckdb_path)
    verify_encrypted_duckdb_path(duckdb_destination)
    setattr(args, "sanitized_duckdb_path", duckdb_destination)
    runtime_app = build_professional_runtime_application(
        app,
        skip_ingest=args.skip_ingest,
        symbols_csv=args.symbols,
        duckdb_path=str(duckdb_destination),
    )

    summary: Mapping[str, Any] | None = None
    try:
        async with app:
            if not args.enable_trading:
                runtime_app.trading = None
            await run_runtime_application(
                runtime_app,
                timeout=args.timeout,
                logger=logger,
                namespace="tools.run_simulation",
            )
            summary = app.summary()
    finally:
        await app.shutdown()

    final_summary = app.summary()
    if final_summary:
        return final_summary
    if summary is not None:
        return summary
    return {}


def _write_summary(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    applied_env: Mapping[str, str],
    args: argparse.Namespace,
) -> None:
    summary_path = summary_path.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    diary_path = Path(os.environ.get("DECISION_DIARY_PATH", "")).expanduser()
    diary_info: dict[str, Any] = {
        "path": str(diary_path) if diary_path else None,
        "exists": diary_path.exists(),
    }
    if not diary_info["exists"] and diary_info["path"] is not None:
        placeholder = {
            "event": "simulation_placeholder",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": "No decision diary entries were emitted during the simulation run.",
        }
        diary_path.write_text(json.dumps(placeholder) + "\n", encoding="utf-8")
        diary_info["exists"] = True

    if diary_info["exists"]:
        diary_info["size_bytes"] = diary_path.stat().st_size

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "timeout_seconds": args.timeout,
            "tick_interval": args.tick_interval,
            "max_ticks": args.max_ticks,
            "symbols": args.symbols,
            "skip_ingest": args.skip_ingest,
            "trading_enabled": args.enable_trading,
            "duckdb_path": str(getattr(args, "sanitized_duckdb_path", args.duckdb_path)),
        },
        "environment_applied": dict(applied_env),
        "diary": diary_info,
        "summary": summary,
    }

    summary_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO))

    applied = _apply_environment(args)

    summary = asyncio.run(_run_simulation(args))

    _write_summary(summary, summary_path=args.summary_path, applied_env=applied, args=args)

    logger.info("Simulation summary written to %s", args.summary_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(1)
