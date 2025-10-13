"""Command-line utilities for the Professional Predator runtime builder.

This CLI translates the roadmap's runtime builder follow-ups into an
operator-friendly entrypoint.  It exposes subcommands that print runtime
summaries, run ingestion/trading workloads with optional timeouts, execute a
single ingest cycle, and rehearse restart flows to verify structured shutdown
hooks.  The CLI intentionally leans on :mod:`runtime_builder` so it stays in
lockstep with the production orchestration code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from contextlib import suppress
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from src.governance.system_config import SystemConfig
from src.runtime.paper_run_guardian import (
    PaperRunConfig,
    PaperRunStatus,
    run_guarded_paper_session,
)
from src.runtime.predator_app import ProfessionalPredatorApp, build_professional_predator_app
from src.runtime.runtime_builder import RuntimeApplication, build_professional_runtime_application
from src.runtime.runtime_runner import run_runtime_application


logger = logging.getLogger(__name__)


AsyncRuntimeHandler = Callable[[argparse.Namespace], Awaitable[int]]


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict") and callable(value.as_dict):
        try:
            return _json_ready(value.as_dict())
        except Exception:  # pragma: no cover - defensive guard
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return str(value)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:  # pragma: no cover - argparse surfaces error message
        raise argparse.ArgumentTypeError(f"Invalid float value: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Timeout must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - argparse surfaces error message
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:  # pragma: no cover - argparse surfaces error message
        raise argparse.ArgumentTypeError(f"Invalid float value: {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be greater than or equal to zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emp-runtime", description="EMP Professional Predator runtime CLI"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices={"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
        help="Logging level for the CLI session",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_runtime_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--skip-ingest",
            action="store_true",
            help="Disable ingest workloads for this invocation",
        )
        subparser.add_argument(
            "--symbols",
            default="EURUSD,GBPUSD",
            help="Comma-separated symbols for bootstrap ingest fallbacks",
        )
        subparser.add_argument(
            "--duckdb-path",
            default="data/tier0.duckdb",
            help="Destination path for Tier-0 DuckDB ingest",
        )

    summary_parser = subparsers.add_parser(
        "summary", help="Print runtime configuration and workload summary"
    )
    _add_runtime_options(summary_parser)
    summary_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary payload as JSON",
    )
    summary_parser.add_argument(
        "--no-trading",
        action="store_true",
        help="Exclude the trading workload from the summary payload",
    )
    summary_parser.set_defaults(handler=_handle_summary)

    run_parser = subparsers.add_parser("run", help="Run the runtime application")
    _add_runtime_options(run_parser)
    run_parser.add_argument(
        "--no-trading",
        action="store_true",
        help="Disable the trading workload for this run",
    )
    run_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=None,
        help="Optional timeout (seconds) after which the runtime is shut down",
    )
    run_parser.set_defaults(handler=_handle_run)

    ingest_parser = subparsers.add_parser(
        "ingest-once", help="Execute a single ingest cycle and exit"
    )
    _add_runtime_options(ingest_parser)
    ingest_parser.set_defaults(handler=_handle_ingest_once)

    restart_parser = subparsers.add_parser(
        "restart", help="Run the runtime for multiple restart cycles"
    )
    _add_runtime_options(restart_parser)
    restart_parser.add_argument(
        "--cycles",
        type=_positive_int,
        default=2,
        help="Number of run/shutdown cycles to execute",
    )
    restart_parser.add_argument(
        "--no-trading",
        action="store_true",
        help="Disable the trading workload for each cycle",
    )
    restart_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=None,
        help="Optional timeout applied to each cycle",
    )
    restart_parser.set_defaults(handler=_handle_restart)

    paper_parser = subparsers.add_parser(
        "paper-run",
        help="Run the paper runtime with guardian telemetry",
    )
    paper_parser.add_argument(
        "--duration-hours",
        type=_non_negative_float,
        default=None,
        help="Optional duration (hours) before the guardian requests shutdown",
    )
    paper_parser.add_argument(
        "--progress-interval",
        type=_positive_float,
        default=60.0,
        help="Seconds between guardian progress snapshots",
    )
    paper_parser.add_argument(
        "--latency-p99-max",
        type=_positive_float,
        default=None,
        help="Latency p99 threshold in seconds; exceedance marks the run as degraded",
    )
    paper_parser.add_argument(
        "--memory-growth-max",
        type=_non_negative_float,
        default=None,
        help="Maximum allowed memory growth in MB before the run is marked degraded",
    )
    paper_parser.add_argument(
        "--min-orders",
        type=_positive_int,
        default=0,
        help="Minimum order count before the guardian allows shutdown",
    )
    paper_parser.add_argument(
        "--allow-invariant-errors",
        action="store_true",
        help="Do not stop the run when risk invariant breaches are observed",
    )
    paper_parser.add_argument(
        "--report-path",
        default=None,
        help="Path where the guardian summary JSON will be written",
    )
    paper_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the guardian summary as JSON to stdout",
    )
    paper_parser.set_defaults(handler=_handle_paper_run)

    return parser


async def _build_runtime(
    args: argparse.Namespace,
) -> tuple[ProfessionalPredatorApp, RuntimeApplication, SystemConfig]:
    config = SystemConfig.from_env()
    app = await build_professional_predator_app(config=config)
    runtime_app = build_professional_runtime_application(
        app,
        skip_ingest=args.skip_ingest,
        symbols_csv=args.symbols,
        duckdb_path=args.duckdb_path,
    )
    return app, runtime_app, config


async def _with_runtime(
    args: argparse.Namespace,
    callback: Callable[[ProfessionalPredatorApp, RuntimeApplication, SystemConfig], Awaitable[int]],
) -> int:
    app, runtime_app, config = await _build_runtime(args)
    try:
        async with app:
            return await callback(app, runtime_app, config)
    finally:
        await app.shutdown()


async def _handle_summary(
    args: argparse.Namespace,
) -> int:
    async def _summarise(
        app: ProfessionalPredatorApp,
        runtime_app: RuntimeApplication,
        config: SystemConfig,
    ) -> int:
        if args.no_trading:
            runtime_app.trading = None

        runtime_summary = runtime_app.summary()
        app_summary = app.summary()
        payload: Mapping[str, Any] = {
            "config": {
                "tier": config.tier.value,
                "backbone_mode": config.data_backbone_mode.value,
            },
            "runtime": runtime_summary,
            "application": app_summary,
        }

        if args.json:
            print(json.dumps(_json_ready(payload), indent=2, sort_keys=True))
        else:
            print("Runtime summary:")
            print(json.dumps(_json_ready(runtime_summary), indent=2, sort_keys=True))
            print("Application summary:")
            print(json.dumps(_json_ready(app_summary), indent=2, sort_keys=True))

        return 0

    return await _with_runtime(args, _summarise)



async def _handle_run(args: argparse.Namespace) -> int:
    async def _run(
        app: ProfessionalPredatorApp,
        runtime_app: RuntimeApplication,
        _: SystemConfig,
    ) -> int:
        if args.no_trading:
            runtime_app.trading = None

        await run_runtime_application(
            runtime_app,
            timeout=args.timeout,
            logger=logger,
            namespace="runtime.cli",
        )
        return 0

    return await _with_runtime(args, _run)


async def _handle_ingest_once(args: argparse.Namespace) -> int:
    async def _ingest(
        app: ProfessionalPredatorApp,
        runtime_app: RuntimeApplication,
        _: SystemConfig,
    ) -> int:
        runtime_app.trading = None
        if runtime_app.ingestion is None:
            logger.info("No ingest workload configured; nothing to execute")
            return 0

        for callback in list(runtime_app.startup_callbacks):
            result = callback()
            if asyncio.iscoroutine(result):
                await result

        try:
            await runtime_app.ingestion.factory()
        finally:
            await runtime_app.shutdown()

        logger.info("Ingest cycle complete")
        return 0

    return await _with_runtime(args, _ingest)


async def _handle_restart(args: argparse.Namespace) -> int:
    cycles = args.cycles
    for index in range(cycles):
        logger.info("🔁 Restart cycle %s/%s", index + 1, cycles)
        print(f"🔁 Running cycle {index + 1}/{cycles}")

        exit_code = await _handle_run(args)
        if exit_code != 0:
            return exit_code

        print(f"✅ Completed cycle {index + 1}/{cycles}")

    return 0


async def _handle_paper_run(args: argparse.Namespace) -> int:
    config = SystemConfig.from_env()

    duration_seconds: float | None
    if args.duration_hours in (None, 0):
        duration_seconds = None
    else:
        duration_seconds = float(args.duration_hours) * 3600.0

    report_path = Path(args.report_path).expanduser() if args.report_path else None

    run_config = PaperRunConfig(
        duration_seconds=duration_seconds,
        progress_interval=args.progress_interval,
        latency_p99_threshold=args.latency_p99_max,
        memory_growth_threshold_mb=args.memory_growth_max,
        allow_invariant_errors=args.allow_invariant_errors,
        report_path=report_path,
        min_orders=args.min_orders,
    )

    summary = await run_guarded_paper_session(config, run_config, logger=logger)

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Status: {summary.status.value}")
        runtime_seconds = summary.runtime_seconds or 0.0
        print(f"Runtime: {runtime_seconds:.2f}s")
        metrics = summary.metrics
        latency_p99 = metrics.get("latency_p99_s")
        if latency_p99 is not None:
            print(f"Latency p99: {latency_p99:.4f}s")
        memory_growth = metrics.get("memory_growth_mb")
        if memory_growth is not None:
            print(f"Memory growth: {memory_growth:.2f}MB")
        if summary.alerts:
            print("Alerts:")
            for alert in summary.alerts:
                print(f"  - {alert}")
        if summary.stop_reasons:
            print("Stop reasons:")
            for reason in summary.stop_reasons:
                print(f"  - {reason}")
        if summary.invariant_breaches:
            print("Invariant breaches observed: %s" % len(summary.invariant_breaches))

        if report_path is not None:
            print(f"Summary persisted to {report_path}")

    if summary.status is PaperRunStatus.FAILED:
        return 2
    if summary.status is PaperRunStatus.DEGRADED:
        return 1
    return 0


async def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    handler: AsyncRuntimeHandler = args.handler
    return await handler(args)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(run_cli(argv))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        return 1


if __name__ == "__main__":  # pragma: no cover - manual execution entrypoint
    sys.exit(main())
