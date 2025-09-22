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
import signal
import sys
from contextlib import suppress
from typing import Any, Awaitable, Callable, Mapping, Sequence

from src.governance.system_config import SystemConfig
from src.runtime.predator_app import ProfessionalPredatorApp, build_professional_predator_app
from src.runtime.runtime_builder import RuntimeApplication, build_professional_runtime_application


logger = logging.getLogger(__name__)


AsyncRuntimeHandler = Callable[[argparse.Namespace], Awaitable[int]]


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
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Runtime summary:")
            print(json.dumps(runtime_summary, indent=2, sort_keys=True))
            print("Application summary:")
            print(json.dumps(app_summary, indent=2, sort_keys=True))

        return 0

    return await _with_runtime(args, _summarise)


async def _run_runtime_with_signals(runtime_app: RuntimeApplication, timeout: float | None) -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _trigger_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _trigger_stop)
        except (NotImplementedError, ValueError):  # pragma: no cover - Windows / nested loop
            pass

    run_task = asyncio.create_task(runtime_app.run(), name="runtime-app-run")
    waiters: set[asyncio.Task[object]] = {run_task}

    stop_task = asyncio.create_task(stop_event.wait(), name="runtime-stop-event")
    waiters.add(stop_task)

    timeout_task: asyncio.Task[object] | None = None
    if timeout is not None:
        timeout_task = asyncio.create_task(asyncio.sleep(timeout), name="runtime-timeout")
        waiters.add(timeout_task)

    done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)

    if run_task in done:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return

    if stop_task in done and stop_event.is_set():
        logger.info("Shutdown signal received; cancelling runtime workloads")
    elif timeout_task is not None and timeout_task in done:
        logger.info("Runtime timeout reached after %ss; cancelling workloads", timeout)

    run_task.cancel()
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    with suppress(asyncio.CancelledError):
        await run_task


async def _handle_run(args: argparse.Namespace) -> int:
    async def _run(
        app: ProfessionalPredatorApp,
        runtime_app: RuntimeApplication,
        _: SystemConfig,
    ) -> int:
        if args.no_trading:
            runtime_app.trading = None

        await _run_runtime_with_signals(runtime_app, args.timeout)
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
