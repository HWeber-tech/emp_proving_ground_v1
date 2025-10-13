"""Command-line helper for executing a paper trading simulation."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence

from src.governance.system_config import (
    ConnectionProtocol,
    RunMode,
    SystemConfig,
    SystemConfigLoadError,
)
from src.runtime.paper_simulation import (
    PaperTradingSimulationProgress,
    _json_default as _simulation_json_default,
    run_paper_trading_simulation,
)


@contextmanager
def _temporary_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    stop_event: asyncio.Event,
) -> None:
    """Install SIGINT/SIGTERM handlers that request a graceful shutdown."""

    signals: list[int] = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        signals.append(signal.SIGTERM)

    previous: dict[int, signal.Handlers] = {}

    def _handle(signum: int, frame) -> None:  # type: ignore[override]
        if stop_event.is_set():
            raise KeyboardInterrupt
        loop.call_soon_threadsafe(stop_event.set)

    for signum in signals:
        try:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle)
        except (ValueError, RuntimeError, OSError):  # pragma: no cover - platform guards
            continue

    try:
        yield
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, RuntimeError, OSError):  # pragma: no cover - platform guards
                continue


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the bootstrap runtime against a paper trading API and emit "
            "a JSON summary of observed orders."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional path to a SystemConfig YAML payload. Environment variables "
            "are used when omitted."
        ),
    )
    parser.add_argument(
        "--paper-api-url",
        dest="paper_api_url",
        help="Base URL for the paper trading REST API",
    )
    parser.add_argument(
        "--paper-api-key",
        dest="paper_api_key",
        help="Optional API key for the paper trading endpoint",
    )
    parser.add_argument(
        "--paper-api-secret",
        dest="paper_api_secret",
        help="Optional API secret for the paper trading endpoint",
    )
    parser.add_argument(
        "--paper-default-stage",
        dest="paper_default_stage",
        help="Override the default release stage used for paper execution",
    )
    parser.add_argument(
        "--paper-order-endpoint",
        dest="paper_order_endpoint",
        help="Relative order submission endpoint (default: /v2/orders)",
    )
    parser.add_argument(
        "--paper-order-id-field",
        dest="paper_order_id_field",
        help="Field in the API response that contains the order identifier",
    )
    parser.add_argument(
        "--paper-timeout",
        dest="paper_timeout",
        type=float,
        help="Optional request timeout (seconds) for the paper trading adapter",
    )
    parser.add_argument(
        "--paper-retry-attempts",
        dest="paper_retry_attempts",
        type=int,
        help="Total retry attempts per order for the paper trading adapter",
    )
    parser.add_argument(
        "--paper-retry-backoff",
        dest="paper_retry_backoff",
        type=float,
        help="Base backoff (seconds) between retries for the paper trading adapter",
    )
    parser.add_argument(
        "--paper-failover-threshold",
        dest="paper_failover_threshold",
        type=int,
        help=(
            "Consecutive broker submission failures required before triggering "
            "the paper adapter failover cooldown."
        ),
    )
    parser.add_argument(
        "--paper-failover-cooldown",
        dest="paper_failover_cooldown",
        type=float,
        help="Failover cooldown duration (seconds) applied after the threshold is reached",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        help="Policy ledger path granting limited_live approval to the strategy",
    )
    parser.add_argument(
        "--diary",
        type=Path,
        help="Path to the decision diary JSON file",
    )
    parser.add_argument(
        "--max-ticks",
        dest="max_ticks",
        type=int,
        help="Optional cap on bootstrap runtime ticks (sets BOOTSTRAP_MAX_TICKS)",
    )
    parser.add_argument(
        "--tick-interval",
        dest="tick_interval",
        type=float,
        help="Override the bootstrap tick interval (seconds)",
    )
    parser.add_argument(
        "--min-orders",
        dest="min_orders",
        type=int,
        default=1,
        help="Minimum number of paper orders required for a successful exit",
    )
    parser.add_argument(
        "--runtime-seconds",
        dest="runtime_seconds",
        type=float,
        help="Maximum wall-clock duration for the simulation",
    )
    parser.add_argument(
        "--poll-interval",
        dest="poll_interval",
        type=float,
        default=0.5,
        help="Polling cadence (seconds) for broker telemetry",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help=(
            "Continue running until the runtime limit even after the minimum "
            "order count is met."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        dest="progress_interval",
        type=float,
        default=0.0,
        help=(
            "Emit JSON progress updates to stderr every N seconds. "
            "Set to 0 to disable progress output."
        ),
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional extras to inject into the SystemConfig",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Persist the simulation report to the specified JSON file",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON summary instead of emitting a single line",
    )
    return parser


def _load_config(path: Path | None) -> SystemConfig:
    if path is None:
        return SystemConfig.from_env()
    try:
        return SystemConfig.from_yaml(path)
    except SystemConfigLoadError as exc:  # pragma: no cover - defensive guard
        raise SystemConfigLoadError(f"Failed to load config {path}: {exc}") from exc


def _apply_overrides(config: SystemConfig, args: argparse.Namespace) -> SystemConfig:
    extras: dict[str, str] = dict(config.extras)

    def _set(name: str, value: object | None) -> None:
        if value is None:
            return
        extras[name] = str(value)

    _set("PAPER_TRADING_API_URL", args.paper_api_url)
    _set("PAPER_TRADING_API_KEY", args.paper_api_key)
    _set("PAPER_TRADING_API_SECRET", args.paper_api_secret)
    _set("PAPER_TRADING_ORDER_ENDPOINT", args.paper_order_endpoint)
    _set("PAPER_TRADING_ORDER_ID_FIELD", args.paper_order_id_field)
    _set("PAPER_TRADING_ORDER_TIMEOUT", args.paper_timeout)
    _set("PAPER_TRADING_RETRY_ATTEMPTS", args.paper_retry_attempts)
    _set("PAPER_TRADING_RETRY_BACKOFF", args.paper_retry_backoff)
    _set("PAPER_TRADING_FAILOVER_THRESHOLD", args.paper_failover_threshold)
    _set("PAPER_TRADING_FAILOVER_COOLDOWN", args.paper_failover_cooldown)
    _set("PAPER_TRADING_DEFAULT_STAGE", args.paper_default_stage)
    _set("POLICY_LEDGER_PATH", args.ledger)
    _set("DECISION_DIARY_PATH", args.diary)
    _set("BOOTSTRAP_MAX_TICKS", args.max_ticks)
    _set("BOOTSTRAP_TICK_INTERVAL", args.tick_interval)

    for token in args.extra or ():
        if "=" not in token:
            raise ValueError(f"Invalid --extra value (expected KEY=VALUE): {token}")
        key, value = token.split("=", 1)
        extras[key.strip()] = value

    return config.with_updated(
        run_mode=RunMode.paper,
        connection_protocol=ConnectionProtocol.paper,
        extras=extras,
    )


async def _run_async(args: argparse.Namespace) -> Mapping[str, object]:
    config = _apply_overrides(_load_config(args.config), args)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    progress_callback: Callable[[PaperTradingSimulationProgress], Awaitable[None] | None] | None = None
    effective_progress_interval: float | None = None

    if args.progress_interval and args.progress_interval > 0.0:
        effective_progress_interval = float(args.progress_interval)

        async def _progress_emit(progress: PaperTradingSimulationProgress) -> None:
            snapshot: dict[str, object] = {
                "timestamp": progress.timestamp.isoformat(),
                "runtime_seconds": progress.runtime_seconds,
                "orders_observed": progress.orders_observed,
                "errors_observed": progress.errors_observed,
                "decisions_observed": progress.decisions_observed,
            }
            metrics = progress.paper_metrics or {}
            if metrics:
                snapshot["total_orders"] = metrics.get("total_orders")
                snapshot["successful_orders"] = metrics.get("successful_orders")
                snapshot["failed_orders"] = metrics.get("failed_orders")
                snapshot["success_ratio"] = metrics.get("success_ratio")
                snapshot["failure_ratio"] = metrics.get("failure_ratio")
            failover = progress.failover or {}
            if failover:
                snapshot["failover_active"] = bool(failover.get("active"))
                snapshot["consecutive_failures"] = failover.get("consecutive_failures")
                snapshot["retry_in_seconds"] = failover.get("retry_in_seconds")
            print(
                json.dumps(snapshot, default=_simulation_json_default),
                file=sys.stderr,
                flush=True,
            )

        progress_callback = _progress_emit

    with _temporary_signal_handlers(loop, stop_event):
        report = await run_paper_trading_simulation(
            config,
            min_orders=max(0, args.min_orders or 0),
            max_runtime=args.runtime_seconds,
            poll_interval=max(args.poll_interval, 0.05),
            stop_when_complete=not args.keep_running,
            report_path=args.output,
            stop_event=stop_event,
            progress_callback=progress_callback,
            progress_interval=effective_progress_interval,
        )
    payload = report.to_dict()
    payload.setdefault("orders_observed", len(report.orders))
    payload.setdefault("errors_observed", len(report.errors))
    payload.setdefault("min_orders", max(0, args.min_orders or 0))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        payload = asyncio.run(_run_async(args))
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        parser.error(str(exc))
        return 1

    text = json.dumps(
        payload,
        indent=2 if args.pretty else None,
        default=_simulation_json_default,
    )
    print(text)

    if payload.get("orders_observed", 0) < payload.get("min_orders", 0):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
