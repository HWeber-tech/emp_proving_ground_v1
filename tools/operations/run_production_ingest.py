"""CLI for orchestrating the institutional Timescale ingest slice.

The roadmap calls for a production-grade ingest slice that runs against
parameterised SQL, supervised background tasks, and emits telemetry aligned
with the data-backbone briefs.  This command wires together the existing ingest
configuration helpers, runtime task supervision, and the production ingest
slice so operators can trigger bootstrap runs or supervised schedules without
writing bespoke glue code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from src.core.event_bus import get_global_bus
from src.data_foundation.cache.redis_cache import RedisConnectionSettings
from src.data_foundation.ingest.configuration import (
    InstitutionalIngestConfig,
    build_institutional_ingest_config,
)
from src.data_foundation.ingest.production_slice import ProductionIngestSlice
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.governance.system_config import SystemConfig
from src.runtime.task_supervisor import TaskSupervisor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the institutional Timescale ingest slice using production wiring",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a SystemConfig YAML file (defaults to environment)",
    )
    parser.add_argument(
        "--extra",
        action="append",
        metavar="KEY=VALUE",
        help="Override SystemConfig extras without editing files or environment",
    )
    parser.add_argument(
        "--mode",
        choices=("once", "schedule"),
        default="once",
        help="Execute a single ingest run or start the supervised schedule (default: once)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "When running in schedule mode, keep services alive for the given seconds. "
            "Defaults to running until interrupted."
        ),
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help=(
            "Skip the bootstrap run before starting the scheduler. "
            "Only applies when --mode schedule is selected."
        ),
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="production-ingest",
        help="Task supervisor namespace for background jobs (default: production-ingest)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown", "none"),
        default="json",
        help="Output format for the ingest summary (default: json)",
    )
    parser.add_argument(
        "--fallback-symbols",
        type=str,
        help="Comma-separated symbols used when TIMESCALE_SYMBOLS is not set",
    )
    return parser


def _parse_extra_arguments(entries: Sequence[str] | None) -> Mapping[str, str]:
    overrides: dict[str, str] = {}
    if not entries:
        return overrides
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"extras override must be KEY=VALUE: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"extras override has empty key: {raw}")
        overrides[key] = value
    return overrides


def _load_system_config(config_path: Path | None) -> SystemConfig:
    if config_path is None:
        return SystemConfig.from_env()
    return SystemConfig.from_yaml(config_path)


def _apply_extras(config: SystemConfig, overrides: Mapping[str, str]) -> SystemConfig:
    if not overrides:
        return config
    merged = dict(config.extras)
    merged.update({str(k): str(v) for k, v in overrides.items()})
    return config.with_updated(extras=merged)


def _fallback_symbols(payload: str | None) -> Sequence[str]:
    if not payload:
        return ()
    tokens = [segment.strip() for segment in payload.split(",")]
    return [token for token in tokens if token]


def _redis_settings(config: SystemConfig) -> RedisConnectionSettings | None:
    settings = RedisConnectionSettings.from_mapping(config.extras)
    return settings if settings.configured else None


def _kafka_mapping(config: SystemConfig) -> Mapping[str, str]:
    return {str(key): str(value) for key, value in config.extras.items() if key.startswith("KAFKA_")}


def _format_summary_json(summary: Mapping[str, object]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)


def _format_summary_markdown(summary: Mapping[str, object]) -> str:
    lines: list[str] = ["# Production ingest summary", ""]
    lines.append(f"- Should run: {'yes' if summary.get('should_run') else 'no'}")
    reason = summary.get("reason")
    if reason:
        lines.append(f"- Reason: {reason}")
    last_error = summary.get("last_error")
    if last_error:
        lines.append(f"- Last error: {last_error}")
    last_run_at = summary.get("last_run_at")
    if last_run_at:
        lines.append(f"- Last run at: {last_run_at}")
    services = summary.get("services")
    if isinstance(services, Mapping):
        scheduler = services.get("schedule")
        if isinstance(scheduler, Mapping):
            interval = scheduler.get("interval_seconds")
            jitter = scheduler.get("jitter_seconds")
            running = scheduler.get("running")
            lines.append(
                f"- Scheduler: interval={interval}s jitter={jitter}s running={'yes' if running else 'no'}"
            )
    last_results = summary.get("last_results")
    if isinstance(last_results, Mapping) and last_results:
        lines.append("")
        lines.append("## Last ingest results")
        for dimension, payload in sorted(last_results.items()):
            if not isinstance(payload, Mapping):
                continue
            rows = payload.get("rows_written")
            symbols = payload.get("symbols")
            lines.append(f"- {dimension}: rows={rows} symbols={symbols}")
    return "\n".join(lines)


def _serialise_summary(
    summary: Mapping[str, object],
    *,
    fmt: str,
) -> str | None:
    if fmt == "none":
        return None
    if fmt == "markdown":
        return _format_summary_markdown(summary)
    return _format_summary_json(summary)


def _ingest_results_to_serialisable(summary: Mapping[str, object]) -> dict[str, object]:
    serialised = dict(summary)
    last_results = summary.get("last_results")
    if isinstance(last_results, Mapping):
        payload: dict[str, dict[str, object]] = {}
        for key, value in last_results.items():
            if isinstance(value, TimescaleIngestResult):
                payload[key] = value.as_dict()
            elif isinstance(value, Mapping):
                payload[key] = {str(k): v for k, v in value.items()}
        serialised["last_results"] = payload
    return serialised


async def _run_once(slice_runtime: ProductionIngestSlice) -> bool:
    return await slice_runtime.run_once()


async def _run_schedule(
    slice_runtime: ProductionIngestSlice,
    *,
    duration: float,
    bootstrap: bool,
) -> bool:
    success = True
    if not bootstrap:
        success = True
    else:
        success = await slice_runtime.run_once()

    slice_runtime.start()
    try:
        if duration > 0:
            await asyncio.sleep(duration)
        else:
            while True:
                await asyncio.sleep(3_600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        success = success and True
    finally:
        await slice_runtime.stop()
    return success


async def _main_async(args: argparse.Namespace) -> int:
    overrides = _parse_extra_arguments(args.extra)
    config = _load_system_config(args.config)
    config = _apply_extras(config, overrides)

    fallback = _fallback_symbols(args.fallback_symbols)
    ingest_config = build_institutional_ingest_config(config, fallback_symbols=fallback)

    if not ingest_config.should_run:
        summary = _ingest_results_to_serialisable(ingest_config.metadata | {
            "should_run": ingest_config.should_run,
            "reason": ingest_config.reason,
            "services": None,
            "last_results": {},
            "last_run_at": None,
            "last_error": ingest_config.reason,
        })
        payload = _serialise_summary(summary, fmt=args.format)
        if payload:
            print(payload)
        return 0

    redis_settings = _redis_settings(config)
    kafka_mapping = _kafka_mapping(config)

    supervisor = TaskSupervisor(namespace=args.namespace)
    event_bus = get_global_bus()

    slice_runtime = ProductionIngestSlice(
        ingest_config,
        event_bus,
        supervisor,
        redis_settings=redis_settings,
        kafka_mapping=kafka_mapping if kafka_mapping else None,
    )

    if args.mode == "schedule":
        success = await _run_schedule(
            slice_runtime,
            duration=max(0.0, float(args.duration)),
            bootstrap=not args.skip_bootstrap,
        )
    else:
        success = await _run_once(slice_runtime)

    summary = _ingest_results_to_serialisable(slice_runtime.summary())
    payload = _serialise_summary(summary, fmt=args.format)
    if payload:
        print(payload)

    return 0 if success else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - argparse already validated
        return exc.code

    try:
        return asyncio.run(_main_async(args))
    except ValueError as exc:
        parser.error(str(exc))  # pragma: no cover - surfaced earlier
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive interruption
        return 130


if __name__ == "__main__":
    sys.exit(main())
