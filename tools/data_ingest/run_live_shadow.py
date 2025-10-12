"""Run the operational data backbone in live-shadow mode.

This command wires the production-style data backbone (Timescale ingest,
Redis cache warming, Kafka streaming, and sensory fusion) and keeps it running
for a configurable duration. It performs an initial supervised ingest pass to
hydrate Timescale and the sensory cortex, optionally launches a streaming
consumer that relays Kafka ingest events back into the sensory organ, and then
starts the recurring ingest scheduler so the pipeline mirrors the live-shadow
storyline described in the roadmap.

The entry point emits a structured summary (JSON/Markdown/text) capturing the
initial ingest evidence together with connectivity probes, cache metrics, and
ingest scheduler telemetry so operators can archive the run as proof that
real services—TimescaleDB, Redis, and Kafka—are online.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_foundation.ingest.scheduler import IngestSchedule
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.pipelines.operational_backbone import OperationalBackbonePipeline

from tools.data_ingest.run_operational_backbone import (
    OperationalIngestRequest,
    _build_event_bus,
    _build_manager,
    _build_pipeline,
    _build_request,
    _connection_metadata,
    _ensure_backbone_mode,
    _load_system_config,
    _result_payload,
)


DEFAULT_DURATION_SECONDS = 300.0
DEFAULT_INTERVAL_SECONDS = 60.0
DEFAULT_JITTER_SECONDS = 0.0
DEFAULT_MAX_FAILURES = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the operational data backbone in live-shadow mode, keeping the "
            "Timescale → Redis → Kafka → sensory pipeline active for a configured window."
        ),
    )
    parser.add_argument("--config", type=Path, help="Optional path to a SystemConfig YAML file.")
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional dotenv-style file loaded before resolving SystemConfig extras.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        metavar="KEY=VALUE",
        help="Override or inject SystemConfig extras using KEY=VALUE pairs.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        help="Symbols to ingest (defaults to TIMESCALE_SYMBOLS or EURUSD).",
    )
    parser.add_argument(
        "--daily-lookback",
        type=int,
        default=None,
        help="Override the number of days to fetch for daily bars.",
    )
    parser.add_argument(
        "--intraday-lookback",
        type=int,
        default=None,
        help="Override the number of days to fetch for intraday trades.",
    )
    parser.add_argument(
        "--intraday-interval",
        type=str,
        default=None,
        help="Override the intraday interval (defaults to config extras or 1m).",
    )
    parser.add_argument("--macro-start", type=str, help="Optional macro ingest start timestamp.")
    parser.add_argument("--macro-end", type=str, help="Optional macro ingest end timestamp.")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="How long to keep the scheduler running (seconds, default: 300).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Ingest scheduler interval in seconds (default: 60).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=DEFAULT_JITTER_SECONDS,
        help="Optional jitter applied to the ingest interval in seconds (default: 0).",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=DEFAULT_MAX_FAILURES,
        help="Consecutive failures before the ingest scheduler stops (default: 3).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown", "text"),
        default="json",
        help="Output format for the run summary (default: json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the summary to a file instead of stdout.",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Enable Kafka streaming to feed sensory snapshots (default: enabled).",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable Kafka streaming and sensory updates from ingest events.",
    )
    parser.add_argument(
        "--require-connectors",
        action="store_true",
        help="Require Timescale, Redis, and Kafka connectors; disable in-memory fallbacks.",
    )
    parser.add_argument(
        "--require-timescale",
        action="store_true",
        help="Require a configured Timescale connection (implies --require-connectors).",
    )
    parser.add_argument(
        "--require-redis",
        action="store_true",
        help="Require a configured Redis cache (implies --require-connectors).",
    )
    parser.add_argument(
        "--require-kafka",
        action="store_true",
        help="Require a configured Kafka publisher (implies --require-connectors).",
    )
    return parser


def _plan_from_request(request: OperationalIngestRequest) -> TimescaleBackbonePlan:
    symbols = request.normalised_symbols()

    daily_plan: DailyBarIngestPlan | None = None
    if request.daily_lookback_days and request.daily_lookback_days > 0:
        daily_plan = DailyBarIngestPlan(
            symbols=symbols,
            lookback_days=int(request.daily_lookback_days),
            source=request.source,
        )

    intraday_plan: IntradayTradeIngestPlan | None = None
    if request.intraday_lookback_days and request.intraday_lookback_days > 0:
        intraday_plan = IntradayTradeIngestPlan(
            symbols=symbols,
            lookback_days=int(request.intraday_lookback_days),
            interval=request.intraday_interval,
            source=request.source,
        )

    macro_plan: MacroEventIngestPlan | None = None
    if request.macro_events:
        macro_plan = MacroEventIngestPlan(
            events=tuple(request.macro_events),
            source=request.macro_source,
        )
    elif request.macro_start and request.macro_end:
        macro_plan = MacroEventIngestPlan(
            start=request.macro_start,
            end=request.macro_end,
            source=request.macro_source,
        )

    return TimescaleBackbonePlan(
        daily=daily_plan,
        intraday=intraday_plan,
        macro=macro_plan,
    )


def _scheduler_metadata(duration: float) -> Mapping[str, object]:
    return {
        "origin": "live_shadow_cli",
        "duration_seconds": duration,
    }


async def _run_live_shadow(args: argparse.Namespace) -> dict[str, Any]:
    config = _ensure_backbone_mode(_load_system_config(args))
    request = _build_request(config, args)

    require_timescale = args.require_timescale or args.require_connectors
    require_redis = args.require_redis or args.require_connectors
    require_kafka = args.require_kafka or args.require_connectors

    manager = _build_manager(
        config,
        require_timescale=require_timescale,
        require_redis=require_redis,
        require_kafka=require_kafka,
    )
    event_bus = _build_event_bus()
    pipeline = _build_pipeline(manager=manager, event_bus=event_bus, config=config)

    initial_result = await pipeline.execute(request)

    scheduler = None
    scheduler_state: Mapping[str, object] | None = None
    streaming_snapshots: Mapping[str, Mapping[str, Any]] = {}
    streaming_task = None
    duration = max(float(args.duration or 0.0), 0.0)

    start_ts = time.perf_counter()
    try:
        if args.stream:
            streaming_task = await pipeline.start_streaming(
                metadata={"origin": "live_shadow_cli"}
            )

        if duration > 0:
            plan_factory = lambda: _plan_from_request(request)
            schedule = IngestSchedule(
                interval_seconds=float(args.interval or DEFAULT_INTERVAL_SECONDS),
                jitter_seconds=float(args.jitter or DEFAULT_JITTER_SECONDS),
                max_failures=int(args.max_failures if args.max_failures is not None else DEFAULT_MAX_FAILURES),
            )
            scheduler = manager.start_ingest_scheduler(
                plan_factory,
                schedule,
                metadata=_scheduler_metadata(duration),
            )
            try:
                await asyncio.sleep(duration)
            except asyncio.CancelledError:
                raise
        else:
            await asyncio.sleep(0)
    finally:
        elapsed = time.perf_counter() - start_ts
        streaming_snapshots = pipeline.streaming_snapshots if args.stream else {}
        cache_metrics = manager.cache_metrics(reset=False)
        connectivity = manager.connectivity_report().as_dict()
        if scheduler is not None:
            scheduler_state = scheduler.state().as_dict()
            await manager.stop_ingest_scheduler()
        else:
            scheduler_state = None

        if streaming_task is not None:
            await pipeline.stop_streaming()

        await pipeline.shutdown()
        await manager.shutdown()

    initial_payload = _result_payload(
        config=config,
        request=request,
        result=initial_result,
    )

    summary: dict[str, Any] = {
        "initial": initial_payload,
        "connectivity": connectivity,
        "cache_metrics": dict(cache_metrics),
        "scheduler": scheduler_state,
        "streaming": {
            "enabled": bool(args.stream),
            "snapshots": {symbol: dict(snapshot) for symbol, snapshot in streaming_snapshots.items()},
        },
        "connections": dict(_connection_metadata(config)),
        "duration_seconds": elapsed,
    }
    return summary


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = ["# Live Shadow Summary"]
    lines.append(f"- Duration (seconds): {payload.get('duration_seconds', 0):.2f}")

    scheduler = payload.get("scheduler") or {}
    if scheduler:
        lines.append("- Scheduler running: {running}".format(running=scheduler.get("running")))
        lines.append(
            "- Scheduler next run: {next}".format(next=scheduler.get("next_run_at") or "n/a")
        )

    connectivity = payload.get("connectivity") or {}
    if connectivity:
        lines.append("\n## Connector Health")
        lines.append("| Service | Healthy | Status |")
        lines.append("| --- | --- | --- |")
        for service in ("timescale", "redis", "kafka"):
            healthy = connectivity.get(service)
            probe = next(
                (
                    probe
                    for probe in connectivity.get("probes", [])
                    if isinstance(probe, Mapping) and probe.get("name") == service
                ),
                {},
            )
            lines.append(
                f"| {service} | {healthy} | {probe.get('status', 'unknown')} |"
            )

    initial = payload.get("initial") or {}
    ingest_results = initial.get("ingest_results") or {}
    if ingest_results:
        lines.append("\n## Initial Ingest")
        lines.append("| Dimension | Rows | Freshness (s) |")
        lines.append("| --- | --- | --- |")
        for dimension, result in ingest_results.items():
            rows = result.get("rows_written", "-")
            freshness = result.get("freshness_seconds", "-")
            lines.append(f"| {dimension} | {rows} | {freshness} |")

    return "\n".join(lines)


def _render_text(payload: Mapping[str, Any]) -> str:
    fragments = [
        f"Duration: {payload.get('duration_seconds', 0):.2f}s",
        "Scheduler: {}".format(payload.get("scheduler", {})),
        "Connectivity: {}".format(payload.get("connectivity", {})),
        "Cache metrics: {}".format(payload.get("cache_metrics", {})),
        "Streaming: {}".format(payload.get("streaming", {})),
    ]
    return "\n".join(fragments)


def _write_output(text: str, *, path: Path | None) -> None:
    if path is None:
        print(text)
        return
    path.write_text(text, encoding="utf-8")


def _emit_summary(payload: Mapping[str, Any], args: argparse.Namespace) -> None:
    if args.format == "json":
        text = json.dumps(payload, indent=2)
    elif args.format == "markdown":
        text = _render_markdown(payload)
    else:
        text = _render_text(payload)
    _write_output(text, path=args.output)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        payload = asyncio.run(_run_live_shadow(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover - surfaced in CLI tests
        print(f"live-shadow ingest failed: {exc}", file=sys.stderr)
        return 1

    _emit_summary(payload, args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    os.environ.setdefault("PYTHONASYNCIODEBUG", "0")
    raise SystemExit(main())
