"""Execute the operational data backbone ingest pipeline from the command line.

This helper wires the ``OperationalBackbonePipeline`` around a ``RealDataManager``
using the connection details supplied via ``SystemConfig`` extras (either from a
YAML configuration file or the ambient environment).  It coordinates Timescale
ingest, Redis cache warming, Kafka telemetry fan-out, and sensory fusion in a
single command so operators can generate evidence bundles without writing
bespoke scripts.

The command prints a summary of the ingest run (rows written, cache metrics,
Kafka events, and sensory snapshot details) in JSON, Markdown, or human-readable
text formats.  This mirrors the roadmap requirement for a reproducible
store→cache→stream drill and feeds the managed ingest environment runbook.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.event_bus import Event  # noqa: E402
from src.data_foundation.persist.timescale import TimescaleIngestResult  # noqa: E402
from src.data_foundation.pipelines.operational_backbone import (  # noqa: E402
    OperationalBackbonePipeline,
    OperationalBackboneResult,
    OperationalIngestRequest,
)
from src.data_foundation.streaming.kafka_stream import (  # noqa: E402
    KafkaConnectionSettings,
    create_ingest_event_consumer,
)
from src.data_integration.real_data_integration import RealDataManager  # noqa: E402
from src.governance.system_config import (  # noqa: E402
    DataBackboneMode,
    SystemConfig,
    SystemConfigLoadError,
)
from src.sensory.real_sensory_organ import RealSensoryOrgan  # noqa: E402
from src.core.event_bus import EventBus  # noqa: E402  (import after sys.path mutation)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the operational data backbone ingest pipeline and summarise the results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a SystemConfig YAML file (defaults to environment variables).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional dotenv-style file used to seed SystemConfig extras before applying overrides.",
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
    parser.add_argument(
        "--macro-start",
        type=str,
        help="Optional macro ingest start timestamp (ISO-8601).",
    )
    parser.add_argument(
        "--macro-end",
        type=str,
        help="Optional macro ingest end timestamp (ISO-8601).",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown", "text"),
        default="json",
        help="Output format for the ingest summary (default: json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the summary to a file instead of stdout.",
    )
    parser.add_argument(
        "--require-connectors",
        action="store_true",
        help="Require Timescale, Redis, and Kafka connectors; disable in-memory fallbacks.",
    )
    parser.add_argument(
        "--require-timescale",
        action="store_true",
        help="Require a configured Timescale connection (implies --require-connectors when set).",
    )
    parser.add_argument(
        "--require-redis",
        action="store_true",
        help="Require a configured Redis cache (implies --require-connectors when set).",
    )
    parser.add_argument(
        "--require-kafka",
        action="store_true",
        help="Require a configured Kafka publisher (implies --require-connectors when set).",
    )
    return parser


def _load_env_file(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for idx, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"env file line {idx} missing '=': {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"env file line {idx} has empty key")
        payload[key] = value
    return payload


def _parse_extra_arguments(entries: Sequence[str] | None) -> dict[str, str]:
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


def _load_system_config(args: argparse.Namespace) -> SystemConfig:
    env_payload = dict(os.environ)
    if args.env_file:
        env_payload.update(_load_env_file(args.env_file))

    extras_override = _parse_extra_arguments(args.extra)

    if args.config:
        base = SystemConfig.from_yaml(args.config, env=env_payload)
    else:
        base = SystemConfig.from_env(env=env_payload)

    if extras_override:
        merged = dict(base.extras)
        merged.update(extras_override)
        base = base.with_updated(extras=merged)

    return base


def _ensure_backbone_mode(config: SystemConfig) -> SystemConfig:
    if config.data_backbone_mode is DataBackboneMode.institutional:
        return config
    # Promote to institutional mode if credentials are present so that the
    # RealDataManager provisions Timescale/Redis/Kafka connectors as expected.
    return config.with_updated(data_backbone_mode=DataBackboneMode.institutional)


def _resolve_symbols(config: SystemConfig, provided: Sequence[str] | None) -> tuple[str, ...]:
    if provided:
        return tuple({symbol.strip().upper() for symbol in provided if symbol.strip()})
    candidates = config.extras.get("TIMESCALE_SYMBOLS") or config.extras.get("SYMBOLS")
    if candidates:
        symbols = [token.strip().upper() for token in candidates.split(",") if token.strip()]
        if symbols:
            return tuple(dict.fromkeys(symbols))
    return ("EURUSD",)


def _resolve_intraday_interval(config: SystemConfig, override: str | None) -> str:
    if override and override.strip():
        return override.strip()
    default = config.extras.get("TIMESCALE_INTRADAY_INTERVAL")
    return default.strip() if isinstance(default, str) and default.strip() else "1m"


def _build_request(config: SystemConfig, args: argparse.Namespace) -> OperationalIngestRequest:
    symbols = _resolve_symbols(config, args.symbols)
    request = OperationalIngestRequest(
        symbols=symbols,
        daily_lookback_days=args.daily_lookback,
        intraday_lookback_days=args.intraday_lookback,
        intraday_interval=_resolve_intraday_interval(config, args.intraday_interval),
        macro_start=args.macro_start,
        macro_end=args.macro_end,
    )
    return request


def _connection_metadata(config: SystemConfig) -> Mapping[str, str | None]:
    extras = config.extras
    redis_url = extras.get("REDIS_URL") or extras.get("CACHE_URL")
    if not redis_url and extras.get("REDIS_HOST"):
        redis_port = extras.get("REDIS_PORT", "6379")
        redis_db = extras.get("REDIS_DB", "0")
        redis_url = f"redis://{extras['REDIS_HOST']}:{redis_port}/{redis_db}"
    kafka_url = (
        extras.get("KAFKA_BOOTSTRAP_SERVERS")
        or extras.get("KAFKA_BROKERS")
        or extras.get("KAFKA_URL")
    )
    timescale_url = extras.get("TIMESCALE_URL") or extras.get("TIMESCALEDB_URL")
    return {
        "timescale_url": timescale_url,
        "redis_url": redis_url,
        "kafka_bootstrap": kafka_url,
    }


def _build_manager(
    config: SystemConfig,
    *,
    require_timescale: bool | None = None,
    require_redis: bool | None = None,
    require_kafka: bool | None = None,
) -> RealDataManager:
    return RealDataManager(
        system_config=config,
        require_timescale=require_timescale,
        require_redis=require_redis,
        require_kafka=require_kafka,
    )


def _build_event_bus() -> EventBus:
    return EventBus()


def _build_pipeline(
    *,
    manager: RealDataManager,
    event_bus: EventBus,
    config: SystemConfig,
) -> OperationalBackbonePipeline:
    kafka_settings = KafkaConnectionSettings.from_mapping(config.extras)

    def _consumer_factory():
        return create_ingest_event_consumer(
            kafka_settings,
            config.extras,
            event_bus=event_bus,
        )

    return OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=_consumer_factory,
        sensory_organ=RealSensoryOrgan(),
    )


async def _execute_pipeline(
    *,
    pipeline: OperationalBackbonePipeline,
    request: OperationalIngestRequest,
) -> OperationalBackboneResult:
    try:
        return await pipeline.execute(request)
    finally:
        await pipeline.shutdown()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return value.tz_convert("UTC").isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return value.astimezone().isoformat()
    return str(value)


def _frame_summary(frame: pd.DataFrame) -> Mapping[str, Any]:
    rowcount = int(frame.shape[0])
    summary: dict[str, Any] = {"rows": rowcount}
    if rowcount == 0:
        return summary
    timestamp_column = next((col for col in ("timestamp", "ts") if col in frame.columns), None)
    if timestamp_column:
        coerced = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        if not coerced.empty:
            summary["start"] = _iso(coerced.min())
            summary["end"] = _iso(coerced.max())
    if "symbol" in frame.columns:
        summary["symbols"] = sorted({str(symbol) for symbol in frame["symbol"].dropna().unique()})
    return summary


def _event_summary(events: Sequence[Event]) -> list[Mapping[str, Any]]:
    summary: list[Mapping[str, Any]] = []
    for event in events:
        payload_keys: Sequence[str]
        if isinstance(event.payload, Mapping):
            payload_keys = sorted(event.payload.keys())
        else:
            payload_keys = ["value"]
        summary.append(
            {
                "type": event.type,
                "source": event.source,
                "payload_keys": list(payload_keys),
                "timestamp": event.timestamp,
            }
        )
    return summary


def _result_payload(
    *,
    config: SystemConfig,
    request: OperationalIngestRequest,
    result: OperationalBackboneResult,
) -> dict[str, Any]:
    ingest = {
        dimension: (
            data.as_dict() if isinstance(data, TimescaleIngestResult) else asdict(data)
        )
        for dimension, data in result.ingest_results.items()
    }
    frames = {dimension: _frame_summary(frame) for dimension, frame in result.frames.items()}

    payload: dict[str, Any] = {
        "symbols": list(request.normalised_symbols()),
        "ingest_results": ingest,
        "frames": frames,
        "cache_metrics": {
            "before": dict(result.cache_metrics_before),
            "after_ingest": dict(result.cache_metrics_after_ingest),
            "after_fetch": dict(result.cache_metrics_after_fetch),
        },
        "events": _event_summary(result.kafka_events),
        "sensory_snapshot": result.sensory_snapshot,
        "connections": dict(_connection_metadata(config)),
    }
    if result.ingest_error:
        payload["ingest_error"] = result.ingest_error
    if result.belief_state is not None:
        payload["belief_state"] = result.belief_state.as_dict()
    if result.regime_signal is not None:
        payload["regime_signal"] = result.regime_signal.as_dict()
    if result.belief_snapshot is not None:
        snapshot = {
            "belief_id": result.belief_snapshot.belief_id,
            "regime": result.belief_snapshot.regime_state.regime,
            "confidence": result.belief_snapshot.regime_state.confidence,
            "features": dict(result.belief_snapshot.features),
        }
        if result.belief_snapshot.feature_flags:
            snapshot["feature_flags"] = dict(result.belief_snapshot.feature_flags)
        payload["belief_snapshot"] = snapshot
    if result.understanding_decision is not None:
        decision = result.understanding_decision.decision
        payload["understanding_decision"] = {
            "tactic_id": decision.tactic_id,
            "selected_weight": decision.selected_weight,
            "guardrails": dict(decision.guardrails),
            "experiments": list(decision.experiments_applied),
        }
    return payload


def _format_markdown(payload: Mapping[str, Any]) -> str:
    rows = ["| Dimension | Rows | Start | End |", "| --- | --- | --- | --- |"]
    frames = payload.get("frames", {})
    if isinstance(frames, Mapping):
        for dimension, info in frames.items():
            if not isinstance(info, Mapping):
                continue
            rows.append(
                "| {dimension} | {rows} | {start} | {end} |".format(
                    dimension=dimension,
                    rows=info.get("rows", 0),
                    start=info.get("start", "-"),
                    end=info.get("end", "-"),
                )
            )
    event_count = len(payload.get("events", [])) if isinstance(payload.get("events"), list) else 0
    symbols = ", ".join(payload.get("symbols", []))
    header = [
        f"### Operational Backbone Summary",
        f"* Symbols: {symbols or 'n/a'}",
        f"* Kafka events processed: {event_count}",
    ]
    connections = payload.get("connections")
    if isinstance(connections, Mapping):
        header.append("* Connections:")
        for key, value in connections.items():
            header.append(f"  * {key}: {value or 'n/a'}")
    return "\n".join(header + ["", *rows])


def _write_output(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        config = _ensure_backbone_mode(_load_system_config(args))
    except (SystemConfigLoadError, ValueError) as exc:
        parser.error(str(exc))
        return 1

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

    try:
        result = asyncio.run(_execute_pipeline(pipeline=pipeline, request=request))
    except Exception as exc:  # pragma: no cover - surfaced as CLI failure
        parser.error(f"Pipeline execution failed: {exc}")
        return 1

    payload = _result_payload(config=config, request=request, result=result)

    if args.format == "json":
        text = json.dumps(payload, indent=2, sort_keys=True)
    elif args.format == "markdown":
        text = _format_markdown(payload)
    else:
        symbols = ", ".join(payload.get("symbols", [])) or "n/a"
        connections = payload.get("connections", {})
        text_lines = [
            f"Symbols: {symbols}",
            f"Timescale URL: {connections.get('timescale_url') or 'n/a'}",
            f"Redis URL: {connections.get('redis_url') or 'n/a'}",
            f"Kafka bootstrap: {connections.get('kafka_bootstrap') or 'n/a'}",
            f"Kafka events processed: {len(payload.get('events', []))}",
        ]
        frames = payload.get("frames", {})
        if isinstance(frames, Mapping):
            for dimension, info in frames.items():
                if isinstance(info, Mapping):
                    text_lines.append(
                        f"{dimension}: rows={info.get('rows', 0)} start={info.get('start', '-') } end={info.get('end', '-')}"
                    )
        text = "\n".join(text_lines)

    _write_output(text, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
