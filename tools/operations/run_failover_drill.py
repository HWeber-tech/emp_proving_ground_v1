from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.data_foundation.ingest.configuration import build_institutional_ingest_config
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.governance.system_config import SystemConfig
from src.operations.failover_drill import FailoverDrillSnapshot, execute_failover_drill


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute a Timescale failover drill using saved ingest results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a YAML config file (defaults to environment variables)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional dotenv-style file used to seed SystemConfig extras before overrides",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to a JSON file containing Timescale ingest results",
    )
    parser.add_argument(
        "--dimensions",
        action="append",
        metavar="DIMENSION",
        help="Dimension to failover; can be provided multiple times (defaults to config)",
    )
    parser.add_argument(
        "--scenario",
        default="timescale_failover",
        help="Optional label describing the drill scenario",
    )
    parser.add_argument(
        "--extra",
        action="append",
        metavar="KEY=VALUE",
        help="Override SystemConfig extras using KEY=VALUE pairs",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the drill summary (default: json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the drill summary to a file instead of stdout",
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


def _load_system_config(config_path: Path | None, env_file: Path | None) -> SystemConfig:
    env_overrides: dict[str, str] = {}
    if env_file is not None:
        env_overrides = _load_env_file(env_file)

    if config_path is None:
        env_payload = dict(os.environ)
        if env_overrides:
            env_payload.update(env_overrides)
        return SystemConfig.from_env(env=env_payload)

    base = SystemConfig.from_yaml(config_path)
    if not env_overrides:
        return base

    env_payload = dict(os.environ)
    env_payload.update(env_overrides)
    return SystemConfig.from_env(env=env_payload, defaults=base)


def _apply_extras(config: SystemConfig, overrides: Mapping[str, str]) -> SystemConfig:
    if not overrides:
        return config
    merged = dict(config.extras)
    merged.update({str(k): str(v) for k, v in overrides.items()})
    return config.with_updated(extras=merged)


def _normalise_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalise_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_normalise_value(item) for item in value]
    return str(value)


def _parse_timestamp(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.fromtimestamp(0, tz=UTC)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise TypeError(f"Unsupported timestamp value: {value!r}")


def _iter_result_payloads(payload: object) -> Iterable[tuple[str | None, Mapping[str, object]]]:
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, Mapping):
                raise TypeError("Result entries must be objects")
            yield entry.get("dimension"), entry
        return
    if isinstance(payload, Mapping):
        if "dimension" in payload:
            yield payload.get("dimension"), payload
            return
        for key, value in payload.items():
            if not isinstance(value, Mapping):
                raise TypeError("Result mapping values must be objects")
            entry = dict(value)
            entry.setdefault("dimension", key)
            yield key, entry
        return
    raise TypeError("Results JSON must be an object or array")


def _load_ingest_results(path: Path) -> dict[str, TimescaleIngestResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, TimescaleIngestResult] = {}
    for hint, payload in _iter_result_payloads(data):
        dimension = str(payload.get("dimension") or hint or "timescale" )
        rows = int(payload.get("rows_written", 0))
        symbols_raw = payload.get("symbols", [])
        if isinstance(symbols_raw, Sequence) and not isinstance(symbols_raw, (str, bytes, bytearray)):
            symbols = tuple(str(symbol) for symbol in symbols_raw)
        else:
            symbols = (str(symbols_raw),) if symbols_raw else tuple()
        start_ts = _parse_timestamp(payload.get("start_ts"))
        end_ts = _parse_timestamp(payload.get("end_ts"))
        duration = float(payload.get("ingest_duration_seconds", 0.0))
        freshness_raw = payload.get("freshness_seconds")
        freshness = float(freshness_raw) if freshness_raw is not None else None
        source = payload.get("source")
        if source is not None:
            source = str(source)
        results[dimension] = TimescaleIngestResult(
            rows,
            symbols,
            start_ts,
            end_ts,
            duration,
            freshness,
            dimension,
            source,
        )
    if not results:
        raise ValueError("No ingest results found in JSON payload")
    return results


def _format_snapshot(snapshot: FailoverDrillSnapshot, output_format: str) -> str:
    if output_format == "markdown":
        lines: list[str] = [
            f"# Timescale Failover Drill ({snapshot.scenario})",
            "",
            f"- Status: {snapshot.status.value.upper()}",
            f"- Generated at: {snapshot.generated_at.isoformat()}",
        ]
        requested = snapshot.metadata.get("requested_dimensions")
        if isinstance(requested, Sequence) and requested:
            dims = ", ".join(str(dim) for dim in requested)
            lines.append(f"- Requested dimensions: {dims}")
        lines.append("")
        lines.append("## Components")
        lines.append(snapshot.to_markdown())
        return "\n".join(lines)
    payload = snapshot.as_dict()
    return json.dumps(payload, indent=2, sort_keys=True)


async def _execute_drill(
    *,
    results: Mapping[str, TimescaleIngestResult],
    dimensions: Sequence[str],
    scenario: str,
    ingest_config,
) -> FailoverDrillSnapshot:
    metadata = {
        "requested_dimensions": list(dimensions),
        "ingest_config": _normalise_value(ingest_config.metadata),
    }
    if ingest_config.failover_drill is not None:
        metadata["failover_drill"] = ingest_config.failover_drill.to_metadata()
    return await execute_failover_drill(
        plan=ingest_config.plan,
        results=results,
        fail_dimensions=dimensions,
        scenario=scenario,
        metadata=metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        system_config = _load_system_config(args.config, args.env_file)
        extras = _parse_extra_arguments(args.extra)
        system_config = _apply_extras(system_config, extras)
        ingest_config = build_institutional_ingest_config(system_config)
        if not ingest_config.should_run:
            reason = ingest_config.reason or "Institutional ingest disabled"
            raise RuntimeError(reason)

        results = _load_ingest_results(args.results)

        dimensions: list[str] = []
        if args.dimensions:
            dimensions.extend(str(dim) for dim in args.dimensions)
        elif ingest_config.failover_drill is not None and ingest_config.failover_drill.dimensions:
            dimensions.extend(ingest_config.failover_drill.dimensions)
        else:
            dimensions.extend(results.keys())

        if not dimensions:
            raise RuntimeError("No drill dimensions provided or configured")

        snapshot = asyncio.run(
            _execute_drill(
                results=results,
                dimensions=dimensions,
                scenario=args.scenario,
                ingest_config=ingest_config,
            )
        )
        output = _format_snapshot(snapshot, args.format)
    except Exception as exc:  # pragma: no cover - surfaced to CLI users
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
