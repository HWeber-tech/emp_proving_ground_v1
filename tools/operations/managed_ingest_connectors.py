from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from sqlalchemy.engine import make_url

from src.core.configuration import Configuration
from src.data_foundation.cache.redis_cache import InMemoryRedis, RedisConnectionSettings
from src.data_foundation.ingest.configuration import build_institutional_ingest_config
from src.data_foundation.ingest.institutional_vertical import (
    InstitutionalIngestProvisioner,
    plan_managed_manifest,
)
from src.governance.system_config import SystemConfig
from src.runtime.task_supervisor import TaskSupervisor


class _CliEventBus:
    """Minimal event-bus stub for connectivity checks."""

    def is_running(self) -> bool:  # pragma: no cover - trivial pass-through
        return True

    def publish_from_sync(self, event) -> None:  # pragma: no cover - stub
        return None

    async def publish(self, event) -> None:  # pragma: no cover - stub
        return None


async def _noop_ingest() -> bool:
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise managed Timescale/Redis/Kafka connectors for institutional ingest runs"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a YAML configuration file (defaults to environment variables)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the connector report (default: json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to a file instead of stdout",
    )
    parser.add_argument(
        "--extra",
        action="append",
        metavar="KEY=VALUE",
        help="Override or inject SystemConfig extras using KEY=VALUE pairs",
    )
    parser.add_argument(
        "--connectivity",
        action="store_true",
        help="Evaluate lightweight connectivity probes against provisioned connectors",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for connectivity probes (default: 5.0)",
    )
    return parser


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


def _load_system_config(config_path: Path | None) -> SystemConfig:
    if config_path is None:
        return SystemConfig.from_env()
    configuration = Configuration.from_yaml(config_path)
    return configuration.system_config


def _apply_extras(config: SystemConfig, overrides: Mapping[str, str]) -> SystemConfig:
    if not overrides:
        return config
    merged = dict(config.extras)
    merged.update({str(k): str(v) for k, v in overrides.items()})
    return config.with_updated(extras=merged)


def _ensure_sqlite_directory(url: str) -> None:
    try:
        parsed = make_url(url)
    except Exception:  # pragma: no cover - defensive fallback
        return
    if parsed.get_backend_name() != "sqlite":
        return
    database = parsed.database
    if not database or database == ":memory:":
        return
    path = Path(database)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)


def _plan_dimensions(ingest_config: object) -> list[str]:
    plan = getattr(ingest_config, "plan", None)
    if plan is None:
        return []
    dimensions: list[str] = []
    if getattr(plan, "daily", None) is not None:
        dimensions.append("daily")
    if getattr(plan, "intraday", None) is not None:
        dimensions.append("intraday")
    if getattr(plan, "macro", None) is not None:
        dimensions.append("macro")
    return dimensions


async def _collect_connectivity(
    ingest_config,
    redis_settings: RedisConnectionSettings,
    kafka_mapping,
    timeout: float,
) -> list[dict[str, object]]:
    provisioner = InstitutionalIngestProvisioner(
        ingest_config,
        redis_settings=redis_settings,
        kafka_mapping=kafka_mapping,
    )
    supervisor = TaskSupervisor(namespace="managed-connectors-cli")
    redis_factory = None
    if redis_settings.configured:
        redis_factory = lambda _settings: InMemoryRedis()

    services = provisioner.provision(
        run_ingest=_noop_ingest,
        event_bus=_CliEventBus(),
        task_supervisor=supervisor,
        redis_client_factory=redis_factory,
    )

    try:
        snapshots = await services.connectivity_report(timeout=timeout)
        return [snapshot.as_dict() for snapshot in snapshots]
    finally:
        await services.stop()


def _render_markdown(report: Mapping[str, object]) -> str:
    lines: list[str] = ["# Institutional Ingest Managed Connectors", ""]
    should_run = "yes" if report.get("should_run") else "no"
    lines.append(f"- Should run: {should_run}")
    reason = report.get("reason")
    if reason:
        lines.append(f"- Reason: {reason}")
    dimensions = report.get("dimensions") or []
    if dimensions:
        lines.append(f"- Dimensions: {', '.join(dimensions)}")
    schedule = report.get("schedule")
    if schedule:
        lines.append(
            "- Schedule: interval={interval_seconds}s, jitter={jitter_seconds}s, max_failures={max_failures}".format(
                interval_seconds=schedule.get("interval_seconds"),
                jitter_seconds=schedule.get("jitter_seconds"),
                max_failures=schedule.get("max_failures"),
            )
        )
    failover = report.get("failover")
    if failover:
        lines.append(
            "- Failover drill: "
            + ", ".join(f"{key}={value}" for key, value in failover.items() if value is not None)
        )
    lines.append("")
    lines.append("## Managed Connectors")
    manifest = report.get("manifest") or []
    for snapshot in manifest:
        name = snapshot.get("name", "unknown")
        configured = "configured" if snapshot.get("configured") else "not configured"
        supervised = "supervised" if snapshot.get("supervised") else "unsupervised"
        lines.append(f"- **{name}** – {configured}, {supervised}")
        metadata = snapshot.get("metadata") or {}
        for key, value in metadata.items():
            lines.append(f"    - {key}: {value}")
    connectivity = report.get("connectivity")
    if connectivity:
        lines.append("")
        lines.append("## Connectivity")
        for snapshot in connectivity:
            status = snapshot.get("healthy")
            if status is None:
                status_label = "unknown"
            else:
                status_label = "healthy" if status else "unhealthy"
            lines.append(f"- {snapshot.get('name')}: {status_label}")
    return "\n".join(lines)


def _generate_report(args: argparse.Namespace) -> dict[str, object]:
    config = _load_system_config(args.config)
    extras = _parse_extra_arguments(args.extra)
    config = _apply_extras(config, extras)

    ingest_config = build_institutional_ingest_config(config)
    redis_settings = RedisConnectionSettings.from_mapping(config.extras)
    kafka_mapping = dict(config.extras)

    manifest_snapshots = plan_managed_manifest(
        ingest_config,
        redis_settings=redis_settings,
        kafka_mapping=kafka_mapping,
    )

    schedule_summary = None
    if ingest_config.schedule is not None:
        schedule_summary = {
            "interval_seconds": ingest_config.schedule.interval_seconds,
            "jitter_seconds": ingest_config.schedule.jitter_seconds,
            "max_failures": ingest_config.schedule.max_failures,
        }

    failover_summary = None
    if ingest_config.failover_drill is not None:
        failover_summary = ingest_config.failover_drill.to_metadata()

    report: dict[str, object] = {
        "should_run": ingest_config.should_run,
        "reason": ingest_config.reason,
        "dimensions": _plan_dimensions(ingest_config),
        "schedule": schedule_summary,
        "manifest": [snapshot.as_dict() for snapshot in manifest_snapshots],
        "failover": failover_summary,
    }

    if args.connectivity:
        _ensure_sqlite_directory(ingest_config.timescale_settings.url)
        connectivity = asyncio.run(
            _collect_connectivity(
                ingest_config,
                redis_settings,
                kafka_mapping,
                timeout=max(0.1, float(args.timeout)),
            )
        )
        report["connectivity"] = connectivity

    return report


def _format_report(report: dict[str, object], output_format: str) -> str:
    if output_format == "markdown":
        return _render_markdown(report)
    return json.dumps(report, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        report = _generate_report(args)
        output = _format_report(report, args.format)
    except Exception as exc:  # pragma: no cover - error surface
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
