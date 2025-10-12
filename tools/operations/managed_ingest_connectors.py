from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence
import logging

from src.data_foundation.ingest.configuration import InstitutionalIngestConfig

from sqlalchemy.engine import make_url

from src.data_foundation.cache.redis_cache import RedisConnectionSettings
from src.data_foundation.ingest.configuration import build_institutional_ingest_config
from src.data_foundation.ingest.institutional_vertical import (
    ConnectivityProbeError,
    InstitutionalIngestProvisioner,
    ProbeCallable,
    plan_managed_manifest,
)
from src.data_foundation.streaming.kafka_stream import (
    KafkaTopicProvisioner,
    KafkaTopicProvisioningSummary,
    resolve_ingest_topic_specs,
)
from src.governance.system_config import SystemConfig
from src.runtime.task_supervisor import TaskSupervisor

logger = logging.getLogger(__name__)


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


def _close_redis_client(client: object) -> None:
    """Best-effort shutdown for Redis clients created by the CLI."""

    for method_name in ("close", "disconnect"):
        method = getattr(client, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("Redis client %s() call failed", method_name, exc_info=True)


def _prepare_redis_client(
    settings: RedisConnectionSettings,
) -> tuple[object | None, str | None]:
    """Instantiate a Redis client and validate connectivity."""

    try:
        client = settings.create_client()
    except Exception as exc:  # pragma: no cover - emits connectivity failure
        return None, f"redis client creation failed: {exc}"

    ping = getattr(client, "ping", None)
    if not callable(ping):
        _close_redis_client(client)
        return None, "redis client does not expose ping()"

    try:
        response = ping()
    except Exception as exc:
        _close_redis_client(client)
        return None, f"redis ping failed: {exc}"

    if not response:
        _close_redis_client(client)
        return None, "redis ping returned falsy response"

    return client, None


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
        "--env-file",
        type=Path,
        help="Optional dotenv-style file used to seed SystemConfig extras before applying overrides",
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
    parser.add_argument(
        "--ensure-topics",
        action="store_true",
        help=(
            "Ensure Kafka ingest topics exist using KafkaTopicProvisioner. "
            "When combined with --topics-dry-run the command reports what would "
            "be created without contacting the broker."
        ),
    )
    parser.add_argument(
        "--topics-dry-run",
        action="store_true",
        help="Run Kafka topic provisioning in dry-run mode.",
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
    kafka_mapping,
    timeout: float,
) -> list[dict[str, object]]:
    provisioner = InstitutionalIngestProvisioner(
        ingest_config,
        redis_settings=ingest_config.redis_settings,
        redis_policy=ingest_config.redis_policy,
        kafka_mapping=kafka_mapping,
    )
    supervisor = TaskSupervisor(namespace="managed-connectors-cli")
    redis_settings = ingest_config.redis_settings
    redis_client: object | None = None
    redis_error: str | None = None
    redis_factory = None
    probe_overrides: dict[str, ProbeCallable] = {}
    if redis_settings and redis_settings.configured:
        redis_client, redis_error = _prepare_redis_client(redis_settings)
        if redis_client is not None:
            redis_factory = lambda _settings: redis_client
        elif redis_error:
            def _redis_probe_failure() -> bool:
                raise ConnectivityProbeError(redis_error)

            probe_overrides["redis"] = _redis_probe_failure

    services = provisioner.provision(
        run_ingest=_noop_ingest,
        event_bus=_CliEventBus(),
        task_supervisor=supervisor,
        redis_client_factory=redis_factory,
    )

    try:
        snapshots = await services.connectivity_report(
            probes=probe_overrides or None,
            timeout=timeout,
        )
        return [snapshot.as_dict() for snapshot in snapshots]
    finally:
        await services.stop()
        cache = getattr(services, "redis_cache", None)
        raw_client = getattr(cache, "raw_client", None) if cache is not None else None
        if redis_client is not None:
            _close_redis_client(redis_client)
        if raw_client is not None and raw_client is not redis_client:
            _close_redis_client(raw_client)


def _render_markdown_sections(report: Mapping[str, object]) -> list[str]:
    """Return Markdown lines describing the managed connector report."""

    lines: list[str] = []
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
            line = f"- {snapshot.get('name')}: {status_label}"
            error = snapshot.get("error")
            if error:
                line += f" - {error}"
            lines.append(line)
    return lines


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = ["# Institutional Ingest Managed Connectors", ""]
    lines.extend(_render_markdown_sections(report))
    return "\n".join(lines)


def _provision_kafka_topics(
    ingest_config,
    kafka_mapping: Mapping[str, str],
    *,
    dry_run: bool,
) -> Mapping[str, object]:
    topic_specs = resolve_ingest_topic_specs(kafka_mapping)
    if not topic_specs:
        return {
            "requested": [],
            "existing": [],
            "created": [],
            "failed": {},
            "dry_run": dry_run,
            "notes": ["no_topics_configured"],
        }

    provisioner = KafkaTopicProvisioner(ingest_config.kafka_settings)
    try:
        summary = provisioner.ensure_topics(topic_specs, dry_run=dry_run)
    except Exception as exc:  # pragma: no cover - defensive guardrail for CLI usage
        return {
            "requested": [spec.name for spec in topic_specs],
            "existing": [],
            "created": [],
            "failed": {"__error__": str(exc)},
            "dry_run": dry_run,
            "notes": ["provisioner_error"],
        }

    if not isinstance(summary, KafkaTopicProvisioningSummary):
        return {
            "requested": [spec.name for spec in topic_specs],
            "existing": [],
            "created": [],
            "failed": {"__error__": "unexpected_provisioner_response"},
            "dry_run": dry_run,
            "notes": ["provisioner_returned_non_summary"],
        }

    payload = summary.as_dict()
    payload.setdefault("requested", list(summary.requested))
    payload.setdefault("dry_run", summary.dry_run)
    return payload


def build_report_for_ingest_config(
    ingest_config: InstitutionalIngestConfig,
    kafka_mapping: Mapping[str, str],
    *,
    connectivity: bool,
    ensure_topics: bool,
    topics_dry_run: bool,
    timeout: float,
) -> dict[str, object]:
    manifest_snapshots = plan_managed_manifest(
        ingest_config,
        redis_settings=ingest_config.redis_settings,
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

    if connectivity:
        _ensure_sqlite_directory(ingest_config.timescale_settings.url)
        connectivity_snapshots = asyncio.run(
            _collect_connectivity(
                ingest_config,
                kafka_mapping,
                timeout=max(0.1, float(timeout)),
            )
        )
        report["connectivity"] = connectivity_snapshots

    if ensure_topics:
        topic_summary = _provision_kafka_topics(
            ingest_config,
            kafka_mapping,
            dry_run=bool(topics_dry_run),
        )
        report["kafka_topic_provisioning"] = topic_summary

    return report


def build_managed_connector_report(
    config: SystemConfig,
    *,
    connectivity: bool = False,
    ensure_topics: bool = False,
    topics_dry_run: bool = False,
    timeout: float = 5.0,
) -> dict[str, object]:
    """Construct the managed connector report for the provided configuration."""

    ingest_config = build_institutional_ingest_config(config)
    kafka_mapping = dict(config.extras)
    return build_report_for_ingest_config(
        ingest_config,
        kafka_mapping,
        connectivity=connectivity,
        ensure_topics=ensure_topics,
        topics_dry_run=topics_dry_run,
        timeout=timeout,
    )


def _generate_report(args: argparse.Namespace) -> dict[str, object]:
    config = _load_system_config(args.config, args.env_file)
    extras = _parse_extra_arguments(args.extra)
    config = _apply_extras(config, extras)

    return build_managed_connector_report(
        config,
        connectivity=bool(args.connectivity),
        ensure_topics=bool(args.ensure_topics),
        topics_dry_run=bool(args.topics_dry_run),
        timeout=float(args.timeout),
    )


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
