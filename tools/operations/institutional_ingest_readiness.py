"""Readiness checks for the institutional ingest vertical.

This CLI stitches together the managed connector report and failover drill
execution so operations teams can capture evidence for Timescale, Redis, and
Kafka provisioning alongside disaster-recovery rehearsals.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Mapping, Sequence

from src.data_foundation.cache.redis_cache import InMemoryRedis
from src.data_foundation.ingest.configuration import build_institutional_ingest_config
from src.data_foundation.ingest.institutional_vertical import (
    InstitutionalIngestProvisioner,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.governance.system_config import SystemConfig
from src.runtime.task_supervisor import TaskSupervisor

from tools.operations import managed_ingest_connectors as mic
from tools.operations import run_failover_drill as drill


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture institutional ingest readiness evidence by combining managed "
            "connector summaries, optional connectivity checks, and failover drills."
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
        "--extra",
        action="append",
        metavar="KEY=VALUE",
        help="Override or inject SystemConfig extras using KEY=VALUE pairs",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for the readiness report (default: json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report to a file instead of stdout",
    )
    parser.add_argument(
        "--connectivity",
        action="store_true",
        help="Evaluate lightweight connectivity probes using the managed provisioner",
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
        help="Ensure Kafka ingest topics exist using KafkaTopicProvisioner",
    )
    parser.add_argument(
        "--topics-dry-run",
        action="store_true",
        help="Run Kafka topic provisioning in dry-run mode",
    )
    parser.add_argument(
        "--ingest-results",
        type=Path,
        help="Path to a JSON file containing Timescale ingest results for failover drills",
    )
    parser.add_argument(
        "--dimensions",
        action="append",
        metavar="DIMENSION",
        help="Override failover drill dimensions (repeat for multiple entries)",
    )
    parser.add_argument(
        "--scenario",
        help="Override the failover drill scenario label",
    )
    return parser


def _load_managed_config(args: argparse.Namespace) -> SystemConfig:
    config = mic._load_system_config(args.config, args.env_file)
    extras = mic._parse_extra_arguments(args.extra)
    return mic._apply_extras(config, extras)


def _render_markdown(report: Mapping[str, object]) -> str:
    lines: list[str] = ["# Institutional Ingest Readiness", ""]

    connectors = report.get("managed_connectors") or {}
    connector_lines = mic._render_markdown_sections(connectors) if connectors else []
    if connector_lines:
        lines.append("## Managed Connectors")
        lines.extend(connector_lines)

    failover = report.get("failover_drill")
    if isinstance(failover, Mapping):
        lines.append("")
        scenario = str(failover.get("scenario") or "timescale_failover")
        lines.append(f"## Failover Drill ({scenario})")
        lines.append(f"- Status: {failover.get('status', 'unknown').upper()}")
        generated_at = failover.get("generated_at")
        if generated_at:
            lines.append(f"- Generated at: {generated_at}")
        metadata = failover.get("metadata") or {}
        requested = metadata.get("requested_dimensions")
        if isinstance(requested, Sequence) and requested:
            dims = ", ".join(str(dim) for dim in requested)
            lines.append(f"- Requested dimensions: {dims}")
        components = failover.get("components") or []
        if components:
            lines.append("")
            lines.append("### Components")
            for component in components:
                name = component.get("name", "component")
                status = component.get("status", "unknown").upper()
                summary = component.get("summary", "")
                line = f"- {name}: {status}"
                if summary:
                    line += f" – {summary}"
                lines.append(line)

    return "\n".join(lines)


async def _execute_failover_drill(
    ingest_config,
    kafka_mapping: Mapping[str, str],
    results: Mapping[str, TimescaleIngestResult],
    *,
    dimensions: Sequence[str] | None,
    scenario: str | None,
) -> Mapping[str, object]:
    provisioner = InstitutionalIngestProvisioner(
        ingest_config,
        redis_settings=ingest_config.redis_settings,
        redis_policy=ingest_config.redis_policy,
        kafka_mapping=kafka_mapping,
    )
    supervisor = TaskSupervisor(namespace="institutional-ingest-readiness")

    redis_factory = None
    if ingest_config.redis_settings.configured:
        redis_factory = lambda _settings: InMemoryRedis()

    services = provisioner.provision(
        run_ingest=mic._noop_ingest,
        event_bus=mic._CliEventBus(),
        task_supervisor=supervisor,
        redis_client_factory=redis_factory,
    )

    try:
        snapshot = await services.run_failover_drill(
            results,
            fail_dimensions=dimensions,
            scenario=scenario,
        )
        return snapshot.as_dict()
    finally:
        await services.stop()


def _load_ingest_results(path: Path) -> Mapping[str, TimescaleIngestResult]:
    data_path = path.expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"ingest results file not found: {data_path}")
    return drill._load_ingest_results(data_path)


def _build_report(args: argparse.Namespace) -> dict[str, object]:
    config = _load_managed_config(args)
    ingest_config = build_institutional_ingest_config(config)
    kafka_mapping = dict(config.extras)

    managed_connectors = mic.build_report_for_ingest_config(
        ingest_config,
        kafka_mapping,
        connectivity=bool(args.connectivity),
        ensure_topics=bool(args.ensure_topics),
        topics_dry_run=bool(args.topics_dry_run),
        timeout=float(args.timeout),
    )

    report: dict[str, object] = {
        "managed_connectors": managed_connectors,
    }

    if args.ingest_results is not None:
        results = _load_ingest_results(args.ingest_results)
        mic._ensure_sqlite_directory(ingest_config.timescale_settings.url)
        snapshot = asyncio.run(
            _execute_failover_drill(
                ingest_config,
                kafka_mapping,
                results,
                dimensions=list(args.dimensions) if args.dimensions else None,
                scenario=args.scenario,
            )
        )
        report["failover_drill"] = snapshot

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        report = _build_report(args)
        if args.format == "markdown":
            output = _render_markdown(report)
        else:
            output = json.dumps(report, indent=2, sort_keys=True)
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())

