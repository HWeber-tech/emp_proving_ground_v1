"""Run the governance reporting cadence using configurable inputs.

This CLI wraps :class:`GovernanceCadenceRunner` so operators can execute the
KYC/AML governance cadence on demand.  It loads compliance and regulatory
snapshots from JSON files (or configuration-derived context), hydrates audit
evidence via Timescale when requested, enforces the reporting interval, and
persists the rolling history to disk.  When the cadence is not yet due the
command exits without updating the history, mirroring the runtime behaviour.

Example usage::

    python -m tools.governance.run_cadence \\
        --compliance compliance.json \\
        --regulatory regulatory.json \\
        --report-path reports/governance.json \\
        --interval 24h
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.governance.system_config import SystemConfig, SystemConfigLoadError
from src.operations.governance_cadence import GovernanceCadenceRunner
from src.operations.governance_reporting import (
    GovernanceReport,
    collect_audit_evidence,
    load_governance_context_from_config,
)


@dataclass(slots=True)
class _NoopEventBus:
    """Minimal event bus stub used by the CLI when not publishing."""

    def is_running(self) -> bool:  # pragma: no cover - trivial
        return False

    def publish_from_sync(self, _event: object) -> None:  # pragma: no cover - trivial
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the governance reporting cadence by fusing compliance, "
            "regulatory, and audit evidence snapshots."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a SystemConfig YAML file (defaults to environment variables).",
    )
    parser.add_argument(
        "--compliance",
        type=Path,
        help="Optional JSON compliance readiness snapshot (falls back to config context).",
    )
    parser.add_argument(
        "--regulatory",
        type=Path,
        help="Optional JSON regulatory telemetry snapshot (falls back to config context).",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help=(
            "Optional JSON audit evidence payload; when omitted the command collects "
            "Timescale evidence or uses the configured context file."
        ),
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip audit evidence collection entirely (section will be marked WARN).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/governance.json"),
        help="Location for the persisted governance history (default: reports/governance.json).",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=12,
        help="Number of historical entries to retain when persisting (default: 12).",
    )
    parser.add_argument(
        "--interval",
        default="24h",
        help="Cadence interval before a new report is due (e.g. 1h, 24h, 7d, 3600).",
    )
    parser.add_argument(
        "--generated-at",
        help="Override the generation timestamp (ISO-8601).",
    )
    parser.add_argument(
        "--strategy-id",
        help="Optional strategy identifier forwarded to the audit collector and metadata.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        metavar="KEY=VALUE",
        help="Additional metadata entries applied to the governance report.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Generate a report even if the cadence interval has not elapsed.",
    )
    parser.add_argument(
        "--emit-markdown",
        action="store_true",
        help="Emit the Markdown representation alongside the JSON payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to receive the JSON payload instead of stdout.",
    )
    return parser


def _load_json_payload(path: Path | None) -> Mapping[str, object] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8") or "null")
    if data is None:
        return None
    if isinstance(data, Mapping):
        return data
    raise TypeError(f"Expected mapping JSON in {path}, received {type(data).__name__}")


def _parse_metadata(entries: Iterable[str] | None) -> MutableMapping[str, object]:
    metadata: MutableMapping[str, object] = {}
    if not entries:
        return metadata
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Metadata entry must be KEY=VALUE, received: {entry!r}")
        key, value = entry.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text)
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


_INTERVAL_PATTERN = re.compile(r"(?P<value>\d+)(?P<unit>[smhd])", re.IGNORECASE)


def _parse_interval(value: str) -> timedelta:
    text = value.strip().lower()
    if not text:
        raise ValueError("Interval cannot be empty")
    if text.isdigit():
        seconds = int(text)
        if seconds <= 0:
            raise ValueError("Interval must be positive")
        return timedelta(seconds=seconds)

    total_seconds = 0
    for match in _INTERVAL_PATTERN.finditer(text):
        amount = int(match.group("value"))
        unit = match.group("unit")
        multiplier = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
        }[unit]
        total_seconds += amount * multiplier

    if total_seconds <= 0:
        raise ValueError(f"Invalid interval value: {value!r}")
    return timedelta(seconds=total_seconds)


def _emit_json(report: GovernanceReport, output: Path | None) -> None:
    payload = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    if output is None:
        print(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")


def _emit_markdown(report: GovernanceReport) -> None:
    print(report.to_markdown())


def _context_metadata(
    *,
    used_compliance_context: bool,
    used_regulatory_context: bool,
    used_audit_context: bool,
    context_dir: Mapping[str, Path | None],
) -> Mapping[str, str]:
    entries: dict[str, str] = {}
    if used_compliance_context and context_dir.get("compliance") is not None:
        entries["compliance"] = str(context_dir["compliance"])
    if used_regulatory_context and context_dir.get("regulatory") is not None:
        entries["regulatory"] = str(context_dir["regulatory"])
    if used_audit_context and context_dir.get("audit") is not None:
        entries["audit"] = str(context_dir["audit"])
    return entries


def _execute(args: argparse.Namespace) -> int:
    try:
        config = (
            SystemConfig.from_yaml(args.config)
            if args.config is not None
            else SystemConfig.from_env()
        )
    except SystemConfigLoadError as exc:
        raise RuntimeError(str(exc)) from exc

    base_path = args.config.parent if args.config is not None else Path.cwd()
    context_sources = load_governance_context_from_config(config, base_path=base_path)

    compliance_payload = _load_json_payload(args.compliance)
    used_compliance_context = False
    if compliance_payload is None:
        compliance_payload = context_sources.compliance
        used_compliance_context = compliance_payload is not None

    regulatory_payload = _load_json_payload(args.regulatory)
    used_regulatory_context = False
    if regulatory_payload is None:
        regulatory_payload = context_sources.regulatory
        used_regulatory_context = regulatory_payload is not None

    used_audit_context = False
    if args.audit is not None:
        audit_payload = _load_json_payload(args.audit)

        def audit_collector(_config: SystemConfig, _strategy_id: str | None) -> Mapping[str, object] | None:
            return audit_payload

    elif args.skip_audit:

        def audit_collector(_config: SystemConfig, _strategy_id: str | None) -> Mapping[str, object] | None:
            return None

    else:
        audit_payload = context_sources.audit
        used_audit_context = audit_payload is not None

        if audit_payload is not None:

            def audit_collector(_config: SystemConfig, _strategy_id: str | None) -> Mapping[str, object] | None:
                return audit_payload

        else:

            def audit_collector(
                config_obj: SystemConfig, strategy_id: str | None
            ) -> Mapping[str, object]:
                return collect_audit_evidence(config_obj, strategy_id=strategy_id)

    interval = _parse_interval(args.interval)
    generated_at = _parse_timestamp(args.generated_at)

    metadata_payload = _parse_metadata(args.metadata)
    metadata_payload.setdefault("cadence_runner", "tools.governance.run_cadence")
    if args.force:
        metadata_payload.setdefault("cadence_forced", True)
    if generated_at is not None:
        metadata_payload.setdefault("reference_timestamp", generated_at.isoformat())

    context_paths = {
        "compliance": context_sources.compliance_path,
        "regulatory": context_sources.regulatory_path,
        "audit": context_sources.audit_path,
    }
    context_meta = _context_metadata(
        used_compliance_context=used_compliance_context,
        used_regulatory_context=used_regulatory_context,
        used_audit_context=used_audit_context,
        context_dir=context_paths,
    )
    if context_meta:
        existing = metadata_payload.get("context_sources")
        merged: dict[str, str] = {}
        if isinstance(existing, Mapping):
            merged.update({str(k): str(v) for k, v in existing.items()})
        merged.update(context_meta)
        metadata_payload["context_sources"] = merged

    metadata_provider = (
        (lambda: dict(metadata_payload)) if metadata_payload else None
    )

    compliance_provider = lambda: compliance_payload
    regulatory_provider = lambda: regulatory_payload
    strategy_provider = (lambda: args.strategy_id) if args.strategy_id else None

    runner = GovernanceCadenceRunner(
        event_bus=_NoopEventBus(),
        config_provider=lambda: config,
        compliance_provider=compliance_provider,
        regulatory_provider=regulatory_provider,
        report_path=args.report_path,
        interval=interval,
        history_limit=max(0, int(args.history_limit)),
        strategy_id_provider=strategy_provider,
        metadata_provider=metadata_provider,
        audit_collector=audit_collector,
        publisher=lambda _bus, _report: None,
    )

    report = runner.run(reference=generated_at, force=args.force)
    if report is None:
        last = runner.last_generated_at
        if last is None:
            print("Governance cadence not due; no previous report found.")
        else:
            next_due = last.astimezone(UTC) + interval
            print(
                "Governance cadence not due; next scheduled at "
                f"{next_due.isoformat()}"
            )
        return 0

    _emit_json(report, args.output)
    if args.emit_markdown:
        print("")
        _emit_markdown(report)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _execute(args)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
