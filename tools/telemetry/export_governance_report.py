"""Command line utility for generating governance compliance reports.

The roadmap highlights the need for a deterministic cadence that fuses
compliance readiness, regulatory telemetry, and Timescale audit evidence ahead
of live-broker pilots.  While the orchestration primitives already exist in
``src.operations.governance_reporting`` and ``governance_cadence``, reviewers
still needed a lightweight CLI to assemble an artefact on demand when the
runtime is not available.  This module fills that gap by composing the three
surfaces and emitting a JSON bundle (with optional Markdown) that mirrors the
cadence output.

The tool intentionally keeps its inputs flexible:

* Compliance and regulatory snapshots can be supplied via JSON files that were
  captured earlier in the day or piped in from other tooling.
* Audit evidence can either be provided directly or collected via
  ``collect_audit_evidence`` using the environment-backed ``SystemConfig``
  extras so institutional deployments keep using the Timescale journals.
* Metadata, report windows, and persistence behaviour are configurable so
  governance packs can match the cadence interval without hand-editing payloads.

Example usage::

    python -m tools.telemetry.export_governance_report \
        --compliance compliance.json \
        --regulatory regulatory.json \
        --audit audit.json \
        --metadata reviewer=ops --metadata runbook=kyc-weekly \
        --persist reports/governance.json

The command writes the latest report to stdout, optionally persists a rolling
history, and exits with status ``0`` on success.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.governance.system_config import SystemConfig
from src.operations.governance_reporting import (
    GovernanceReport,
    collect_audit_evidence,
    generate_governance_report,
    persist_governance_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a governance compliance report by fusing KYC/AML readiness, "
            "regulatory telemetry, and Timescale audit evidence."
        )
    )
    parser.add_argument(
        "--compliance",
        type=Path,
        help="Optional path to a JSON compliance readiness snapshot.",
    )
    parser.add_argument(
        "--regulatory",
        type=Path,
        help="Optional path to a JSON regulatory telemetry snapshot.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help=(
            "Optional path to JSON audit evidence.  When omitted the command will "
            "collect evidence via Timescale using SystemConfig extras."
        ),
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Treat audit evidence as unavailable instead of querying Timescale.",
    )
    parser.add_argument(
        "--strategy-id",
        help="Optional strategy identifier forwarded to the audit collector.",
    )
    parser.add_argument(
        "--period-start",
        help="ISO-8601 timestamp representing the reporting window start.",
    )
    parser.add_argument(
        "--period-end",
        help="ISO-8601 timestamp representing the reporting window end.",
    )
    parser.add_argument(
        "--generated-at",
        help="Override the report generation timestamp (ISO-8601).",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        metavar="KEY=VALUE",
        help="Additional metadata entries applied to the report payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path that will receive the JSON payload.",
    )
    parser.add_argument(
        "--persist",
        type=Path,
        help=(
            "Persist the report and rolling history to the provided JSON file using "
            "persist_governance_report()."
        ),
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=12,
        help="Number of historical entries to retain when persisting (default: 12).",
    )
    parser.add_argument(
        "--emit-markdown",
        action="store_true",
        help="Emit the Markdown representation alongside the JSON payload.",
    )
    return parser


def _load_json_payload(path: Path | None) -> Mapping[str, object] | None:
    if path is None:
        return None
    if str(path) == "-":
        raw = Path("-").read_text()  # pragma: no cover - defensive (stdin unsupported)
        return json.loads(raw)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    payload = json.loads(text or "null")
    if isinstance(payload, Mapping):
        return payload
    if payload is None:
        return None
    raise TypeError(f"Expected mapping JSON in {path}, received {type(payload).__name__}")


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


def _collect_audit_payload(
    *,
    audit_path: Path | None,
    skip: bool,
    strategy_id: str | None,
) -> Mapping[str, object] | None:
    if skip:
        return None
    if audit_path is not None:
        payload = _load_json_payload(audit_path)
        if payload is None:
            raise ValueError("Audit evidence file did not contain a payload")
        return payload

    config = SystemConfig.from_env()
    return collect_audit_evidence(config, strategy_id=strategy_id)


def _emit_json(report: GovernanceReport, *, output: Path | None) -> None:
    payload = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    if output is None:
        print(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")


def _emit_markdown(report: GovernanceReport) -> None:
    print("\n" + report.to_markdown())


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        compliance = _load_json_payload(args.compliance)
        regulatory = _load_json_payload(args.regulatory)
        audit = _collect_audit_payload(
            audit_path=args.audit,
            skip=args.skip_audit,
            strategy_id=args.strategy_id,
        )
        metadata = _parse_metadata(args.metadata)
        report = generate_governance_report(
            compliance_readiness=compliance,
            regulatory_snapshot=regulatory,
            audit_evidence=audit,
            period_start=_parse_timestamp(args.period_start),
            period_end=_parse_timestamp(args.period_end),
            generated_at=_parse_timestamp(args.generated_at),
            metadata=metadata,
        )
    except Exception as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error raises SystemExit

    _emit_json(report, output=args.output)
    if args.emit_markdown:
        _emit_markdown(report)

    if args.persist is not None:
        persist_governance_report(
            report,
            args.persist,
            history_limit=max(0, int(args.history_limit)),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

