"""Export risk, execution, and compliance evidence for governance reviews.

This CLI packages the runtime telemetry highlighted in the institutional
risk/compliance alignment brief into a single JSON payload. It gathers:

* Risk policy and posture snapshots emitted by ``ProfessionalPredatorApp``.
* Execution readiness summaries and recent Timescale execution journal entries.
* Compliance readiness/workflow snapshots plus trade and KYC journal statistics.

Governance teams can invoke the exporter during reviews to capture the latest
telemetry without scraping Markdown summaries or running bespoke SQL queries.

Example usage::

    python -m tools.telemetry.export_risk_compliance_snapshots \
        --output governance.json --recent 2

The resulting JSON bundles the requested runtime summary sections alongside
Timescale journal counts so evidence packs stay aligned with the shipped code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.data_foundation.persist.timescale import (
    TimescaleComplianceJournal,
    TimescaleConnectionSettings,
    TimescaleExecutionJournal,
    TimescaleKycJournal,
)
from src.governance.system_config import SystemConfig


logger = logging.getLogger(__name__)


DEFAULT_SECTIONS: tuple[str, ...] = (
    "risk",
    "execution",
    "compliance_readiness",
    "compliance_workflows",
    "compliance",
    "kyc",
)

DEFAULT_RECENT = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export Professional Predator risk, execution, and compliance "
            "snapshots to JSON so governance reviews can bundle telemetry "
            "and Timescale journal evidence without manual queries."
        ),
    )
    parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        metavar="NAME",
        help=(
            "Snapshot key from ProfessionalPredatorApp.summary() to export. "
            "Can be supplied multiple times; defaults to the risk/execution/"
            "compliance bundle if omitted."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path that will receive the JSON payload.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for the emitted JSON payload (defaults to 2).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Exit with status 0 even when one or more requested sections are "
            "missing from the runtime summary."
        ),
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=DEFAULT_RECENT,
        help=(
            "Number of recent journal entries to include for each Timescale feed (defaults to 3)."
        ),
    )
    parser.add_argument(
        "--strategy-id",
        help=(
            "Optional strategy identifier used to filter compliance and "
            "execution journal statistics."
        ),
    )
    parser.add_argument(
        "--execution-service",
        dest="execution_service",
        help="Optional execution service name used when filtering journals.",
    )
    return parser


def _serialise_entry(entry: object) -> object:
    if hasattr(entry, "as_dict"):
        return entry.as_dict()  # type: ignore[call-arg]
    if isinstance(entry, Mapping):
        return dict(entry)
    return entry


def _summarise_journals(
    config: SystemConfig,
    *,
    strategy_id: str | None,
    execution_service: str | None,
    recent: int,
) -> dict[str, object]:
    settings = TimescaleConnectionSettings.from_mapping(config.extras)
    recent_limit = max(0, int(recent))

    metadata: dict[str, object] = {
        "configured": settings.configured,
        "dialect": "postgresql" if settings.is_postgres() else "sqlite",
        "recent_limit": recent_limit,
    }
    if not settings.configured:
        metadata["note"] = (
            "Timescale fallback (SQLite) in use; journal counts reflect the local store."
        )

    errors: list[str] = []

    def _collect(
        factory,  # type: ignore[no-untyped-def]
        *,
        summary_kwargs: Mapping[str, object],
        fetch_kwargs: Mapping[str, object],
    ) -> dict[str, object]:
        try:
            journal = factory()
            stats = journal.summarise(**summary_kwargs)
            recent_entries: list[object] = []
            if recent_limit:
                entries = journal.fetch_recent(limit=recent_limit, **fetch_kwargs)
                recent_entries = [_serialise_entry(entry) for entry in entries]
            return {"stats": stats, "recent": recent_entries}
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to summarise journal", exc_info=True)
            errors.append(str(exc))
            return {"error": str(exc)}
        finally:
            try:
                journal.close()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to dispose Timescale engine", exc_info=True)

    compliance_summary = _collect(
        lambda: TimescaleComplianceJournal(settings.create_engine()),
        summary_kwargs={"strategy_id": strategy_id},
        fetch_kwargs={"strategy_id": strategy_id} if strategy_id else {},
    )

    kyc_summary = _collect(
        lambda: TimescaleKycJournal(settings.create_engine()),
        summary_kwargs={"strategy_id": strategy_id},
        fetch_kwargs={"strategy_id": strategy_id} if strategy_id else {},
    )

    execution_kwargs: dict[str, object] = {}
    if execution_service:
        execution_kwargs["service"] = execution_service
    if strategy_id:
        execution_kwargs["strategy_id"] = strategy_id

    execution_summary = _collect(
        lambda: TimescaleExecutionJournal(settings.create_engine()),
        summary_kwargs=execution_kwargs,
        fetch_kwargs=execution_kwargs,
    )

    payload: dict[str, object] = {
        "metadata": metadata,
        "compliance": compliance_summary,
        "kyc": kyc_summary,
        "execution": execution_summary,
    }
    if errors:
        payload["errors"] = errors
    return payload


async def _collect_snapshots(
    sections: Iterable[str],
    *,
    recent: int,
    strategy_id: str | None,
    execution_service: str | None,
) -> tuple[dict[str, object], list[str]]:
    config = SystemConfig.from_env()
    builder = _load_builder()
    app = await builder(config=config)
    try:
        async with app:
            summary: Mapping[str, object] = app.summary()
    finally:
        await app.shutdown()

    requested = list(dict.fromkeys(sections))
    snapshots: dict[str, object] = {}
    missing: list[str] = []

    for section in requested:
        data = summary.get(section)
        if data is None:
            missing.append(section)
        else:
            snapshots[section] = data

    journals = _summarise_journals(
        config,
        strategy_id=strategy_id,
        execution_service=execution_service,
        recent=recent,
    )

    payload: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "config": {
            "tier": config.tier.value,
            "backbone_mode": config.data_backbone_mode.value,
        },
        "sections_requested": requested,
        "snapshots": snapshots,
        "missing_sections": missing,
        "journal_summary": journals,
    }

    return payload, missing


def _load_builder():  # pragma: no cover - exercised via monkeypatch in tests
    from src.runtime.predator_app import build_professional_predator_app

    return build_professional_predator_app


def _emit_payload(text: str, output: Path | None) -> None:
    if output is None:
        print(text)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{text}\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    sections: Sequence[str] = args.sections if args.sections is not None else DEFAULT_SECTIONS

    try:
        payload, missing = asyncio.run(
            _collect_snapshots(
                sections,
                recent=args.recent,
                strategy_id=args.strategy_id,
                execution_service=args.execution_service,
            )
        )
    except Exception:  # pragma: no cover - surfaced as CLI failure with logging
        logger.exception("Failed to collect risk/compliance snapshots")
        return 1

    text = json.dumps(payload, indent=args.indent, sort_keys=True)
    _emit_payload(text, args.output)

    if missing and not args.allow_missing:
        print(
            f"warning: missing sections: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
