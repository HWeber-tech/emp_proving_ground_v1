"""Export operational runtime telemetry snapshots for dashboard ingestion.

This CLI surfaces the high-signal operational blocks recorded by the
``ProfessionalPredatorApp`` so external dashboards (Grafana, DataDog, etc.) can
poll a single JSON payload rather than scraping Markdown summaries. The command
keeps context engineering front-and-centre by defaulting to the sections called
out in the operational readiness alignment brief: professional readiness,
security posture, incident response, and system validation snapshots.

Example usage::

    python -m tools.telemetry.export_operational_snapshots \
        --output telemetry.json

Downstream dashboards can then consume ``telemetry.json`` to render the latest
operational posture without bespoke event-bus subscribers.
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

from src.governance.system_config import SystemConfig


logger = logging.getLogger(__name__)

DEFAULT_SECTIONS: tuple[str, ...] = (
    "professional_readiness",
    "security",
    "incident_response",
    "system_validation",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export selected Professional Predator operational snapshots to JSON "
            "for observability dashboards."
        ),
    )
    parser.add_argument(
        "--section",
        dest="sections",
        action="append",
        metavar="NAME",
        help=(
            "Snapshot key from ProfessionalPredatorApp.summary() to export. "
            "Can be supplied multiple times; defaults to the operational "
            "readiness quartet if omitted."
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
    return parser


async def _collect_snapshots(
    sections: Iterable[str],
) -> tuple[dict[str, object], list[str]]:
    """Gather requested snapshot blocks from the professional runtime summary."""

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

    payload: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "config": {
            "tier": config.tier.value,
            "backbone_mode": config.data_backbone_mode.value,
        },
        "sections_requested": requested,
        "snapshots": snapshots,
        "missing_sections": missing,
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
        payload, missing = asyncio.run(_collect_snapshots(sections))
    except Exception:  # pragma: no cover - surfaced as CLI failure with logging
        logger.exception("Failed to collect operational snapshots")
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
