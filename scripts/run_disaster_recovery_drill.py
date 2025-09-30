"""Run the institutional disaster recovery drill and emit a Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.operations.disaster_recovery import (
    DisasterRecoveryStatus,
    format_disaster_recovery_markdown,
    simulate_default_disaster_recovery,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate the Timescale disaster recovery drill",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/deployment/drills/disaster_recovery_drill.md"),
        help="Destination Markdown file for the drill summary.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSON artefact destination.",
    )
    parser.add_argument(
        "--scenario",
        default="timescale_failover",
        help="Scenario label recorded in the report metadata.",
    )
    parser.add_argument(
        "--fail-dimension",
        action="append",
        dest="fail_dimensions",
        default=["daily_bars"],
        help="Dimension(s) to target in the simulated outage.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    report = simulate_default_disaster_recovery(
        scenario=args.scenario,
        fail_dimensions=tuple(args.fail_dimensions),
    )

    markdown = format_disaster_recovery_markdown(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    if args.json_output:
        payload = report.as_dict()
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0 if report.status is DisasterRecoveryStatus.ready else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
