"""Aggregate pytest failure metadata into a simple Markdown report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class FailureStat:
    count: int
    last_seen: datetime | None
    last_message: str


def _load_entries(paths: Iterable[Path]) -> tuple[int, dict[str, FailureStat]]:
    stats: dict[str, FailureStat] = {}
    runs = 0

    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        runs += 1
        run_at = _parse_datetime(payload.get("run_at"))
        for failure in payload.get("failures", []):
            nodeid = str(failure.get("nodeid", "")).strip()
            if not nodeid:
                continue
            message = str(failure.get("message", "")).strip()

            existing = stats.get(nodeid)
            if existing is None:
                stats[nodeid] = FailureStat(count=1, last_seen=run_at, last_message=message)
            else:
                existing.count += 1
                if run_at and (existing.last_seen is None or run_at > existing.last_seen):
                    existing.last_seen = run_at
                    if message:
                        existing.last_message = message
                elif not existing.last_message and message:
                    existing.last_message = message

    return runs, stats


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _build_markdown(runs: int, stats: dict[str, FailureStat], max_rows: int) -> str:
    lines: list[str] = []
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines.append("# Pytest flake leaderboard")
    lines.append("")
    lines.append(f"Generated from {runs} run(s) on {generated_at}.")
    lines.append("")

    if not stats:
        lines.append("No failures recorded across the analyzed runs.")
        return "\n".join(lines) + "\n"

    total_failures = sum(entry.count for entry in stats.values())
    lines.append(
        f"Total distinct failing tests: {len(stats)}; total failure events: {total_failures}."
    )
    lines.append("")
    lines.append("| Test | Failures | Last seen (UTC) | Last message |")
    lines.append("| --- | --- | --- | --- |")

    sorted_entries = sorted(
        stats.items(),
        key=lambda item: (-item[1].count, item[0]),
    )
    for nodeid, stat in sorted_entries[:max_rows]:
        message = stat.last_message or "(no message captured)"
        message = message.replace("|", "\\|")
        lines.append(
            f"| `{nodeid}` | {stat.count} | {_format_timestamp(stat.last_seen)} | {message} |"
        )

    if len(sorted_entries) > max_rows:
        lines.append("")
        lines.append(f"_Showing top {max_rows} entries out of {len(sorted_entries)}._")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        required=True,
        help="Path to a pytest failure metadata JSON file (repeatable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the Markdown report.",
    )
    parser.add_argument(
        "--append-summary",
        type=Path,
        help="Append the Markdown report to the provided file (for CI step summaries).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Maximum number of entries to include in the table (default: 20).",
    )

    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    runs, stats = _load_entries(input_paths)
    markdown = _build_markdown(runs, stats, max_rows=args.max_rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")

    if args.append_summary:
        args.append_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.append_summary.open("a", encoding="utf-8") as handle:
            handle.write(markdown)
            if not markdown.endswith("\n"):
                handle.write("\n")

    if not args.output and not args.append_summary:
        print(markdown)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
