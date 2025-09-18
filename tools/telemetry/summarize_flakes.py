"""Summarise pytest flake telemetry into a human-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_LOG = Path("tests/.telemetry/flake_runs.json")


def summarize_flake_log(path: Path) -> dict[str, Any]:
    """Return aggregate statistics for the given flake telemetry JSON file."""

    data = json.loads(path.read_text())
    events: list[dict[str, Any]] = list(data.get("events", []))
    failures = [event for event in events if event.get("outcome") != "passed"]

    history = [
        {"run_id": entry.get("run_id"), "conclusion": entry.get("conclusion")}
        for entry in data.get("history", [])
    ]

    return {
        "session_start": data.get("meta", {}).get("session_start"),
        "session_end": data.get("meta", {}).get("session_end"),
        "exit_status": data.get("meta", {}).get("exit_status"),
        "failure_count": len(failures),
        "failing_tests": [entry.get("nodeid") for entry in failures],
        "events_recorded": len(events),
        "history": history,
    }


def _format_history(history: Iterable[dict[str, Any]]) -> str:
    parts = []
    for row in history:
        run_id = row.get("run_id", "?")
        conclusion = row.get("conclusion", "unknown")
        parts.append(f"#{run_id}: {conclusion}")
    return ", ".join(parts) if parts else "<no runs recorded>"


def format_summary(summary: dict[str, Any]) -> str:
    """Return a human-readable summary for interactive use."""

    lines = [
        "Pytest flake telemetry",
        "----------------------",
        f"Session start: {summary.get('session_start')}",
        f"Session end: {summary.get('session_end')}",
        f"Exit status: {summary.get('exit_status')}",
        f"Events recorded: {summary.get('events_recorded')}",
        f"Failure count: {summary.get('failure_count')}",
    ]
    failing_tests = summary.get("failing_tests", [])
    if failing_tests:
        lines.append("Failing tests:")
        lines.extend(f"  - {test}" for test in failing_tests)
    else:
        lines.append("Failing tests: none")

    lines.append(f"Run history: {_format_history(summary.get('history', []))}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_LOG,
        help="Path to the flake telemetry JSON file",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    path = args.path
    if not path.exists():
        parser.error(f"Telemetry log not found: {path}")
    summary = summarize_flake_log(path)
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
