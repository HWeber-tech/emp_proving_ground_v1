"""Render remediation progress snapshots into Markdown summaries.

This helper sits on top of the telemetry metrics tooling so status reports and
dashboards can surface a consistent summary of remediation progress without
hand-maintained spreadsheets.  It intentionally produces plain Markdown that
slots into existing status pages, letting CI automation or release managers
publish the latest quality/observability posture directly from the
`tests/.telemetry/ci_metrics.json` feed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .ci_metrics import DEFAULT_METRICS_PATH, load_metrics


def _normalise_entries(raw_entries: Iterable[object]) -> list[Mapping[str, object]]:
    """Return only the mapping-like remediation entries from ``raw_entries``."""

    entries: list[Mapping[str, object]] = []
    for entry in raw_entries:
        if isinstance(entry, Mapping):
            entries.append(entry)
    return entries


def _format_cell(value: object) -> str:
    if value in (None, ""):
        return "—"
    return str(value).replace("\n", " ")


def _status_rows(entry: Mapping[str, object]) -> list[tuple[str, str, str, str, str]]:
    label = _format_cell(entry.get("label", "(unlabelled)"))
    source = _format_cell(entry.get("source"))
    note = _format_cell(entry.get("note"))

    statuses_obj = entry.get("statuses")
    if not isinstance(statuses_obj, Mapping) or not statuses_obj:
        return [(label, "—", "—", source, note)]

    rows: list[tuple[str, str, str, str, str]] = []
    first = True
    for status_key, value in sorted(statuses_obj.items(), key=lambda item: str(item[0])):
        rows.append(
            (
                label if first else " ",  # Non-breaking space keeps column width
                _format_cell(status_key),
                _format_cell(value),
                source,
                note if first else " ",
            )
        )
        first = False
    return rows


def _render_table(entries: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Label | Status key | Value | Source | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in reversed(entries):  # Show most recent entries first
        for row in _status_rows(entry):
            lines.append("| " + " | ".join(row) + " |")
    return lines


def _latest_status_lines(entries: Sequence[Mapping[str, object]]) -> list[str]:
    latest = entries[-1]
    statuses_obj = latest.get("statuses")
    if not isinstance(statuses_obj, Mapping) or not statuses_obj:
        return []

    lines = ["## Latest status overview", ""]
    for key, value in sorted(statuses_obj.items(), key=lambda item: str(item[0])):
        lines.append(f"- **{_format_cell(key)}**: {_format_cell(value)}")

    note = latest.get("note")
    if note not in (None, ""):
        lines.append(f"- _Note_: {_format_cell(note)}")
    source = latest.get("source")
    if source not in (None, ""):
        lines.append(f"- _Evidence_: {_format_cell(source)}")
    return lines


def render_remediation_summary(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    *,
    limit: int | None = None,
) -> str:
    """Return a Markdown summary of remediation trend entries."""

    metrics = load_metrics(metrics_path)
    entries = _normalise_entries(metrics.get("remediation_trend", []))
    if limit is not None and limit >= 0:
        entries = entries[-limit:]

    lines = ["# Remediation progress", ""]
    if not entries:
        lines.append("_No remediation snapshots recorded._")
        lines.append("")
        return "\n".join(lines)

    lines.extend(_render_table(entries))
    latest_lines = _latest_status_lines(entries)
    if latest_lines:
        lines.append("")
        lines.extend(latest_lines)
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint rendering the remediation summary to stdout or a file."""

    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Render remediation trend entries from tests/.telemetry/ci_metrics.json "
            "into a Markdown summary."
        )
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the telemetry metrics JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of most recent entries to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the Markdown summary to (prints to stdout otherwise)",
    )

    args = parser.parse_args(argv)
    summary = render_remediation_summary(args.metrics_path, limit=args.limit)

    if args.output is None:
        sys.stdout.write(summary)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI invocation
    raise SystemExit(main())


__all__ = ["render_remediation_summary", "main"]
