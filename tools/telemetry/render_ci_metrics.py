"""Render CI telemetry metrics into human-readable Markdown snapshots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ci_metrics import DEFAULT_METRICS_PATH, load_metrics


@dataclass(frozen=True)
class _TrendSection:
    """Describes a rendered Markdown section."""

    heading: str
    lines: list[str]


def _format_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> list[str]:
    header_list = list(headers)
    header_row = " | ".join(header_list)
    separator = " | ".join(["---"] * len(header_list))
    table_lines = [f"| {header_row} |", f"| {separator} |"]
    for row in rows:
        cells = list(row)
        table_lines.append(f"| {' | '.join(cells)} |")
    return table_lines


def _format_percentage(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    return f"{numeric:.2f}%"


def _coverage_section(coverage_trend: Iterable[Mapping[str, Any]], *, limit: int) -> _TrendSection:
    entries = list(coverage_trend)[-limit:]
    if not entries:
        return _TrendSection("Coverage trend", ["_No coverage entries recorded._"])

    rows: list[list[str]] = []
    for entry in entries:
        rows.append(
            [
                str(entry.get("label", "")),
                _format_percentage(entry.get("coverage_percent")),
                str(entry.get("source", "")),
            ]
        )
    lines = _format_table(["Label", "Coverage", "Source"], rows)
    return _TrendSection("Coverage trend", lines)


def _domains_section(domain_trend: Iterable[Mapping[str, Any]], *, limit: int) -> _TrendSection:
    entries = list(domain_trend)[-limit:]
    if not entries:
        return _TrendSection("Coverage domain snapshots", ["_No per-domain coverage entries recorded._"])

    sections: list[str] = []
    for entry in entries:
        label = str(entry.get("label", ""))
        sections.append(f"**{label or 'Domain snapshot'}**")
        threshold = entry.get("threshold")
        if threshold is not None:
            sections.append(f"- Threshold: {float(threshold):.2f}%")
        lagging = {str(name) for name in entry.get("lagging_domains", [])}
        domains = entry.get("domains", [])
        if not domains:
            sections.append("_No domain breakdown available._")
            continue
        rows: list[list[str]] = []
        for domain in domains:
            name = str(domain.get("name", ""))
            coverage = _format_percentage(domain.get("coverage_percent"))
            status = "Lagging" if name in lagging else "OK"
            rows.append([name, coverage, status])
        sections.extend(_format_table(["Domain", "Coverage", "Status"], rows))
        sections.append("")
    if sections and sections[-1] == "":
        sections.pop()
    return _TrendSection("Coverage domain snapshots", sections)


def _formatter_section(formatter_trend: Iterable[Mapping[str, Any]], *, limit: int) -> _TrendSection:
    entries = list(formatter_trend)[-limit:]
    if not entries:
        return _TrendSection("Formatter adoption", ["_No formatter telemetry recorded._"])

    rows: list[list[str]] = []
    for entry in entries:
        rows.append(
            [
                str(entry.get("label", "")),
                str(entry.get("mode", "")),
                str(entry.get("total_entries", "")),
            ]
        )
    lines = _format_table(["Label", "Mode", "Allowlist entries"], rows)
    return _TrendSection("Formatter adoption", lines)


def _remediation_section(remediation_trend: Iterable[Mapping[str, Any]], *, limit: int) -> _TrendSection:
    entries = list(remediation_trend)[-limit:]
    if not entries:
        return _TrendSection("Remediation progress", ["_No remediation entries recorded._"])

    sections: list[str] = []
    for entry in entries:
        label = str(entry.get("label", "")) or "Remediation entry"
        sections.append(f"**{label}**")
        statuses = entry.get("statuses", {})
        if statuses:
            for key, value in sorted(statuses.items()):
                sections.append(f"- {key}: {value}")
        if entry.get("source"):
            sections.append(f"- Source: {entry['source']}")
        if entry.get("note"):
            sections.append(f"- Note: {entry['note']}")
        sections.append("")
    if sections and sections[-1] == "":
        sections.pop()
    return _TrendSection("Remediation progress", sections)


def render_markdown(
    metrics: Mapping[str, Any],
    *,
    limit: int = 5,
) -> str:
    """Render the provided metrics mapping into Markdown text."""

    sections = [
        _coverage_section(metrics.get("coverage_trend", []), limit=limit),
        _domains_section(metrics.get("coverage_domain_trend", []), limit=limit),
        _formatter_section(metrics.get("formatter_trend", []), limit=limit),
        _remediation_section(metrics.get("remediation_trend", []), limit=limit),
    ]

    lines = ["# CI metrics snapshot", ""]
    for section in sections:
        lines.append(f"## {section.heading}")
        lines.extend(section.lines)
        lines.append("")

    if lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render CI telemetry metrics to Markdown for dashboards and runbooks.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the CI metrics JSON file (defaults to tests/.telemetry/ci_metrics.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the Markdown snapshot (defaults to stdout)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of recent entries to include from each trend (default: 5)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit must be a positive integer")

    metrics = load_metrics(args.metrics)
    markdown = render_markdown(metrics, limit=args.limit)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
