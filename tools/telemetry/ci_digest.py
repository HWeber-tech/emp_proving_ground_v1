"""Summaries for CI coverage telemetry and weekly status updates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ci_metrics import DEFAULT_METRICS_PATH, load_metrics


@dataclass(slots=True)
class CoverageSummary:
    label: str | None
    previous_label: str | None
    value: float | None
    delta: float | None
    source: str | None


@dataclass(slots=True)
class CoverageDomainSummary:
    label: str | None
    previous_label: str | None
    source: str | None
    lagging_count: int | None
    lagging_delta: int | None
    lagging_domains: tuple[str, ...]
    worst_domain: str | None
    worst_domain_percent: float | None


@dataclass(slots=True)
class RemediationSummary:
    label: str | None
    previous_label: str | None
    entries: tuple[tuple[str, str, float | None], ...]
    note: str | None
    source: str | None


def _normalise_mappings(raw_entries: object) -> list[Mapping[str, Any]]:
    if not isinstance(raw_entries, Sequence):
        return []
    entries: list[Mapping[str, Any]] = []
    for entry in raw_entries:
        if isinstance(entry, Mapping):
            entries.append(entry)
    return entries


def _coerce_float(value: object) -> float | None:
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            text = text[:-1]
        text = text.replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return None
    return None


def _format_delta(delta: float | None, *, suffix: str = "", precision: int = 2) -> str:
    if delta is None:
        return ""
    if abs(delta) < 1e-12:
        return ""
    formatted = f"{delta:+.{precision}f}"
    if suffix:
        formatted += suffix
    return f" (Δ {formatted})"


def summarise_coverage(metrics: Mapping[str, Any]) -> CoverageSummary:
    entries = _normalise_mappings(metrics.get("coverage_trend"))
    if not entries:
        return CoverageSummary(None, None, None, None, None)
    latest = entries[-1]
    previous_label: str | None = None
    delta: float | None = None
    value = _coerce_float(latest.get("coverage_percent"))
    if len(entries) > 1:
        previous = entries[-2]
        previous_label = str(previous.get("label")) if previous.get("label") is not None else None
        previous_value = _coerce_float(previous.get("coverage_percent"))
        if value is not None and previous_value is not None:
            delta = value - previous_value
    label = str(latest.get("label")) if latest.get("label") is not None else None
    source = str(latest.get("source")) if latest.get("source") is not None else None
    return CoverageSummary(label, previous_label, value, delta, source)


def summarise_coverage_domains(metrics: Mapping[str, Any]) -> CoverageDomainSummary:
    entries = _normalise_mappings(metrics.get("coverage_domain_trend"))
    if not entries:
        return CoverageDomainSummary(None, None, None, None, None, (), None, None)
    latest = entries[-1]
    previous_label: str | None = None
    lagging_delta: int | None = None
    lagging_count = _coerce_int(latest.get("lagging_count"))
    if len(entries) > 1:
        previous = entries[-2]
        previous_label = str(previous.get("label")) if previous.get("label") is not None else None
        previous_count = _coerce_int(previous.get("lagging_count"))
        if lagging_count is not None and previous_count is not None:
            lagging_delta = lagging_count - previous_count
    raw_domains = latest.get("lagging_domains")
    lagging_domains: tuple[str, ...] = ()
    if isinstance(raw_domains, Sequence):
        collected: list[str] = []
        for entry in raw_domains:
            collected.append(str(entry))
        lagging_domains = tuple(collected)
    worst = latest.get("worst_domain")
    worst_name: str | None = None
    worst_percent: float | None = None
    if isinstance(worst, Mapping):
        if worst.get("name") is not None:
            worst_name = str(worst.get("name"))
        worst_percent = _coerce_float(worst.get("coverage_percent"))
    label = str(latest.get("label")) if latest.get("label") is not None else None
    source = str(latest.get("source")) if latest.get("source") is not None else None
    return CoverageDomainSummary(
        label,
        previous_label,
        source,
        lagging_count,
        lagging_delta,
        lagging_domains,
        worst_name,
        worst_percent,
    )


def summarise_remediation(metrics: Mapping[str, Any]) -> RemediationSummary:
    entries = _normalise_mappings(metrics.get("remediation_trend"))
    if not entries:
        return RemediationSummary(None, None, tuple(), None, None)
    latest = entries[-1]
    previous_label: str | None = None
    previous_statuses: Mapping[str, Any] | None = None
    if len(entries) > 1:
        previous = entries[-2]
        previous_label = str(previous.get("label")) if previous.get("label") is not None else None
        raw_statuses = previous.get("statuses")
        if isinstance(raw_statuses, Mapping):
            previous_statuses = raw_statuses
    raw_latest_statuses = latest.get("statuses")
    summary_entries: list[tuple[str, str, float | None]] = []
    if isinstance(raw_latest_statuses, Mapping):
        for key in sorted(raw_latest_statuses):
            value_obj = raw_latest_statuses[key]
            delta_value: float | None = None
            if previous_statuses and key in previous_statuses:
                delta_value = None
                current_numeric = _coerce_float(value_obj)
                previous_numeric = _coerce_float(previous_statuses.get(key))
                if current_numeric is not None and previous_numeric is not None:
                    delta_value = current_numeric - previous_numeric
            summary_entries.append((str(key), str(value_obj), delta_value))
    label = str(latest.get("label")) if latest.get("label") is not None else None
    note = str(latest.get("note")) if latest.get("note") not in (None, "") else None
    source = str(latest.get("source")) if latest.get("source") is not None else None
    return RemediationSummary(
        label,
        previous_label,
        tuple(summary_entries),
        note,
        source,
    )


def render_dashboard_summary(metrics_path: Path = DEFAULT_METRICS_PATH) -> str:
    metrics = load_metrics(metrics_path)
    coverage = summarise_coverage(metrics)
    domains = summarise_coverage_domains(metrics)

    parts: list[str] = []
    if coverage.value is not None:
        coverage_text = f"{coverage.value:.2f}%"
        coverage_text += _format_delta(coverage.delta, suffix=" pts")
        if coverage.previous_label:
            coverage_text += f" vs {coverage.previous_label}"
        if coverage.source:
            coverage_text += f" — {coverage.source}"
        parts.append(f"Coverage: {coverage_text}")
    if domains.lagging_count is not None:
        lagging_text = f"{domains.lagging_count} lagging domain(s)"
        if domains.lagging_delta is not None and domains.lagging_delta != 0:
            lagging_text += f" (Δ {domains.lagging_delta:+d})"
        if domains.lagging_domains:
            lagging_text += f" — {', '.join(domains.lagging_domains)}"
        if domains.worst_domain:
            worst = domains.worst_domain
            if domains.worst_domain_percent is not None:
                worst += f" ({domains.worst_domain_percent:.2f}%)"
            lagging_text += f"; worst: {worst}"
        if domains.source:
            lagging_text += f" — {domains.source}"
        parts.append(lagging_text)
    if not parts:
        return "No coverage telemetry recorded yet."
    return "; ".join(parts)


def render_weekly_digest(metrics_path: Path = DEFAULT_METRICS_PATH) -> str:
    metrics = load_metrics(metrics_path)
    coverage = summarise_coverage(metrics)
    domains = summarise_coverage_domains(metrics)
    remediation = summarise_remediation(metrics)

    header = coverage.label or domains.label or remediation.label or "Latest snapshot"
    lines = [f"## {header}", ""]

    if coverage.value is None:
        lines.append("- Coverage: _No entries recorded._")
    else:
        coverage_line = f"- Coverage: {coverage.value:.2f}%"
        coverage_line += _format_delta(coverage.delta, suffix=" pts")
        if coverage.previous_label:
            coverage_line += f" vs {coverage.previous_label}"
        if coverage.source:
            coverage_line += f" (source: {coverage.source})"
        lines.append(coverage_line)

    if domains.lagging_count is None:
        lines.append("- Lagging domains: _No domain breakdown recorded._")
    else:
        domain_line = f"- Lagging domains: {domains.lagging_count}"
        if domains.lagging_delta is not None and domains.lagging_delta != 0:
            domain_line += f" (Δ {domains.lagging_delta:+d})"
        if domains.lagging_domains:
            domain_line += f" — {', '.join(domains.lagging_domains)}"
        if domains.worst_domain:
            worst = domains.worst_domain
            if domains.worst_domain_percent is not None:
                worst += f" ({domains.worst_domain_percent:.2f}%)"
            domain_line += f"; worst: {worst}"
        if domains.source:
            domain_line += f" (source: {domains.source})"
        lines.append(domain_line)

    if remediation.entries:
        lines.append("- Remediation statuses:")
        for key, value, delta in remediation.entries:
            entry_line = f"  - {key}: {value}"
            entry_line += _format_delta(delta)
            lines.append(entry_line)
    else:
        lines.append("- Remediation statuses: _No remediation snapshots recorded._")

    if remediation.note:
        lines.append(f"- Note: {remediation.note}")
    if remediation.source:
        lines.append(f"- Evidence: {remediation.source}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise CI telemetry for dashboards or weekly updates.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the CI metrics JSON file (defaults to tests/.telemetry/ci_metrics.json)",
    )
    parser.add_argument(
        "--mode",
        choices=("dashboard", "weekly"),
        default="weekly",
        help="Render the dashboard summary or weekly digest output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the rendered summary to",
    )
    args = parser.parse_args(argv)

    if args.mode == "dashboard":
        content = render_dashboard_summary(args.metrics)
    else:
        content = render_weekly_digest(args.metrics)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n")
    else:
        print(content)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = [
    "CoverageDomainSummary",
    "CoverageSummary",
    "RemediationSummary",
    "main",
    "render_dashboard_summary",
    "render_weekly_digest",
    "summarise_coverage",
    "summarise_coverage_domains",
    "summarise_remediation",
]
