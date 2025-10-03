from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping
import xml.etree.ElementTree as ET

from .coverage_matrix import build_coverage_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence
    from src.operations.observability_dashboard import ObservabilityDashboard

DEFAULT_METRICS_PATH = Path("tests/.telemetry/ci_metrics.json")


def _empty_metrics() -> dict[str, Any]:
    return {
        "coverage_trend": [],
        "formatter_trend": [],
        "coverage_domain_trend": [],
        "remediation_trend": [],
    }


def _ensure_defaults(data: dict[str, Any]) -> dict[str, Any]:
    data.setdefault("coverage_trend", [])
    data.setdefault("formatter_trend", [])
    data.setdefault("coverage_domain_trend", [])
    data.setdefault("remediation_trend", [])
    return data


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_metrics()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Invalid metrics payload: expected a JSON object")
    return _ensure_defaults(payload)


def save_metrics(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "coverage_trend": list(data.get("coverage_trend", [])),
        "formatter_trend": list(data.get("formatter_trend", [])),
        "coverage_domain_trend": list(data.get("coverage_domain_trend", [])),
        "remediation_trend": list(data.get("remediation_trend", [])),
    }
    path.write_text(json.dumps(serialisable, indent=2, sort_keys=True) + "\n")


def _parse_line_rate(root: ET.Element) -> float | None:
    value = root.attrib.get("line-rate")
    if value is None:
        return None
    try:
        return float(value) * 100.0
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid line-rate value in coverage report") from exc


def _coverage_from_lines(root: ET.Element) -> float:
    total = 0
    covered = 0
    for line in root.iter("line"):
        total += 1
        hits = int(line.attrib.get("hits", "0"))
        if hits > 0:
            covered += 1
    if total == 0:
        raise ValueError("Coverage report does not contain any line entries")
    return (covered / total) * 100.0


def parse_coverage_percentage(coverage_report: Path) -> float:
    tree = ET.parse(coverage_report)
    root = tree.getroot()
    line_rate = _parse_line_rate(root)
    if line_rate is not None:
        return line_rate
    return _coverage_from_lines(root)


def _timestamp_label() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def record_coverage(
    metrics_path: Path,
    coverage_report: Path,
    *,
    label: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not coverage_report.exists():
        raise FileNotFoundError(f"Coverage report not found: {coverage_report}")
    metrics = _ensure_defaults(data or load_metrics(metrics_path))
    percent = parse_coverage_percentage(coverage_report)
    entry = {
        "label": label or _timestamp_label(),
        "coverage_percent": round(percent, 2),
        "source": str(coverage_report),
    }
    metrics["coverage_trend"].append(entry)
    save_metrics(metrics_path, metrics)
    return metrics


def record_coverage_domains(
    metrics_path: Path,
    coverage_report: Path,
    *,
    label: str | None = None,
    threshold: float | None = 80.0,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not coverage_report.exists():
        raise FileNotFoundError(f"Coverage report not found: {coverage_report}")

    metrics = _ensure_defaults(data or load_metrics(metrics_path))
    matrix = build_coverage_matrix(coverage_report)
    entry: dict[str, Any] = {
        "label": label or _timestamp_label(),
        "source": str(coverage_report),
        "generated_at": matrix.generated_at,
        "totals": matrix.totals.as_dict(),
        "domains": [domain.as_dict() for domain in matrix.domains],
    }

    if threshold is not None:
        entry["threshold"] = float(threshold)
        entry["lagging_domains"] = [
            domain["name"]
            for domain in entry["domains"]
            if float(domain["coverage_percent"]) < float(threshold)
        ]

    metrics["coverage_domain_trend"].append(entry)
    save_metrics(metrics_path, metrics)
    return metrics


def _parse_allowlist_entries(allowlist: Iterable[str]) -> list[str]:
    entries: list[str] = []
    for raw_line in allowlist:
        stripped = raw_line.split("#", 1)[0].strip()
        if stripped:
            entries.append(stripped)
    return entries


def _load_allowlist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Formatter allowlist not found: {path}")
    return _parse_allowlist_entries(path.read_text().splitlines())


def _formatter_metrics(entries: list[str]) -> dict[str, int]:
    directory_count = sum(1 for entry in entries if entry.endswith("/"))
    file_count = len(entries) - directory_count
    return {
        "total_entries": len(entries),
        "directory_count": directory_count,
        "file_count": file_count,
    }


def record_formatter(
    metrics_path: Path,
    allowlist_path: Path | None = None,
    *,
    mode: str | None = None,
    label: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _ensure_defaults(data or load_metrics(metrics_path))
    formatter_mode = mode or ("allowlist" if allowlist_path is not None else "global")
    trend_entry: dict[str, Any] = {
        "label": label or _timestamp_label(),
        "mode": formatter_mode,
    }
    if formatter_mode == "allowlist":
        if allowlist_path is None:
            raise ValueError("Allowlist path required when mode is 'allowlist'")
        entries = _load_allowlist(allowlist_path)
        trend_entry.update(_formatter_metrics(entries))
    else:
        trend_entry.update(
            {
                "total_entries": 0,
                "directory_count": 0,
                "file_count": 0,
            }
        )
    metrics["formatter_trend"].append(trend_entry)
    save_metrics(metrics_path, metrics)
    return metrics


def record_remediation(
    metrics_path: Path,
    *,
    statuses: Mapping[str, str],
    label: str | None = None,
    source: str | None = None,
    note: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not statuses:
        raise ValueError("At least one remediation status must be provided")

    metrics = _ensure_defaults(data or load_metrics(metrics_path))
    entry: dict[str, Any] = {
        "label": label or _timestamp_label(),
        "statuses": {key: str(value) for key, value in sorted(statuses.items())},
    }
    if source is not None:
        entry["source"] = source
    if note is not None:
        entry["note"] = note

    metrics["remediation_trend"].append(entry)
    save_metrics(metrics_path, metrics)
    return metrics


def summarise_dashboard_payload(
    dashboard: Mapping[str, Any] | "ObservabilityDashboard",
) -> dict[str, Any]:
    """Normalise an observability dashboard payload for remediation tracking."""

    if hasattr(dashboard, "remediation_summary"):
        summary = dashboard.remediation_summary()  # type: ignore[assignment]
        if not isinstance(summary, Mapping):
            raise TypeError("Dashboard remediation summary must be mapping-like")
        return {
            "generated_at": summary.get("generated_at"),
            "overall_status": summary.get("overall_status"),
            "panel_counts": dict(summary.get("panel_counts", {})),
            "failing_panels": tuple(summary.get("failing_panels", ())),
            "warning_panels": tuple(summary.get("warning_panels", ())),
            "healthy_panels": tuple(summary.get("healthy_panels", ())),
        }

    if not isinstance(dashboard, Mapping):
        raise TypeError("Dashboard payload must be mapping-like")

    status = str(dashboard.get("status", "unknown"))
    panels_obj = dashboard.get("panels", [])
    failing: list[str] = []
    warning: list[str] = []
    healthy: list[str] = []

    if isinstance(panels_obj, Iterable):
        for raw_panel in panels_obj:
            if not isinstance(raw_panel, Mapping):
                continue
            panel_name = str(raw_panel.get("name", "(unknown panel)"))
            panel_status = str(raw_panel.get("status", "unknown")).lower()
            if panel_status == "fail":
                failing.append(panel_name)
            elif panel_status == "warn":
                warning.append(panel_name)
            elif panel_status == "ok":
                healthy.append(panel_name)

    counts = {
        "ok": len(healthy),
        "warn": len(warning),
        "fail": len(failing),
    }

    generated_at = dashboard.get("generated_at")
    if not isinstance(generated_at, str):
        generated_at = _timestamp_label()

    return {
        "generated_at": generated_at,
        "overall_status": status,
        "panel_counts": counts,
        "failing_panels": tuple(failing),
        "warning_panels": tuple(warning),
        "healthy_panels": tuple(healthy),
    }


def record_dashboard_remediation(
    metrics_path: Path,
    *,
    summary: Mapping[str, Any] | "ObservabilityDashboard",
    label: str | None = None,
    source: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an observability dashboard snapshot as a remediation entry."""

    normalised = summarise_dashboard_payload(summary)

    panel_counts = normalised.get("panel_counts", {})
    statuses: dict[str, Any] = {
        "overall_status": normalised.get("overall_status", "unknown"),
        "panels_ok": panel_counts.get("ok", 0),
        "panels_warn": panel_counts.get("warn", 0),
        "panels_fail": panel_counts.get("fail", 0),
    }

    note_parts: list[str] = []
    failing = tuple(str(name) for name in normalised.get("failing_panels", ()))
    warnings = tuple(str(name) for name in normalised.get("warning_panels", ()))

    if failing:
        note_parts.append(f"Failing panels: {', '.join(failing)}")
    if warnings:
        note_parts.append(f"Warning panels: {', '.join(warnings)}")

    note = "; ".join(note_parts) if note_parts else None

    entry_label = label or str(normalised.get("generated_at", _timestamp_label()))

    return record_remediation(
        metrics_path,
        statuses=statuses,
        label=entry_label,
        source=source,
        note=note,
        data=data,
    )


__all__ = [
    "DEFAULT_METRICS_PATH",
    "load_metrics",
    "record_coverage",
    "record_coverage_domains",
    "record_formatter",
    "record_remediation",
    "record_dashboard_remediation",
    "summarise_dashboard_payload",
    "parse_coverage_percentage",
    "save_metrics",
]
