"""Helpers for rendering CI dashboard rows and weekly status digests from telemetry.

The quality and observability roadmap leans on telemetry captured in
``tests/.telemetry/ci_metrics.json`` plus optional observability dashboard
snapshots.  This module stitches those feeds into Markdown that can be pasted
straight into ``docs/status/ci_health.md`` or weekly status updates without
manual collation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .ci_metrics import (
    DEFAULT_METRICS_PATH,
    load_metrics,
    summarise_dashboard_payload,
    summarise_trend_staleness,
)


Row = tuple[str, str, str]


def _iter_mappings(raw: object) -> list[Mapping[str, Any]]:
    if not isinstance(raw, Iterable):
        return []
    entries: list[Mapping[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            entries.append(item)
    return entries


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
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
            return int(float(text))
        except ValueError:
            return None
    return None


def _clean_label(label: object) -> str:
    if label in (None, ""):
        return ""
    return str(label).strip()


def _format_value(primary: str, *extras: str) -> str:
    details = [entry for entry in extras if entry]
    if details:
        return f"{primary} ({', '.join(details)})"
    return primary


def _format_list(items: Iterable[str]) -> str:
    collected = [entry for entry in items if entry]
    if not collected:
        return "—"
    return "; ".join(collected)


def _coverage_snapshot(metrics: Mapping[str, Any]) -> Mapping[str, Any] | None:
    entries = _iter_mappings(metrics.get("coverage_trend"))
    if not entries:
        return None

    latest = entries[-1]
    previous = entries[-2] if len(entries) > 1 else None

    percent = _coerce_float(latest.get("coverage_percent"))
    previous_percent = (
        _coerce_float(previous.get("coverage_percent")) if previous is not None else None
    )

    delta = None
    if percent is not None and previous_percent is not None:
        delta = percent - previous_percent

    snapshot: MutableMapping[str, Any] = {
        "percent": percent,
        "delta": delta,
        "label": _clean_label(latest.get("label")),
        "source": _clean_label(latest.get("source")),
    }
    if previous is not None:
        snapshot["previous_label"] = _clean_label(previous.get("label"))
    return snapshot


def _domain_snapshot(metrics: Mapping[str, Any]) -> Mapping[str, Any] | None:
    entries = _iter_mappings(metrics.get("coverage_domain_trend"))
    if not entries:
        return None

    latest = entries[-1]
    domains_obj = latest.get("domains", [])
    domains: dict[str, float | None] = {}
    if isinstance(domains_obj, Iterable):
        for entry in domains_obj:
            if not isinstance(entry, Mapping):
                continue
            name = _clean_label(entry.get("name"))
            if not name:
                continue
            domains[name] = _coerce_float(entry.get("coverage_percent"))

    lagging_items: list[str] = []
    lagging_obj = latest.get("lagging_domains")
    if isinstance(lagging_obj, Iterable):
        for raw_name in lagging_obj:
            name = _clean_label(raw_name)
            if not name:
                continue
            percent = domains.get(name)
            if percent is None:
                lagging_items.append(name)
            else:
                lagging_items.append(f"{name} {percent:.2f}%")

    worst_text = ""
    worst = latest.get("worst_domain")
    if isinstance(worst, Mapping):
        worst_name = _clean_label(worst.get("name"))
        worst_percent = _coerce_float(worst.get("coverage_percent"))
        if worst_name:
            if worst_percent is None:
                worst_text = f"Worst: {worst_name}"
            else:
                worst_text = f"Worst: {worst_name} {worst_percent:.2f}%"

    snapshot: MutableMapping[str, Any] = {
        "label": _clean_label(latest.get("label")),
        "lagging_count": _coerce_int(latest.get("lagging_count")),
        "lagging": tuple(lagging_items),
        "threshold": _coerce_float(latest.get("threshold")),
        "worst": worst_text,
        "source": _clean_label(latest.get("source")),
    }
    return snapshot


def _formatter_snapshot(metrics: Mapping[str, Any]) -> Mapping[str, Any] | None:
    entries = _iter_mappings(metrics.get("formatter_trend"))
    if not entries:
        return None
    latest = entries[-1]
    snapshot: MutableMapping[str, Any] = {
        "label": _clean_label(latest.get("label")),
        "mode": _clean_label(latest.get("mode")) or "unknown",
        "total_entries": _coerce_int(latest.get("total_entries")) or 0,
        "directory_count": _coerce_int(latest.get("directory_count")) or 0,
        "file_count": _coerce_int(latest.get("file_count")) or 0,
    }
    return snapshot


def _remediation_snapshot(metrics: Mapping[str, Any]) -> Mapping[str, Any] | None:
    entries = _iter_mappings(metrics.get("remediation_trend"))
    if not entries:
        return None
    latest = entries[-1]
    statuses_obj = latest.get("statuses")
    statuses: list[tuple[str, str]] = []
    if isinstance(statuses_obj, Mapping):
        for key, value in sorted(statuses_obj.items(), key=lambda item: str(item[0])):
            statuses.append((_clean_label(key) or "status", str(value)))
    snapshot: MutableMapping[str, Any] = {
        "label": _clean_label(latest.get("label")),
        "statuses": tuple(statuses),
        "note": _clean_label(latest.get("note")),
        "source": _clean_label(latest.get("source")),
    }
    return snapshot


def _alert_snapshot(metrics: Mapping[str, Any]) -> Mapping[str, Any] | None:
    entries = _iter_mappings(metrics.get("alert_response_trend"))
    if not entries:
        return None

    latest = entries[-1]
    statuses_obj = latest.get("statuses")
    statuses: Mapping[str, Any] = statuses_obj if isinstance(statuses_obj, Mapping) else {}

    def _status_float(key: str) -> float | None:
        value = statuses.get(key)
        if value in (None, ""):
            value = latest.get(key)
        return _coerce_float(value)

    def _status_text(key: str) -> str:
        value = statuses.get(key)
        if value in (None, ""):
            value = latest.get(key)
        return _clean_label(value)

    snapshot: MutableMapping[str, Any] = {
        "label": _clean_label(latest.get("label")),
        "incident_id": _clean_label(latest.get("incident_id")),
        "drill": bool(latest.get("drill")),
        "opened_at": _clean_label(latest.get("opened_at")),
        "acknowledged_at": _clean_label(latest.get("acknowledged_at")),
        "resolved_at": _clean_label(latest.get("resolved_at")),
        "generated_at": _clean_label(latest.get("generated_at")),
        "ack_channel": _status_text("ack_channel"),
        "ack_actor": _status_text("ack_actor"),
        "resolve_channel": _status_text("resolve_channel"),
        "resolve_actor": _status_text("resolve_actor"),
        "mtta_minutes": _status_float("mtta_minutes"),
        "mtta_readable": _clean_label(statuses.get("mtta_readable")),
        "mttr_minutes": _status_float("mttr_minutes"),
        "mttr_readable": _clean_label(statuses.get("mttr_readable")),
        "note": _clean_label(latest.get("note")),
        "source": _clean_label(latest.get("source")),
    }

    return snapshot


def _dashboard_snapshot(dashboard: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if dashboard is None:
        return None
    summary = summarise_dashboard_payload(dashboard)
    failing = tuple(str(name) for name in summary.get("failing_panels", ()))
    warnings = tuple(str(name) for name in summary.get("warning_panels", ()))
    healthy = tuple(str(name) for name in summary.get("healthy_panels", ()))
    snapshot: MutableMapping[str, Any] = {
        "status": _clean_label(summary.get("overall_status")) or "unknown",
        "generated_at": _clean_label(summary.get("generated_at")),
        "panel_counts": summary.get("panel_counts", {}),
        "failing": failing,
        "warnings": warnings,
        "healthy": healthy,
    }
    return snapshot


def _freshness_snapshot(
    metrics: Mapping[str, Any], *, max_age_hours: float
) -> Mapping[str, Any]:
    summary = summarise_trend_staleness(metrics, max_age_hours=max_age_hours)
    trends = summary.get("trends", {})
    stale: list[str] = []
    fresh: list[str] = []
    if isinstance(trends, Mapping):
        for key, value in sorted(trends.items()):
            if not isinstance(value, Mapping):
                continue
            age = _coerce_float(value.get("age_hours"))
            label = key.replace("_", " ")
            segment = f"{label}"
            if age is not None:
                segment = f"{label} {age:.1f}h"
            if value.get("is_stale"):
                stale.append(segment)
            else:
                fresh.append(segment)
    return {
        "evaluated_at": _clean_label(summary.get("evaluated_at")),
        "threshold_hours": max_age_hours,
        "stale": tuple(stale),
        "fresh": tuple(fresh),
    }


def _render_rows(
    *,
    metrics: Mapping[str, Any],
    dashboard: Mapping[str, Any] | None,
    freshness_hours: float,
) -> list[Row]:
    rows: list[Row] = []

    coverage = _coverage_snapshot(metrics)
    if coverage is None:
        rows.append(("Coverage", "not recorded", "Run update_ci_metrics with --coverage-report"))
    else:
        percent = coverage.get("percent")
        label = coverage.get("label")
        delta = coverage.get("delta")
        extras: list[str] = []
        if label:
            extras.append(str(label))
        if delta is not None:
            extras.append(f"change {float(delta):+.2f}pp")
        value = _format_value(
            f"{float(percent):.2f}%" if percent is not None else "not recorded",
            *extras,
        )
        notes = []
        source = coverage.get("source")
        if source:
            notes.append(f"Source: {source}")
        previous = coverage.get("previous_label")
        if previous:
            notes.append(f"Previous label: {previous}")
        rows.append(("Coverage", value, _format_list(notes)))

    domains = _domain_snapshot(metrics)
    if domains is None:
        rows.append(("Coverage domains", "not recorded", "No domain coverage snapshots recorded"))
    else:
        lagging_count = domains.get("lagging_count")
        threshold = domains.get("threshold")
        label = domains.get("label")
        extras: list[str] = []
        if threshold is not None:
            extras.append(f"threshold {float(threshold):.1f}%")
        if label:
            extras.append(str(label))
        value = _format_value(
            f"{lagging_count} lagging" if lagging_count is not None else "no lagging",
            *extras,
        )
        notes = list(domains.get("lagging", ()))
        worst = domains.get("worst")
        if worst:
            notes.append(str(worst))
        source = domains.get("source")
        if source:
            notes.append(f"Source: {source}")
        rows.append(("Coverage domains", value, _format_list(notes)))

    formatter = _formatter_snapshot(metrics)
    if formatter is None:
        rows.append(("Formatter", "not recorded", "Record formatter mode via update_ci_metrics"))
    else:
        mode = formatter.get("mode", "unknown")
        label = formatter.get("label")
        value = _format_value(f"mode {mode}", str(label) if label else "")
        notes = []
        if mode == "allowlist":
            notes.append(
                "Allowlist entries: {total} (dirs {dirs}, files {files})".format(
                    total=formatter.get("total_entries", 0),
                    dirs=formatter.get("directory_count", 0),
                    files=formatter.get("file_count", 0),
                )
            )
        else:
            notes.append("Repo-wide formatting enforced")
        rows.append(("Formatter", value, _format_list(notes)))

    remediation = _remediation_snapshot(metrics)
    if remediation is None:
        rows.append(("Remediation", "not recorded", "Record remediation statuses via update_ci_metrics"))
    else:
        label = remediation.get("label")
        statuses = remediation.get("statuses", ())
        status_text = "; ".join(
            f"{key}={value}" for key, value in statuses if key or value
        ) or "no statuses"
        value = _format_value(status_text, str(label) if label else "")
        notes = []
        note = remediation.get("note")
        if note:
            notes.append(str(note))
        source = remediation.get("source")
        if source:
            notes.append(f"Source: {source}")
        rows.append(("Remediation", value, _format_list(notes)))

    alerts = _alert_snapshot(metrics)
    if alerts is None:
        rows.append(
            (
                "Alert response",
                "not recorded",
                "Capture drills via update_ci_metrics --alert-timeline",
            )
        )
    else:
        mtta = alerts.get("mtta_minutes")
        mttr = alerts.get("mttr_minutes")

        def _fmt_minutes(value: float | None, prefix: str) -> str:
            if value is None:
                return f"{prefix} n/a"
            return f"{prefix} {float(value):.2f}m"

        primary = " / ".join((_fmt_minutes(mtta, "MTTA"), _fmt_minutes(mttr, "MTTR")))

        extras: list[str] = []
        label = alerts.get("label") or alerts.get("incident_id")
        if label:
            text = str(label)
            if alerts.get("drill"):
                text += " drill"
            extras.append(text)

        channel_notes: list[str] = []
        ack_channel = alerts.get("ack_channel")
        if ack_channel and ack_channel != "unknown":
            channel_notes.append(f"ack via {ack_channel}")
        resolve_channel = alerts.get("resolve_channel")
        if resolve_channel and resolve_channel != "unknown":
            channel_notes.append(f"resolve via {resolve_channel}")
        if channel_notes:
            extras.append(", ".join(channel_notes))

        value = _format_value(primary, *extras)

        notes: list[str] = []
        opened_at = alerts.get("opened_at")
        if opened_at:
            notes.append(f"Opened {opened_at}")
        acknowledged_at = alerts.get("acknowledged_at")
        if acknowledged_at:
            ack_actor = alerts.get("ack_actor")
            entry = f"Acknowledged {acknowledged_at}"
            if ack_actor and ack_actor != "unknown":
                entry += f" by {ack_actor}"
            notes.append(entry)
        resolved_at = alerts.get("resolved_at")
        if resolved_at:
            resolve_actor = alerts.get("resolve_actor")
            entry = f"Resolved {resolved_at}"
            if resolve_actor and resolve_actor != "unknown":
                entry += f" by {resolve_actor}"
            notes.append(entry)

        mtta_readable = alerts.get("mtta_readable")
        if mtta_readable:
            notes.append(f"MTTA {mtta_readable}")
        mttr_readable = alerts.get("mttr_readable")
        if mttr_readable:
            notes.append(f"MTTR {mttr_readable}")

        alert_note = alerts.get("note")
        if alert_note:
            notes.append(str(alert_note))
        source = alerts.get("source")
        if source:
            notes.append(f"Source: {source}")

        rows.append(("Alert response", value, _format_list(notes)))

    freshness = _freshness_snapshot(metrics, max_age_hours=freshness_hours)
    stale = freshness.get("stale", ())
    fresh = freshness.get("fresh", ())
    value = "All telemetry fresh" if not stale else f"Stale: {', '.join(stale)}"
    notes = list(fresh)
    evaluated_at = freshness.get("evaluated_at")
    if evaluated_at:
        notes.append(f"Evaluated at {evaluated_at}")
    rows.append(("Telemetry freshness", value, _format_list(notes)))

    dashboard_snapshot = _dashboard_snapshot(dashboard)
    if dashboard_snapshot is not None:
        status = dashboard_snapshot.get("status", "unknown").upper()
        counts = dashboard_snapshot.get("panel_counts", {})
        value = (
            f"{status} (ok={counts.get('ok', 0)}, "
            f"warn={counts.get('warn', 0)}, fail={counts.get('fail', 0)})"
        )
        notes = []
        failing = dashboard_snapshot.get("failing", ())
        warnings = dashboard_snapshot.get("warnings", ())
        healthy = dashboard_snapshot.get("healthy", ())
        if failing:
            notes.append("Failing: " + ", ".join(str(name) for name in failing))
        if warnings:
            notes.append("Warnings: " + ", ".join(str(name) for name in warnings))
        if healthy:
            notes.append("Healthy: " + ", ".join(str(name) for name in healthy))
        generated = dashboard_snapshot.get("generated_at")
        if generated:
            notes.append(f"Snapshot: {generated}")
        rows.append(("Observability dashboard", value, _format_list(notes)))

    return rows


def render_ci_dashboard_table(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    *,
    dashboard: Mapping[str, Any] | None = None,
    freshness_hours: float = 24.0,
) -> str:
    """Return a Markdown table summarising CI telemetry."""

    metrics = load_metrics(metrics_path)
    rows = _render_rows(
        metrics=metrics,
        dashboard=dashboard,
        freshness_hours=freshness_hours,
    )

    lines = ["| Signal | Value | Notes |", "| --- | --- | --- |"]
    for signal, value, notes in rows:
        lines.append(f"| {signal} | {value} | {notes} |")
    lines.append("")
    return "\n".join(lines)


def _section(title: str, body_lines: Sequence[str]) -> list[str]:
    lines = [f"## {title}"]
    if body_lines:
        lines.extend(body_lines)
    else:
        lines.append("_No telemetry recorded._")
    lines.append("")
    return lines


def _coverage_section(snapshot: Mapping[str, Any] | None) -> list[str]:
    if snapshot is None:
        return ["_No coverage telemetry recorded._", ""]
    lines: list[str] = []
    percent = snapshot.get("percent")
    if percent is not None:
        lines.append(f"- Latest: {float(percent):.2f}%")
    else:
        lines.append("- Latest: not recorded")
    label = snapshot.get("label")
    if label:
        lines[-1] += f" ({label})"
    delta = snapshot.get("delta")
    if delta is not None:
        prev_label = snapshot.get("previous_label")
        if prev_label:
            lines.append(
                f"- Change vs previous ({prev_label}): {float(delta):+.2f}pp"
            )
        else:
            lines.append(f"- Change since last reading: {float(delta):+.2f}pp")
    source = snapshot.get("source")
    if source:
        lines.append(f"- Source: {source}")
    return lines


def _domains_section(snapshot: Mapping[str, Any] | None) -> list[str]:
    if snapshot is None:
        return ["_No coverage domain telemetry recorded._"]
    lines: list[str] = []
    lagging = snapshot.get("lagging", ())
    threshold = snapshot.get("threshold")
    lagging_count = snapshot.get("lagging_count")
    label = snapshot.get("label")
    header = "- Lagging domains"
    if threshold is not None:
        header += f" (threshold {float(threshold):.1f}%)"
    if label:
        header += f" from {label}"
    header += ": "
    header += ", ".join(lagging) if lagging else "none"
    lines.append(header)
    worst = snapshot.get("worst")
    if worst:
        lines.append(f"- {worst}")
    source = snapshot.get("source")
    if source:
        lines.append(f"- Source: {source}")
    if lagging_count is not None:
        lines.append(f"- Lagging count: {lagging_count}")
    return lines


def _formatter_section(snapshot: Mapping[str, Any] | None) -> list[str]:
    if snapshot is None:
        return ["_No formatter telemetry recorded._"]
    lines: list[str] = []
    mode = snapshot.get("mode", "unknown")
    label = snapshot.get("label")
    text = f"- Mode: {mode}"
    if label:
        text += f" ({label})"
    lines.append(text)
    if mode == "allowlist":
        lines.append(
            "- Allowlist coverage: {total} entries (dirs {dirs}, files {files})".format(
                total=snapshot.get("total_entries", 0),
                dirs=snapshot.get("directory_count", 0),
                files=snapshot.get("file_count", 0),
            )
        )
    else:
        lines.append("- Repo-wide formatting enforced")
    return lines


def _remediation_section(snapshot: Mapping[str, Any] | None) -> list[str]:
    if snapshot is None:
        return ["_No remediation telemetry recorded._"]
    lines: list[str] = []
    label = snapshot.get("label")
    if label:
        lines.append(f"- Label: {label}")
    statuses = snapshot.get("statuses", ())
    if statuses:
        lines.append("- Statuses:")
        for key, value in statuses:
            lines.append(f"  - {key}: {value}")
    note = snapshot.get("note")
    if note:
        lines.append(f"- Note: {note}")
    source = snapshot.get("source")
    if source:
        lines.append(f"- Evidence: {source}")
    return lines


def _freshness_section(snapshot: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    threshold = snapshot.get("threshold_hours")
    if threshold is not None:
        lines.append(f"- Threshold: {float(threshold):.0f}h")
    stale = snapshot.get("stale", ())
    if stale:
        lines.append("- Stale feeds:")
        for entry in stale:
            lines.append(f"  - {entry}")
    else:
        lines.append("- All feeds within freshness threshold")
    fresh = snapshot.get("fresh", ())
    if fresh:
        lines.append("- Fresh feeds:")
        for entry in fresh:
            lines.append(f"  - {entry}")
    evaluated = snapshot.get("evaluated_at")
    if evaluated:
        lines.append(f"- Evaluated at: {evaluated}")
    return lines


def _dashboard_section(snapshot: Mapping[str, Any] | None) -> list[str]:
    if snapshot is None:
        return ["_No observability dashboard snapshot supplied._"]
    lines: list[str] = []
    status = snapshot.get("status", "unknown").upper()
    generated_at = snapshot.get("generated_at")
    header = f"- Status: {status}"
    if generated_at:
        header += f" (snapshot {generated_at})"
    lines.append(header)
    failing = snapshot.get("failing", ())
    warnings = snapshot.get("warnings", ())
    healthy = snapshot.get("healthy", ())
    if failing:
        lines.append("- Failing panels:")
        for name in failing:
            lines.append(f"  - {name}")
    if warnings:
        lines.append("- Warning panels:")
        for name in warnings:
            lines.append(f"  - {name}")
    if healthy:
        lines.append("- Healthy panels:")
        for name in healthy:
            lines.append(f"  - {name}")
    return lines


def _alert_section(snapshot: Mapping[str, Any] | None) -> list[str]:
    if snapshot is None:
        return ["_No alert response telemetry recorded._"]

    lines: list[str] = []
    label = snapshot.get("label") or snapshot.get("incident_id")
    if label:
        text = f"- Label: {label}"
        if snapshot.get("drill"):
            text += " (drill)"
        lines.append(text)

    opened_at = snapshot.get("opened_at")
    if opened_at:
        lines.append(f"- Opened: {opened_at}")

    acknowledged_at = snapshot.get("acknowledged_at")
    if acknowledged_at:
        details: list[str] = []
        ack_channel = snapshot.get("ack_channel")
        if ack_channel and ack_channel != "unknown":
            details.append(f"via {ack_channel}")
        ack_actor = snapshot.get("ack_actor")
        if ack_actor and ack_actor != "unknown":
            details.append(f"by {ack_actor}")
        detail_suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- Acknowledged: {acknowledged_at}{detail_suffix}")

    resolved_at = snapshot.get("resolved_at")
    if resolved_at:
        details = []
        resolve_channel = snapshot.get("resolve_channel")
        if resolve_channel and resolve_channel != "unknown":
            details.append(f"via {resolve_channel}")
        resolve_actor = snapshot.get("resolve_actor")
        if resolve_actor and resolve_actor != "unknown":
            details.append(f"by {resolve_actor}")
        detail_suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- Resolved: {resolved_at}{detail_suffix}")

    mtta = snapshot.get("mtta_minutes")
    if mtta is not None:
        human = snapshot.get("mtta_readable")
        suffix = f" ({human})" if human else ""
        lines.append(f"- MTTA: {float(mtta):.2f} minutes{suffix}")

    mttr = snapshot.get("mttr_minutes")
    if mttr is not None:
        human = snapshot.get("mttr_readable")
        suffix = f" ({human})" if human else ""
        lines.append(f"- MTTR: {float(mttr):.2f} minutes{suffix}")

    note = snapshot.get("note")
    if note:
        lines.append(f"- Note: {note}")

    source = snapshot.get("source")
    if source:
        lines.append(f"- Evidence: {source}")

    return lines


def render_weekly_status_summary(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    *,
    dashboard: Mapping[str, Any] | None = None,
    freshness_hours: float = 168.0,
) -> str:
    """Render a Markdown weekly status summary from telemetry feeds."""

    metrics = load_metrics(metrics_path)
    coverage = _coverage_snapshot(metrics)
    domains = _domain_snapshot(metrics)
    formatter = _formatter_snapshot(metrics)
    remediation = _remediation_snapshot(metrics)
    alerts = _alert_snapshot(metrics)
    freshness = _freshness_snapshot(metrics, max_age_hours=freshness_hours)
    dashboard_snapshot = _dashboard_snapshot(dashboard)

    lines = ["# Weekly CI telemetry", ""]
    lines.extend(_section("Coverage", _coverage_section(coverage)))
    lines.extend(_section("Coverage domains", _domains_section(domains)))
    lines.extend(_section("Formatter", _formatter_section(formatter)))
    lines.extend(_section("Remediation", _remediation_section(remediation)))
    lines.extend(_section("Alert response", _alert_section(alerts)))
    lines.extend(_section("Telemetry freshness", _freshness_section(freshness)))
    lines.extend(_section("Observability dashboard", _dashboard_section(dashboard_snapshot)))
    return "\n".join(lines).rstrip() + "\n"


def _load_dashboard(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Dashboard snapshot not found: {path}")
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text or "null")
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("Dashboard JSON must contain an object")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render CI dashboard table rows or weekly status digests from telemetry feeds.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to the CI metrics JSON file (defaults to tests/.telemetry/ci_metrics.json)",
    )
    parser.add_argument(
        "--dashboard",
        type=Path,
        help="Optional path to an observability dashboard JSON snapshot",
    )
    parser.add_argument(
        "--mode",
        choices=("ci-dashboard", "weekly-status"),
        default="ci-dashboard",
        help="Output format to render",
    )
    parser.add_argument(
        "--freshness-hours",
        type=float,
        default=None,
        help="Override the freshness window in hours (defaults to 24h for CI dashboard, 168h for weekly status)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to write the rendered Markdown",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dashboard_payload = _load_dashboard(args.dashboard)

    if args.mode == "ci-dashboard":
        freshness_hours = args.freshness_hours if args.freshness_hours is not None else 24.0
        rendered = render_ci_dashboard_table(
            args.metrics,
            dashboard=dashboard_payload,
            freshness_hours=freshness_hours,
        )
    else:
        freshness_hours = args.freshness_hours if args.freshness_hours is not None else 168.0
        rendered = render_weekly_status_summary(
            args.metrics,
            dashboard=dashboard_payload,
            freshness_hours=freshness_hours,
        )

    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
