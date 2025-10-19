from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    UTC = timezone.utc
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
        "alert_response_trend": [],
    }


def _ensure_defaults(data: dict[str, Any]) -> dict[str, Any]:
    data.setdefault("coverage_trend", [])
    data.setdefault("formatter_trend", [])
    data.setdefault("coverage_domain_trend", [])
    data.setdefault("remediation_trend", [])
    data.setdefault("alert_response_trend", [])
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
        "alert_response_trend": list(data.get("alert_response_trend", [])),
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


def _now() -> datetime:
    """Return the current UTC timestamp.

    Centralising the time source makes it easier to monkeypatch in tests so we
    can exercise staleness calculations deterministically without depending on
    real wall-clock time.
    """

    return datetime.now(tz=UTC)


def _timestamp_label() -> str:
    return _now().isoformat(timespec="seconds")


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


def _format_lagging_note(
    entry: Mapping[str, Any], *, default: str | None = None
) -> str | None:
    lagging = entry.get("lagging_domains")
    domains = entry.get("domains")
    if not lagging:
        return default

    lagging_set = {str(name) for name in lagging}
    domain_lookup: dict[str, float] = {}
    if isinstance(domains, Iterable):
        for raw_domain in domains:
            if not isinstance(raw_domain, Mapping):
                continue
            name = str(raw_domain.get("name"))
            if name in lagging_set:
                try:
                    percent = float(raw_domain.get("coverage_percent", 0))
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    percent = 0.0
                domain_lookup[name] = percent

    if not domain_lookup:
        return default

    rendered = ", ".join(
        f"{name} ({percent:.2f}%)" for name, percent in sorted(domain_lookup.items())
    )
    return f"Lagging domains: {rendered}"


def record_coverage_domains(
    metrics_path: Path,
    coverage_report: Path,
    *,
    label: str | None = None,
    threshold: float | None = 80.0,
    record_remediation_entry: bool = False,
    remediation_label: str | None = None,
    remediation_note: str | None = None,
    remediation_source: str | None = None,
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
        entry["lagging_count"] = len(entry["lagging_domains"])

    if matrix.domains:
        worst_domain = min(matrix.domains, key=lambda domain: domain.percent)
        entry["worst_domain"] = worst_domain.as_dict()
    else:
        entry["worst_domain"] = None

    metrics["coverage_domain_trend"].append(entry)
    save_metrics(metrics_path, metrics)

    if record_remediation_entry:
        if remediation_source is None:
            remediation_source = str(coverage_report)

        statuses: dict[str, Any] = {
            "overall_coverage": round(matrix.totals.percent, 2),
            "lagging_count": len(entry.get("lagging_domains", [])),
        }

        if threshold is not None:
            statuses["coverage_threshold"] = float(threshold)

        worst_domain = entry.get("worst_domain")
        if isinstance(worst_domain, Mapping) and worst_domain.get("name") is not None:
            statuses["worst_domain"] = str(worst_domain.get("name"))
            try:
                statuses["worst_domain_coverage"] = float(
                    worst_domain.get("coverage_percent", 0.0)
                )
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass

        note = remediation_note or _format_lagging_note(entry)

        metrics = record_remediation(
            metrics_path,
            statuses=statuses,
            label=remediation_label or label or _timestamp_label(),
            source=remediation_source,
            note=note,
            data=metrics,
        )

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


def _normalise_event_type(value: object) -> str:
    return str(value or "").strip().lower()


def _format_duration(seconds: float | None) -> tuple[str | None, str | None, str | None]:
    if seconds is None:
        return None, None, None
    total_seconds = max(0.0, float(seconds))
    rounded_seconds = int(round(total_seconds))
    minutes = total_seconds / 60.0
    human = str(timedelta(seconds=rounded_seconds))
    return str(rounded_seconds), f"{minutes:.2f}", human


def summarise_alert_timeline(timeline: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise a CI alert timeline for MTTA/MTTR tracking."""

    if not isinstance(timeline, Mapping):
        raise TypeError("Alert timeline payload must be mapping-like")

    raw_events = timeline.get("events", [])
    events: list[tuple[datetime, Mapping[str, Any]]] = []
    if isinstance(raw_events, Iterable):
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                continue
            timestamp = _parse_timestamp(raw_event.get("timestamp"))
            if timestamp is None:
                continue
            events.append((timestamp, raw_event))
    events.sort(key=lambda entry: entry[0])

    def pick_event(
        candidates: Iterable[str],
        *,
        timestamp_field: str | None = None,
        channel_field: str | None = None,
        actor_field: str | None = None,
        evidence_field: str | None = None,
    ) -> tuple[datetime | None, Mapping[str, Any] | None]:
        candidate_set = {_normalise_event_type(value) for value in candidates}
        for event_timestamp, event in events:
            event_type = _normalise_event_type(event.get("type"))
            if event_type in candidate_set:
                return event_timestamp, event

        if timestamp_field is not None:
            fallback_timestamp = _parse_timestamp(timeline.get(timestamp_field))
            if fallback_timestamp is not None:
                fallback_event: dict[str, Any] = {"type": timestamp_field}
                if channel_field is not None:
                    fallback_event["channel"] = timeline.get(channel_field)
                if actor_field is not None:
                    fallback_event["actor"] = timeline.get(actor_field)
                if evidence_field is not None:
                    fallback_event["evidence"] = timeline.get(evidence_field)
                return fallback_timestamp, fallback_event

        return None, None

    opened_ts, opened_event = pick_event(
        {
            "alert_opened",
            "opened",
            "failure_detected",
            "issue_opened",
            "failure_logged",
        },
        timestamp_field="opened_at",
        channel_field="opened_channel",
        actor_field="opened_by",
    )

    ack_ts, ack_event = pick_event(
        {
            "alert_acknowledged",
            "acknowledged",
            "ack",
            "acknowledge",
            "slack_ack",
        },
        timestamp_field="acknowledged_at",
        channel_field="ack_channel",
        actor_field="ack_actor",
        evidence_field="ack_channel_evidence",
    )

    resolved_ts, resolved_event = pick_event(
        {
            "alert_resolved",
            "resolved",
            "recovered",
            "closed",
            "issue_closed",
        },
        timestamp_field="resolved_at",
        channel_field="resolve_channel",
        actor_field="resolve_actor",
        evidence_field="resolve_channel_evidence",
    )

    mtta_seconds: float | None = None
    if opened_ts is not None and ack_ts is not None:
        mtta_seconds = (ack_ts - opened_ts).total_seconds()

    mttr_seconds: float | None = None
    if opened_ts is not None and resolved_ts is not None:
        mttr_seconds = (resolved_ts - opened_ts).total_seconds()

    incident_id = timeline.get("incident_id") or timeline.get("id")
    incident_label = timeline.get("label") or incident_id
    generated_at = _parse_timestamp(timeline.get("generated_at"))

    def _event_evidence(event: Mapping[str, Any] | None) -> str | None:
        if not isinstance(event, Mapping):
            return None
        for key in ("evidence", "evidence_url", "link"):
            value = event.get(key)
            if value not in (None, ""):
                return str(value)
        return None

    ack_channel = None
    ack_actor = None
    ack_channel_evidence = None
    if isinstance(ack_event, Mapping):
        ack_channel = ack_event.get("channel")
        ack_actor = ack_event.get("actor")
        ack_channel_evidence = _event_evidence(ack_event)

    resolve_channel = None
    resolve_actor = None
    resolve_channel_evidence = None
    if isinstance(resolved_event, Mapping):
        resolve_channel = resolved_event.get("channel")
        resolve_actor = resolved_event.get("actor")
        resolve_channel_evidence = _event_evidence(resolved_event)

    note_parts: list[str] = []
    mtta_seconds_str, mtta_minutes_str, mtta_human = _format_duration(mtta_seconds)
    mttr_seconds_str, mttr_minutes_str, mttr_human = _format_duration(mttr_seconds)

    if mtta_human is not None:
        channel = str(ack_channel or "unknown")
        note_parts.append(f"Ack via {channel} in {mtta_human}")
    if mttr_human is not None:
        channel = str(resolve_channel or "unknown")
        note_parts.append(f"Resolve via {channel} in {mttr_human}")

    default_note = "; ".join(note_parts) if note_parts else None
    note = str(timeline.get("note")) if timeline.get("note") else default_note

    summary_generated_at = (
        generated_at
        or opened_ts
        or ack_ts
        or resolved_ts
        or _now()
    )

    return {
        "incident_id": str(incident_id) if incident_id is not None else None,
        "label": str(incident_label) if incident_label is not None else None,
        "drill": bool(timeline.get("drill", False)),
        "opened_at": opened_ts.isoformat(timespec="seconds") if opened_ts else None,
        "acknowledged_at": ack_ts.isoformat(timespec="seconds") if ack_ts else None,
        "resolved_at": resolved_ts.isoformat(timespec="seconds") if resolved_ts else None,
        "ack_channel": str(ack_channel) if ack_channel not in (None, "") else None,
        "ack_actor": str(ack_actor) if ack_actor not in (None, "") else None,
        "ack_channel_evidence": str(ack_channel_evidence)
        if ack_channel_evidence not in (None, "")
        else None,
        "resolve_channel": str(resolve_channel)
        if resolve_channel not in (None, "")
        else None,
        "resolve_actor": str(resolve_actor)
        if resolve_actor not in (None, "")
        else None,
        "resolve_channel_evidence": str(resolve_channel_evidence)
        if resolve_channel_evidence not in (None, "")
        else None,
        "mtta_seconds": int(round(mtta_seconds)) if mtta_seconds is not None else None,
        "mtta_minutes": float(mtta_minutes_str) if mtta_minutes_str is not None else None,
        "mtta_readable": mtta_human,
        "mttr_seconds": int(round(mttr_seconds)) if mttr_seconds is not None else None,
        "mttr_minutes": float(mttr_minutes_str) if mttr_minutes_str is not None else None,
        "mttr_readable": mttr_human,
        "generated_at": summary_generated_at.isoformat(timespec="seconds"),
        "note": note,
        "source": timeline.get("source"),
    }


def record_alert_response(
    metrics_path: Path,
    *,
    timeline: Mapping[str, Any],
    label: str | None = None,
    source: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record MTTA/MTTR metrics from a CI alert timeline."""

    metrics = _ensure_defaults(data or load_metrics(metrics_path))
    summary = summarise_alert_timeline(timeline)

    entry_label = label or summary.get("label") or summary.get("incident_id")
    if entry_label is None:
        entry_label = _timestamp_label()

    def _string_or_na(value: object, *, fallback: str = "n/a") -> str:
        if value is None:
            return fallback
        return str(value)

    statuses = {
        "mtta_seconds": _string_or_na(summary.get("mtta_seconds")),
        "mtta_minutes": _string_or_na(summary.get("mtta_minutes")),
        "mtta_readable": _string_or_na(summary.get("mtta_readable")),
        "mttr_seconds": _string_or_na(summary.get("mttr_seconds")),
        "mttr_minutes": _string_or_na(summary.get("mttr_minutes")),
        "mttr_readable": _string_or_na(summary.get("mttr_readable")),
        "ack_channel": _string_or_na(summary.get("ack_channel"), fallback="unknown"),
        "ack_actor": _string_or_na(summary.get("ack_actor"), fallback="unknown"),
        "ack_channel_evidence": _string_or_na(
            summary.get("ack_channel_evidence"), fallback="unknown"
        ),
        "resolve_channel": _string_or_na(
            summary.get("resolve_channel"), fallback="unknown"
        ),
        "resolve_actor": _string_or_na(
            summary.get("resolve_actor"), fallback="unknown"
        ),
        "resolve_channel_evidence": _string_or_na(
            summary.get("resolve_channel_evidence"), fallback="unknown"
        ),
        "drill": "true" if summary.get("drill") else "false",
    }

    entry: dict[str, Any] = {
        "label": entry_label,
        "incident_id": summary.get("incident_id"),
        "drill": bool(summary.get("drill")),
        "opened_at": summary.get("opened_at"),
        "acknowledged_at": summary.get("acknowledged_at"),
        "resolved_at": summary.get("resolved_at"),
        "generated_at": summary.get("generated_at"),
        "statuses": statuses,
    }

    ack_channel_evidence_summary = summary.get("ack_channel_evidence")
    if ack_channel_evidence_summary not in (None, ""):
        entry["ack_channel_evidence"] = str(ack_channel_evidence_summary)

    resolve_channel_evidence_summary = summary.get("resolve_channel_evidence")
    if resolve_channel_evidence_summary not in (None, ""):
        entry["resolve_channel_evidence"] = str(resolve_channel_evidence_summary)

    entry_source = source or summary.get("source")
    if entry_source:
        entry["source"] = entry_source

    note = summary.get("note")
    if note:
        entry["note"] = str(note)

    metrics["alert_response_trend"].append(entry)
    save_metrics(metrics_path, metrics)
    return metrics


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    return parsed.astimezone(UTC)


def summarise_trend_staleness(
    metrics: Mapping[str, Any], *, max_age_hours: float = 24.0
) -> dict[str, Any]:
    """Summarise freshness for CI telemetry trends.

    The quality and observability roadmap leans heavily on telemetry snapshots
    to prove remediation progress. This helper inspects the existing metrics
    payload, calculates the age of the most recent entry per trend, and flags
    any surfaces that have gone stale beyond the configured threshold.
    """

    now = _now()
    threshold = float(max_age_hours)
    trends: dict[str, dict[str, Any]] = {}

    for key in (
        "coverage_trend",
        "coverage_domain_trend",
        "formatter_trend",
        "remediation_trend",
        "alert_response_trend",
    ):
        raw_entries = metrics.get(key, [])
        entries: list[Mapping[str, Any]] = (
            [entry for entry in raw_entries if isinstance(entry, Mapping)]
            if isinstance(raw_entries, Iterable)
            else []
        )

        last_timestamp: datetime | None = None
        for entry in entries:
            for field in ("generated_at", "label", "timestamp"):
                candidate = _parse_timestamp(entry.get(field))
                if candidate is None:
                    continue
                if last_timestamp is None or candidate > last_timestamp:
                    last_timestamp = candidate

        age_hours: float | None = None
        is_stale = True
        last_timestamp_str: str | None = None

        if last_timestamp is not None:
            age_hours = (now - last_timestamp).total_seconds() / 3600.0
            last_timestamp_str = last_timestamp.isoformat(timespec="seconds")
            is_stale = age_hours > threshold
        else:
            is_stale = True

        trends[key] = {
            "entry_count": len(entries),
            "last_timestamp": last_timestamp_str,
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
            "is_stale": is_stale,
        }

    return {
        "evaluated_at": now.isoformat(timespec="seconds"),
        "threshold_hours": threshold,
        "trends": trends,
    }


__all__ = [
    "DEFAULT_METRICS_PATH",
    "load_metrics",
    "record_coverage",
    "record_coverage_domains",
    "record_formatter",
    "record_remediation",
    "record_dashboard_remediation",
    "record_alert_response",
    "summarise_dashboard_payload",
    "summarise_alert_timeline",
    "summarise_trend_staleness",
    "parse_coverage_percentage",
    "save_metrics",
]
