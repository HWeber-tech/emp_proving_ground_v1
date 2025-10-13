"""Utilities for turning final dry run evidence into backlog items."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.operations.dry_run_audit import DryRunStatus

__all__ = [
    "BacklogItem",
    "collect_backlog_items",
    "format_backlog_markdown",
    "items_to_json_serialisable",
]


@dataclass(frozen=True)
class BacklogItem:
    """Single follow-up captured from the final dry run evidence bundle."""

    severity: DryRunStatus
    category: str
    summary: str
    occurred_at: datetime | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "severity": self.severity.value,
            "category": self.category,
            "summary": self.summary,
        }
        if self.occurred_at is not None:
            payload["occurred_at"] = (
                self.occurred_at.astimezone(UTC).isoformat()
            )
        if self.context:
            payload["context"] = {
                str(key): value for key, value in self.context.items()
            }
        return payload


_SEVERITY_ORDER: Mapping[DryRunStatus, int] = {
    DryRunStatus.fail: 2,
    DryRunStatus.warn: 1,
    DryRunStatus.pass_: 0,
}


def collect_backlog_items(
    summary: Mapping[str, Any] | None,
    *,
    sign_off: Mapping[str, Any] | None = None,
    review: Mapping[str, Any] | None = None,
    include_pass: bool = False,
) -> tuple[BacklogItem, ...]:
    """Collate backlog items from dry run summary/sign-off/review payloads."""

    items: list[BacklogItem] = []
    if summary:
        _extract_log_items(summary.get("logs"), target=items)
        _extract_diary_items(summary.get("diary"), target=items)
        _extract_performance_items(summary.get("performance"), target=items)

    if sign_off:
        _extract_sign_off_items(sign_off, target=items)

    if review:
        _extract_review_items(review, target=items)

    deduped = _deduplicate(
        item
        for item in items
        if include_pass or item.severity is not DryRunStatus.pass_
    )
    ordered = sorted(
        deduped,
        key=lambda item: (
            -_SEVERITY_ORDER.get(item.severity, -1),
            item.occurred_at or datetime.min.replace(tzinfo=UTC),
            item.category,
            item.summary,
        ),
    )
    return tuple(ordered)


def format_backlog_markdown(
    items: Sequence[BacklogItem],
    *,
    title: str = "Final Dry Run Backlog",
) -> str:
    """Render backlog items as a Markdown list."""

    lines = [f"# {title}", ""]
    if not items:
        lines.append("No backlog items detected.")
        return "\n".join(lines) + "\n"

    for item in items:
        prefix = (
            f"- **{item.severity.value.upper()}**"
            f" [{item.category}]: {item.summary}"
        )
        lines.append(prefix)
        if item.occurred_at is not None:
            lines.append(
                f"  - occurred_at: {item.occurred_at.astimezone(UTC).isoformat()}"
            )
        for key, value in sorted(item.context.items()):
            lines.append(f"  - {key}: {value}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def items_to_json_serialisable(items: Sequence[BacklogItem]) -> list[Mapping[str, Any]]:
    """Convert backlog items into JSON serialisable dictionaries."""

    return [item.as_dict() for item in items]


def _extract_log_items(
    payload: Mapping[str, Any] | None,
    *,
    target: list[BacklogItem],
) -> None:
    if not isinstance(payload, Mapping):
        return

    for incident in payload.get("errors", []) or []:
        _append_incident(incident, "log_error", target)
    for incident in payload.get("warnings", []) or []:
        _append_incident(incident, "log_warning", target)
    for incident in payload.get("gap_incidents", []) or []:
        _append_incident(incident, "log_gap", target)
    for incident in payload.get("content_incidents", []) or []:
        _append_incident(incident, "log_content", target)


def _extract_diary_items(
    payload: Mapping[str, Any] | None,
    *,
    target: list[BacklogItem],
) -> None:
    if not isinstance(payload, Mapping):
        return

    for issue in payload.get("issues", []) or []:
        severity = _parse_status(issue.get("severity"))
        summary = _format_diary_summary(issue)
        occurred_at = _parse_datetime(issue.get("recorded_at"))
        context = {
            "entry_id": issue.get("entry_id"),
            "policy_id": issue.get("policy_id"),
            "reason": issue.get("reason"),
            **_normalise_mapping(issue.get("metadata")),
        }
        target.append(
            BacklogItem(
                severity=severity,
                category="diary",
                summary=summary,
                occurred_at=occurred_at,
                context={k: v for k, v in context.items() if v is not None},
            )
        )


def _extract_performance_items(
    payload: Mapping[str, Any] | None,
    *,
    target: list[BacklogItem],
) -> None:
    if not isinstance(payload, Mapping):
        return

    status = _parse_status(payload.get("status"))
    if status is DryRunStatus.pass_:
        return

    occurred_at = _parse_datetime(payload.get("generated_at"))
    summary = "Performance telemetry raised warnings."
    if status is DryRunStatus.fail:
        summary = "Performance telemetry failed sign-off criteria."
    context = {
        "total_trades": payload.get("total_trades"),
        "roi": payload.get("roi"),
        "win_rate": payload.get("win_rate"),
        "sharpe_ratio": payload.get("sharpe_ratio"),
        "metadata": _normalise_mapping(payload.get("metadata")),
    }
    target.append(
        BacklogItem(
            severity=status,
            category="performance",
            summary=summary,
            occurred_at=occurred_at,
            context={k: v for k, v in context.items() if v not in (None, {})},
        )
    )


def _extract_sign_off_items(
    payload: Mapping[str, Any],
    *,
    target: list[BacklogItem],
) -> None:
    findings = payload.get("findings", [])
    if not findings:
        return
    evaluated_at = _parse_datetime(payload.get("evaluated_at"))
    for finding in findings:
        severity = _parse_status(finding.get("severity"))
        summary = str(finding.get("message") or "Sign-off finding")
        context = _normalise_mapping(finding.get("metadata"))
        target.append(
            BacklogItem(
                severity=severity,
                category="sign_off",
                summary=summary,
                occurred_at=evaluated_at,
                context=context,
            )
        )


def _extract_review_items(
    payload: Mapping[str, Any],
    *,
    target: list[BacklogItem],
) -> None:
    for action in payload.get("action_items", []) or []:
        severity = _parse_status(action.get("severity"))
        summary = str(action.get("description") or "Action item")
        occurred_at = None
        context = _normalise_mapping(action.get("context"))
        target.append(
            BacklogItem(
                severity=severity,
                category=str(action.get("category") or "action"),
                summary=summary,
                occurred_at=occurred_at,
                context=context,
            )
        )

    for objective in payload.get("objectives", []) or []:
        severity = _parse_status(objective.get("status"))
        if severity is DryRunStatus.pass_:
            continue
        summary = f"Objective {objective.get('name')} status {severity.value.upper()}"
        note = objective.get("note")
        if note:
            summary += f": {note}"
        context = _normalise_mapping(objective.get("evidence"))
        target.append(
            BacklogItem(
                severity=severity,
                category="objective",
                summary=summary,
                occurred_at=None,
                context=context,
            )
        )


def _append_incident(
    data: Mapping[str, Any],
    category: str,
    target: list[BacklogItem],
) -> None:
    severity = _parse_status(data.get("severity"))
    summary = str(data.get("summary") or data.get("message") or "Log incident")
    occurred_at = _parse_datetime(data.get("occurred_at"))
    context = _normalise_mapping(data.get("metadata"))
    target.append(
        BacklogItem(
            severity=severity,
            category=category,
            summary=summary,
            occurred_at=occurred_at,
            context=context,
        )
    )


def _format_diary_summary(issue: Mapping[str, Any]) -> str:
    policy = issue.get("policy_id")
    reason = issue.get("reason")
    entry_id = issue.get("entry_id")
    parts = []
    if policy:
        parts.append(f"Policy {policy}")
    if entry_id:
        parts.append(f"Entry {entry_id}")
    description = " / ".join(parts) if parts else "Diary issue"
    if reason:
        description += f": {reason}"
    return description


def _normalise_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): val for key, val in value.items()}
    return {}


def _parse_status(value: Any) -> DryRunStatus:
    if isinstance(value, DryRunStatus):
        return value
    token = str(value or "").strip().lower()
    if not token:
        return DryRunStatus.warn
    try:
        return DryRunStatus(token)
    except ValueError:
        if token in {"warning", "warn"}:
            return DryRunStatus.warn
        if token in {"error", "fail", "failed"}:
            return DryRunStatus.fail
        return DryRunStatus.pass_


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _deduplicate(items: Iterable[BacklogItem]) -> list[BacklogItem]:
    seen: set[tuple[str, str, str]] = set()
    result: list[BacklogItem] = []
    for item in items:
        key = (item.severity.value, item.category, item.summary)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result

