"""Change management workflow primitives aligned with the roadmap expectations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Iterable, Mapping, MutableMapping, Sequence


__all__ = [
    "ChangeImpact",
    "ChangeStatus",
    "ChangeAssessmentStatus",
    "ChangeApproval",
    "ChangeWindow",
    "ChangeRequest",
    "ChangeManagementPolicy",
    "ChangeManagementAssessment",
    "evaluate_change_request",
    "generate_change_management_markdown",
]


class ChangeImpact(StrEnum):
    """Impact level associated with a change request."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ChangeStatus(StrEnum):
    """Lifecycle state for a change request."""

    draft = "draft"
    pending_approval = "pending_approval"
    scheduled = "scheduled"
    approved = "approved"
    rejected = "rejected"
    completed = "completed"


class ChangeAssessmentStatus(StrEnum):
    """Outcome of applying policy checks to a change request."""

    approved = "approved"
    needs_attention = "needs_attention"
    blocked = "blocked"


def _ensure_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class ChangeApproval:
    """Approval captured for a change request."""

    approver: str
    role: str
    approved_at: datetime | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "approver", str(self.approver).strip() or "unknown")
        object.__setattr__(self, "role", str(self.role).strip().lower() or "unspecified")
        if self.approved_at is not None:
            object.__setattr__(self, "approved_at", _ensure_datetime(self.approved_at))
        if self.note is not None:
            cleaned = str(self.note).strip()
            object.__setattr__(self, "note", cleaned or None)

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "approver": self.approver,
            "role": self.role,
        }
        if self.approved_at is not None:
            payload["approved_at"] = self.approved_at.isoformat()
        if self.note is not None:
            payload["note"] = self.note
        return payload


@dataclass(frozen=True)
class ChangeWindow:
    """Scheduled implementation window for a change."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _ensure_datetime(self.start))
        object.__setattr__(self, "end", _ensure_datetime(self.end))

    def contains(self, moment: datetime) -> bool:
        moment_utc = _ensure_datetime(moment)
        return self.start <= moment_utc <= self.end

    def duration_hours(self) -> float:
        duration = self.end - self.start
        return max(duration.total_seconds(), 0.0) / 3600.0


def _normalise_tags(tags: Iterable[str] | None) -> tuple[str, ...]:
    if not tags:
        return ()
    cleaned: list[str] = []
    for tag in tags:
        text = str(tag).strip()
        if text:
            cleaned.append(text.lower())
    return tuple(dict.fromkeys(cleaned))


def _normalise_metadata(metadata: Mapping[str, object] | None) -> Mapping[str, object]:
    if metadata is None:
        return {}
    return {str(key): value for key, value in metadata.items()}


@dataclass(frozen=True)
class ChangeRequest:
    """Structured representation of a change ready for evaluation."""

    change_id: str
    title: str
    description: str
    impact: ChangeImpact
    requested_by: str
    created_at: datetime
    window: ChangeWindow
    approvals: tuple[ChangeApproval, ...] = field(default_factory=tuple)
    status: ChangeStatus = ChangeStatus.draft
    metadata: Mapping[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "change_id", str(self.change_id).strip() or "unknown")
        object.__setattr__(self, "title", str(self.title).strip() or "Untitled change")
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "requested_by", str(self.requested_by).strip() or "unknown")
        object.__setattr__(self, "created_at", _ensure_datetime(self.created_at))
        approvals_sorted = tuple(sorted(self.approvals, key=lambda approval: (
            approval.approved_at or datetime.min.replace(tzinfo=UTC),
            approval.role,
        )))
        object.__setattr__(self, "approvals", approvals_sorted)
        object.__setattr__(self, "metadata", _normalise_metadata(self.metadata))
        object.__setattr__(self, "tags", _normalise_tags(self.tags))

    def approval_roles(self) -> set[str]:
        roles: set[str] = set()
        for approval in self.approvals:
            if approval.approved_at is None:
                continue
            roles.add(approval.role)
        return roles

    def as_dict(self) -> Mapping[str, object]:
        return {
            "change_id": self.change_id,
            "title": self.title,
            "description": self.description,
            "impact": self.impact.value,
            "requested_by": self.requested_by,
            "created_at": self.created_at.isoformat(),
            "window": {
                "start": self.window.start.isoformat(),
                "end": self.window.end.isoformat(),
            },
            "approvals": [approval.as_dict() for approval in self.approvals],
            "status": self.status.value,
            "metadata": dict(self.metadata),
            "tags": list(self.tags),
        }


DEFAULT_REQUIRED_ROLES: Mapping[ChangeImpact, tuple[str, ...]] = {
    ChangeImpact.low: ("operations",),
    ChangeImpact.medium: ("operations", "team_owner"),
    ChangeImpact.high: ("operations", "risk", "compliance"),
    ChangeImpact.critical: ("operations", "risk", "compliance", "security"),
}

DEFAULT_MINIMUM_LEAD_TIME: Mapping[ChangeImpact, timedelta] = {
    ChangeImpact.low: timedelta(hours=1),
    ChangeImpact.medium: timedelta(hours=4),
    ChangeImpact.high: timedelta(hours=24),
    ChangeImpact.critical: timedelta(hours=48),
}

DEFAULT_MAX_WINDOW_HOURS: Mapping[ChangeImpact, float] = {
    ChangeImpact.low: 4.0,
    ChangeImpact.medium: 8.0,
    ChangeImpact.high: 12.0,
    ChangeImpact.critical: 24.0,
}


@dataclass(frozen=True)
class ChangeManagementPolicy:
    """Policy thresholds used when evaluating change requests."""

    required_roles: Mapping[ChangeImpact, Sequence[str]] = field(
        default_factory=lambda: dict(DEFAULT_REQUIRED_ROLES)
    )
    minimum_lead_time: Mapping[ChangeImpact, timedelta] = field(
        default_factory=lambda: dict(DEFAULT_MINIMUM_LEAD_TIME)
    )
    max_window_hours: Mapping[ChangeImpact, float] = field(
        default_factory=lambda: dict(DEFAULT_MAX_WINDOW_HOURS)
    )

    def required_roles_for(self, impact: ChangeImpact) -> tuple[str, ...]:
        roles = self.required_roles.get(impact, ())
        return tuple(str(role).strip().lower() for role in roles if str(role).strip())

    def minimum_lead_time_for(self, impact: ChangeImpact) -> timedelta:
        return self.minimum_lead_time.get(impact, timedelta())

    def max_window_hours_for(self, impact: ChangeImpact) -> float:
        return float(self.max_window_hours.get(impact, 0.0))


@dataclass(frozen=True)
class ChangeManagementAssessment:
    """Result of evaluating a change request."""

    status: ChangeAssessmentStatus
    issues: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    approvals_missing: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status.value,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "approvals_missing": list(self.approvals_missing),
            "metadata": dict(self.metadata),
        }


def evaluate_change_request(
    change_request: ChangeRequest,
    policy: ChangeManagementPolicy | None = None,
    *,
    reference_time: datetime | None = None,
) -> ChangeManagementAssessment:
    """Evaluate a change request against governance policy rules."""

    policy = policy or ChangeManagementPolicy()
    now = _ensure_datetime(reference_time or datetime.now(tz=UTC))

    issues: list[str] = []
    warnings: list[str] = []

    window = change_request.window
    if window.start >= window.end:
        issues.append("Change window end must be after the start time.")

    duration_hours = window.duration_hours()
    max_hours_allowed = policy.max_window_hours_for(change_request.impact)
    if duration_hours > max_hours_allowed:
        warnings.append(
            "Change window duration "
            f"({duration_hours:.1f}h) exceeds policy maximum of {max_hours_allowed:.1f}h"
        )

    lead_time_required = policy.minimum_lead_time_for(change_request.impact)
    lead_time_actual = window.start - change_request.created_at
    if lead_time_actual < lead_time_required:
        issues.append(
            "Lead time is shorter than policy minimum "
            f"({lead_time_actual.total_seconds() / 3600:.1f}h provided, "
            f"{lead_time_required.total_seconds() / 3600:.1f}h required)."
        )

    if window.start < now:
        warnings.append("Change window start is in the past relative to evaluation time.")
    if window.end < now:
        warnings.append("Change window has already concluded.")

    if change_request.status is ChangeStatus.rejected:
        issues.append("Change request is marked as rejected.")
    elif change_request.status not in {
        ChangeStatus.approved,
        ChangeStatus.scheduled,
        ChangeStatus.completed,
    }:
        warnings.append(
            f"Change request is in '{change_request.status.value}' status and not yet fully approved."
        )

    obtained_roles = change_request.approval_roles()
    required_roles = policy.required_roles_for(change_request.impact)
    approvals_missing = tuple(role for role in required_roles if role not in obtained_roles)
    if approvals_missing:
        issues.append(
            "Missing required approvals: " + ", ".join(sorted(approvals_missing))
        )

    metadata_payload: dict[str, object] = {
        "impact": change_request.impact.value,
        "status": change_request.status.value,
        "window_start": window.start.isoformat(),
        "window_end": window.end.isoformat(),
        "window_duration_hours": round(duration_hours, 2),
        "lead_time_hours": round(lead_time_actual.total_seconds() / 3600, 2),
        "required_roles": list(required_roles),
        "approved_roles": sorted(obtained_roles),
    }
    if approvals_missing:
        metadata_payload["approvals_missing"] = list(approvals_missing)

    status = ChangeAssessmentStatus.approved
    if issues:
        status = ChangeAssessmentStatus.blocked
    elif warnings:
        status = ChangeAssessmentStatus.needs_attention

    return ChangeManagementAssessment(
        status=status,
        issues=tuple(issues),
        warnings=tuple(warnings),
        approvals_missing=approvals_missing,
        metadata=metadata_payload,
    )


def generate_change_management_markdown(
    change_request: ChangeRequest,
    assessment: ChangeManagementAssessment,
) -> str:
    """Render a markdown summary for change management reviewers."""

    lines = [
        f"# Change Request {change_request.change_id}",
        f"- Title: {change_request.title}",
        f"- Impact: {change_request.impact.value.title()}",
        f"- Status: {change_request.status.value}",
        f"- Requested by: {change_request.requested_by}",
        f"- Scheduled window: {change_request.window.start.isoformat()} → {change_request.window.end.isoformat()}",
        f"- Tags: {', '.join(change_request.tags) if change_request.tags else '—'}",
        f"- Assessment: {assessment.status.value}",
    ]

    if assessment.issues:
        lines.append("\n## Issues")
        for issue in assessment.issues:
            lines.append(f"- {issue}")

    if assessment.warnings:
        lines.append("\n## Warnings")
        for warning in assessment.warnings:
            lines.append(f"- {warning}")

    if assessment.approvals_missing:
        lines.append("\n## Missing approvals")
        for role in assessment.approvals_missing:
            lines.append(f"- {role}")

    if change_request.approvals:
        lines.append("\n## Approvals")
        for approval in change_request.approvals:
            approved_at = (
                approval.approved_at.isoformat() if approval.approved_at else "pending"
            )
            note = f" — {approval.note}" if approval.note else ""
            lines.append(f"- {approval.role}: {approval.approver} at {approved_at}{note}")

    if change_request.description:
        lines.append("\n## Description")
        lines.append(change_request.description)

    return "\n".join(lines)

