"""Create review briefs for the AlphaTrade final dry run sign-off."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.operations.dry_run_audit import (
    DryRunDiaryIssue,
    DryRunIncident,
    DryRunSignOffFinding,
    DryRunSignOffReport,
    DryRunStatus,
    DryRunSummary,
    humanise_timedelta,
)
from src.operations.dry_run_packet import DryRunPacketPaths


@dataclass(frozen=True)
class ReviewActionItem:
    """Follow-up item captured for the final dry run wrap-up."""

    severity: DryRunStatus
    category: str
    description: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "description": self.description,
            "context": {key: value for key, value in self.context.items()},
        }


@dataclass(frozen=True)
class FinalDryRunReview:
    """Structured brief for the final dry run review meeting."""

    generated_at: datetime
    summary: DryRunSummary
    sign_off_report: DryRunSignOffReport | None
    action_items: tuple[ReviewActionItem, ...]
    run_label: str | None = None
    attendees: tuple[str, ...] = tuple()
    notes: tuple[str, ...] = tuple()
    highlights: Mapping[str, Any] = field(default_factory=dict)
    evidence_packet: DryRunPacketPaths | None = None

    @property
    def status(self) -> DryRunStatus:
        severities: list[DryRunStatus] = [self.summary.status]
        if self.sign_off_report is not None:
            severities.append(self.sign_off_report.status)
        severities.extend(item.severity for item in self.action_items)
        if any(severity is DryRunStatus.fail for severity in severities):
            return DryRunStatus.fail
        if any(severity is DryRunStatus.warn for severity in severities):
            return DryRunStatus.warn
        return DryRunStatus.pass_

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "summary_status": self.summary.status.value,
            "sign_off_status": (
                self.sign_off_report.status.value if self.sign_off_report else None
            ),
            "run_label": self.run_label,
            "attendees": list(self.attendees),
            "notes": list(self.notes),
            "highlights": {key: value for key, value in self.highlights.items()},
            "action_items": [item.as_dict() for item in self.action_items],
        }
        if self.evidence_packet is not None:
            payload["evidence_packet"] = self.evidence_packet.as_dict()
        return payload

    def to_markdown(
        self,
        *,
        include_summary: bool = True,
        include_sign_off: bool = True,
    ) -> str:
        lines = [
            f"# Final Dry Run Review — {self.status.value.upper()}",
        ]
        if self.run_label:
            lines.append(f"Run: {self.run_label}")
        lines.append(
            f"Generated: {self.generated_at.astimezone(UTC).isoformat()}"
        )
        if self.attendees:
            lines.append("Attendees: " + ", ".join(self.attendees))
        lines.append("")

        if self.highlights:
            lines.append("## Highlights")
            for key, value in sorted(self.highlights.items()):
                lines.append(f"- {key}: {value}")
            lines.append("")

        lines.append("## Evidence Status")
        lines.append(f"- Dry run summary: {self.summary.status.value.upper()}")
        if self.sign_off_report is not None:
            lines.append(
                f"- Sign-off assessment: {self.sign_off_report.status.value.upper()}"
            )
        if self.evidence_packet is not None:
            if self.evidence_packet.archive_path is not None:
                lines.append(
                    f"- Evidence packet archive: {self.evidence_packet.archive_path.as_posix()}"
                )
            else:
                lines.append(
                    f"- Evidence packet directory: {self.evidence_packet.output_dir.as_posix()}"
                )
        lines.append("")

        lines.append("## Action Items")
        if self.action_items:
            for item in self.action_items:
                lines.append(
                    f"- {item.severity.value.upper()} — {item.category}: {item.description}"
                )
                if item.context:
                    for key, value in sorted(item.context.items()):
                        lines.append(f"  - {key}: {value}")
        else:
            lines.append("- None")
        lines.append("")

        if self.notes:
            lines.append("## Notes")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        appendix_sections: list[tuple[str, str]] = []
        if include_summary:
            appendix_sections.append(
                ("### Dry run summary", self.summary.to_markdown().rstrip())
            )
        if include_sign_off and self.sign_off_report is not None:
            appendix_sections.append(
                ("### Sign-off assessment", self.sign_off_report.to_markdown().rstrip())
            )
        if appendix_sections:
            lines.append("## Appendices")
            for heading, body in appendix_sections:
                lines.append(heading)
                lines.append(body)
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def build_review(
    summary: DryRunSummary,
    sign_off_report: DryRunSignOffReport | None = None,
    *,
    run_label: str | None = None,
    attendees: Iterable[str] = (),
    notes: Iterable[str] = (),
    evidence_packet: DryRunPacketPaths | None = None,
) -> FinalDryRunReview:
    """Assemble a :class:`FinalDryRunReview` from dry run evidence."""

    attendees_tuple = tuple(
        token
        for raw in attendees
        for token in [_normalise_token(raw)]
        if token
    )
    notes_tuple = tuple(note.strip() for note in notes if note and note.strip())
    highlights = _extract_highlights(summary)
    action_items = _collect_action_items(summary, sign_off_report)
    generated_at = datetime.now(tz=UTC)

    run_token = _normalise_token(run_label) if run_label else ""

    return FinalDryRunReview(
        generated_at=generated_at,
        summary=summary,
        sign_off_report=sign_off_report,
        action_items=action_items,
        run_label=run_token or None,
        attendees=attendees_tuple,
        notes=notes_tuple,
        highlights=highlights,
        evidence_packet=evidence_packet,
    )


def _extract_highlights(summary: DryRunSummary) -> Mapping[str, Any]:
    highlights: MutableMapping[str, Any] = {}
    log_summary = summary.log_summary
    if log_summary is not None:
        if log_summary.started_at is not None:
            highlights["Log start"] = log_summary.started_at.astimezone(UTC).isoformat()
        if log_summary.ended_at is not None:
            highlights["Log end"] = log_summary.ended_at.astimezone(UTC).isoformat()
        if log_summary.duration is not None:
            highlights["Observed duration"] = humanise_timedelta(log_summary.duration)
        if log_summary.uptime_ratio is not None:
            highlights["Uptime"] = f"{log_summary.uptime_ratio * 100:.2f}%"
        if log_summary.level_counts:
            highlights["Log levels"] = ", ".join(
                f"{level}={count}" for level, count in sorted(log_summary.level_counts.items())
            )
    diary_summary = summary.diary_summary
    if diary_summary is not None:
        highlights["Diary entries"] = diary_summary.total_entries
        if diary_summary.first_recorded_at is not None:
            highlights["Diary first"] = diary_summary.first_recorded_at.astimezone(UTC).isoformat()
        if diary_summary.last_recorded_at is not None:
            highlights["Diary last"] = diary_summary.last_recorded_at.astimezone(UTC).isoformat()
    performance_summary = summary.performance_summary
    if performance_summary is not None:
        highlights["Trades"] = performance_summary.total_trades
        if performance_summary.roi is not None:
            highlights["ROI"] = f"{performance_summary.roi * 100:.2f}%"
        if performance_summary.win_rate is not None:
            highlights["Win rate"] = f"{performance_summary.win_rate * 100:.2f}%"
        if performance_summary.sharpe_ratio is not None:
            highlights["Sharpe ratio"] = round(performance_summary.sharpe_ratio, 2)
    return highlights


def _collect_action_items(
    summary: DryRunSummary,
    sign_off_report: DryRunSignOffReport | None,
) -> tuple[ReviewActionItem, ...]:
    items: list[ReviewActionItem] = []

    log_summary = summary.log_summary
    if log_summary is not None:
        items.extend(
            _incident_to_action(incident, category="logs")
            for incident in log_summary.errors
        )
        items.extend(
            _incident_to_action(incident, category="logs")
            for incident in log_summary.warnings
        )
        items.extend(
            _incident_to_action(incident, category="log_gap")
            for incident in log_summary.gap_incidents
        )
        items.extend(
            _incident_to_action(incident, category="log_content")
            for incident in log_summary.content_incidents
        )

    diary_summary = summary.diary_summary
    if diary_summary is not None:
        items.extend(_diary_issue_to_action(issue) for issue in diary_summary.issues)

    performance_summary = summary.performance_summary
    if performance_summary is not None and performance_summary.status is DryRunStatus.warn:
        items.append(
            ReviewActionItem(
                severity=DryRunStatus.warn,
                category="performance",
                description="Performance telemetry raised warnings.",
                context=performance_summary.as_dict(),
            )
        )

    if sign_off_report is not None:
        items.extend(_sign_off_finding_to_action(finding) for finding in sign_off_report.findings)

    unique_items = _deduplicate_actions(items)
    unique_items.sort(
        key=lambda item: (
            _SEVERITY_ORDER.get(item.severity, 99),
            item.category,
            item.description,
        )
    )
    return tuple(unique_items)


def _incident_to_action(incident: DryRunIncident, *, category: str) -> ReviewActionItem:
    context = incident.as_dict()
    return ReviewActionItem(
        severity=incident.severity,
        category=category,
        description=incident.summary,
        context=context,
    )


def _diary_issue_to_action(issue: DryRunDiaryIssue) -> ReviewActionItem:
    return ReviewActionItem(
        severity=issue.severity,
        category="diary",
        description=f"Diary policy {issue.policy_id} flagged entry {issue.entry_id}",
        context=issue.as_dict(),
    )


def _sign_off_finding_to_action(finding: DryRunSignOffFinding) -> ReviewActionItem:
    return ReviewActionItem(
        severity=finding.severity,
        category="sign_off",
        description=finding.message,
        context=finding.as_dict(),
    )


def _deduplicate_actions(items: Sequence[ReviewActionItem]) -> list[ReviewActionItem]:
    seen: set[tuple[str, str, str]] = set()
    result: list[ReviewActionItem] = []
    for item in items:
        key = (item.severity.value, item.category, item.description)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


_SEVERITY_ORDER: Mapping[DryRunStatus, int] = {
    DryRunStatus.fail: 0,
    DryRunStatus.warn: 1,
    DryRunStatus.pass_: 2,
}


def _normalise_token(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(part for part in value.strip().split() if part)


__all__ = [
    "FinalDryRunReview",
    "ReviewActionItem",
    "build_review",
]
