"""Markdown changelog builder for policy ledger promotions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, MutableMapping, Sequence

from .policy_ledger import (
    LedgerReleaseManager,
    PolicyDelta,
    PolicyLedgerRecord,
    PolicyLedgerStage,
    PolicyLedgerStore,
)


DEFAULT_POLICY_PROMOTION_RUNBOOK = (
    "docs/operations/runbooks/policy_promotion_governance.md"
)

try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    UTC = timezone.utc  # type: ignore[assignment]

__all__ = [
    "DEFAULT_POLICY_PROMOTION_RUNBOOK",
    "render_policy_ledger_changelog",
]


@dataclass(frozen=True)
class _RenderedHistoryEntry:
    applied_at: str
    prior_stage: str
    next_stage: str
    approvals: str
    evidence: str
    delta_summary: str


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _format_collection(values: Iterable[str]) -> str:
    items = sorted({item for item in values if item})
    return ", ".join(items) if items else "none (see runbook)"


def _format_evidence(evidence_id: str | None) -> str:
    return evidence_id or "missing (see runbook)"


def _summarise_thresholds(thresholds: Mapping[str, object]) -> str:
    filtered: MutableMapping[str, object] = {
        key: value
        for key, value in thresholds.items()
        if key != "stage"
    }
    if not filtered:
        return "default profile"
    parts = []
    for key in sorted(filtered):
        value = filtered[key]
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _summarise_policy_delta(delta: Mapping[str, object] | PolicyDelta | None) -> str:
    if isinstance(delta, PolicyDelta):
        payload = delta.as_dict()
    elif isinstance(delta, Mapping):
        payload = dict(delta)
    else:
        return "none"

    if not payload:
        return "none"

    parts: list[str] = []
    regime = payload.get("regime")
    if regime:
        parts.append(f"regime={regime}")
    risk_payload = payload.get("risk_config")
    if isinstance(risk_payload, Mapping) and risk_payload:
        keys = ",".join(sorted(str(key) for key in risk_payload.keys()))
        parts.append(f"risk_config[{keys}]")
    guardrail_payload = payload.get("router_guardrails")
    if isinstance(guardrail_payload, Mapping) and guardrail_payload:
        keys = ",".join(sorted(str(key) for key in guardrail_payload.keys()))
        parts.append(f"router_guardrails[{keys}]")
    notes = payload.get("notes")
    if isinstance(notes, Sequence) and notes:
        parts.append(f"notes={len(notes)}")
    return ", ".join(parts) if parts else "none"


def _render_history(record: PolicyLedgerRecord) -> Sequence[_RenderedHistoryEntry]:
    entries: list[_RenderedHistoryEntry] = []
    for entry in record.history:
        if not isinstance(entry, Mapping):
            continue
        applied_at = str(entry.get("applied_at") or "")
        prior_stage = str(entry.get("prior_stage") or "-")
        next_stage = str(entry.get("next_stage") or "-")
        approvals = _format_collection(
            str(value)
            for value in entry.get("approvals", ())
            if isinstance(value, str) or value is not None
        )
        evidence = _format_evidence(entry.get("evidence_id"))
        delta_summary = "none"
        payload = entry.get("policy_delta")
        if isinstance(payload, Mapping):
            delta_summary = _summarise_policy_delta(payload)
        entries.append(
            _RenderedHistoryEntry(
                applied_at=applied_at,
                prior_stage=prior_stage,
                next_stage=next_stage,
                approvals=approvals,
                evidence=evidence,
                delta_summary=delta_summary,
            )
        )
    if not entries:
        entries.append(
            _RenderedHistoryEntry(
                applied_at="",
                prior_stage="-",
                next_stage=record.stage.value,
                approvals=_format_collection(record.approvals),
                evidence=_format_evidence(record.evidence_id),
                delta_summary=_summarise_policy_delta(record.policy_delta),
            )
        )
    return entries


def _render_record_section(
    record: PolicyLedgerRecord,
    *,
    runbook_url: str,
    thresholds: Mapping[str, object],
) -> list[str]:
    lines = [
        f"## {record.policy_id} -> {record.stage.value.upper()}",
        f"- tactic: `{record.tactic_id}`",
        f"- stage: `{record.stage.value}`",
        f"- updated: {_isoformat(record.updated_at)}",
        f"- approvals: {_format_collection(record.approvals)}",
        f"- evidence: {_format_evidence(record.evidence_id)}",
        f"- thresholds: {_summarise_thresholds(thresholds)}",
        f"- runbook: {runbook_url}",
    ]

    delta_summary = _summarise_policy_delta(record.policy_delta)
    if delta_summary != "none":
        lines.append(f"- policy_delta: {delta_summary}")

    lines.append("")
    lines.append("| applied_at | prior_stage | next_stage | approvals | evidence | delta |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for entry in _render_history(record):
        lines.append(
            "| {applied} | {prior} | {next} | {approvals} | {evidence} | {delta} |".format(
                applied=entry.applied_at or "-",
                prior=entry.prior_stage or "-",
                next=entry.next_stage or "-",
                approvals=entry.approvals,
                evidence=entry.evidence,
                delta=entry.delta_summary,
            )
        )

    lines.append("")
    return lines


def render_policy_ledger_changelog(
    store: PolicyLedgerStore,
    *,
    title: str = "Policy Ledger Governance Summary",
    runbook_url: str = DEFAULT_POLICY_PROMOTION_RUNBOOK,
    generated_at: datetime | None = None,
) -> str:
    """Render a Markdown changelog describing the policy ledger posture."""

    manager = LedgerReleaseManager(
        store,
        default_stage=PolicyLedgerStage.EXPERIMENT,
    )
    generated_ts = (generated_at or datetime.now(tz=UTC)).astimezone(UTC)
    records = sorted(
        store.iter_records(),
        key=lambda record: record.updated_at,
        reverse=True,
    )

    lines = [
        f"# {title}",
        "",
        f"- generated_at: {generated_ts.isoformat()}",
        f"- records: {len(records)}",
        f"- runbook: {runbook_url}",
        "",
    ]

    if not records:
        lines.append("_No policy ledger records available._")
        lines.append("")
        return "\n".join(lines)

    for record in records:
        thresholds = manager.resolve_thresholds(record.policy_id)
        lines.extend(
            _render_record_section(
                record,
                runbook_url=runbook_url,
                thresholds=thresholds,
            )
        )

    return "\n".join(lines)
