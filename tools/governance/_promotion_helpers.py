"""Shared helpers for policy promotion CLIs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.governance.policy_ledger import PolicyLedgerRecord


def build_log_entry(
    record: PolicyLedgerRecord,
    posture: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a JSON-serialisable payload describing the promotion."""

    payload: dict[str, Any] = {
        "policy_id": record.policy_id,
        "tactic_id": record.tactic_id,
        "stage": record.stage.value,
        "approvals": list(record.approvals),
        "updated_at": record.updated_at.isoformat(),
        "release_posture": dict(posture),
    }
    if record.evidence_id:
        payload["evidence_id"] = record.evidence_id
    if record.threshold_overrides:
        payload["threshold_overrides"] = dict(record.threshold_overrides)
    if record.policy_delta is not None and not record.policy_delta.is_empty():
        payload["policy_delta"] = dict(record.policy_delta.as_dict())
    if record.accepted_proposals:
        payload["accepted_proposals"] = list(record.accepted_proposals)
    if record.rejected_proposals:
        payload["rejected_proposals"] = list(record.rejected_proposals)
    if record.human_signoffs:
        payload["human_signoffs"] = [dict(entry) for entry in record.human_signoffs]
    if record.metadata:
        payload["metadata"] = dict(record.metadata)
    return payload


def write_promotion_log(log_path: Path, entry: Mapping[str, Any]) -> Path:
    """Append a promotion entry to the governance log file."""

    log_path = log_path.expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(entry)
    payload.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return log_path


def format_markdown_summary(
    record: PolicyLedgerRecord,
    posture: Mapping[str, Any],
) -> str:
    """Return a human-friendly Markdown summary of the promotion."""

    approvals = ", ".join(record.approvals) if record.approvals else "None"
    lines = [
        "# Policy Promotion Summary",
        "",
        f"- Policy ID: `{record.policy_id}`",
        f"- Tactic ID: `{record.tactic_id}`",
        f"- Stage: `{record.stage.value}`",
        f"- Approvals: {approvals}",
    ]
    if record.evidence_id:
        lines.append(f"- Evidence ID: `{record.evidence_id}`")
    if record.accepted_proposals:
        lines.append(
            "- Accepted Proposals: " + ", ".join(record.accepted_proposals)
        )
    if record.rejected_proposals:
        lines.append(
            "- Rejected Proposals: " + ", ".join(record.rejected_proposals)
        )
    lines.append(f"- Updated At: {record.updated_at.isoformat()}")

    thresholds = posture.get("thresholds", {})
    if thresholds:
        lines.append("")
        lines.append("## Effective Thresholds")
        for key, value in sorted(thresholds.items()):
            lines.append(f"- `{key}`: {value}")

    if record.threshold_overrides:
        lines.append("")
        lines.append("## Threshold Overrides")
        for key, value in sorted(record.threshold_overrides.items()):
            lines.append(f"- `{key}`: {value}")

    if record.policy_delta is not None and not record.policy_delta.is_empty():
        delta = record.policy_delta.as_dict()
        lines.append("")
        lines.append("## Policy Delta")
        for section, payload in sorted(delta.items()):
            lines.append(f"- **{section}**: {json.dumps(payload, sort_keys=True)}")

    if record.metadata:
        lines.append("")
        lines.append("## Metadata")
        for key, value in sorted(record.metadata.items()):
            lines.append(f"- `{key}`: {json.dumps(value, sort_keys=True)}")

    if record.human_signoffs:
        lines.append("")
        lines.append("## Human Sign-offs")
        for raw_entry in record.human_signoffs:
            entry = dict(raw_entry)
            summary_parts = []
            reviewer = entry.get("reviewer") or entry.get("name")
            if reviewer:
                summary_parts.append(str(reviewer))
            role = entry.get("role")
            if role:
                summary_parts.append(f"role={role}")
            verdict = entry.get("verdict")
            if verdict:
                summary_parts.append(f"verdict={verdict}")
            signed_at = entry.get("signed_at")
            if signed_at:
                summary_parts.append(f"signed_at={signed_at}")
            if summary_parts:
                lines.append("- " + ", ".join(summary_parts))
            else:
                lines.append(
                    "- " + json.dumps(entry, sort_keys=True, default=str)
                )

    declared_stage = posture.get("declared_stage")
    audit_stage = posture.get("audit_stage")
    audit_gaps = posture.get("audit_gaps", [])
    lines.append("")
    lines.append("## Governance Posture")
    lines.append(f"- Declared Stage: `{declared_stage}`")
    lines.append(f"- Audit Stage: `{audit_stage}`")
    lines.append(f"- Audit Enforced: {posture.get('audit_enforced', False)}")
    if audit_gaps:
        lines.append("- Audit Gaps: " + ", ".join(str(item) for item in audit_gaps))
    else:
        lines.append("- Audit Gaps: None")

    return "\n".join(lines) + "\n"
