"""Governance review gate registry and verdict tracking helpers.

This module formalises the roadmap requirement for governance review gates by
providing a typed registry that pairs static sign-off criteria (definitions)
with recorded verdicts.  It exposes helpers for loading gate definitions from a
YAML payload, persisting verdict state as JSON, and rendering the merged view as
structured compliance workflows or Markdown summaries suitable for governance
packets.

Typical usage::

    registry = ReviewGateRegistry.load(Path("config/governance/review_gates.yaml"),
                                      state_path=Path("artifacts/governance/review_gates.json"))
    registry.record_decision(
        gate_id="operations_final_signoff",
        decision=ReviewGateDecision(
            gate_id="operations_final_signoff",
            verdict=ReviewVerdict.pass_,
            decided_by=("Operations Lead", "Risk Chair"),
            notes=("Final dry run completed with clean evidence",),
            criteria_status={
                "dry_run_duration": ReviewCriterionStatus.met,
                "evidence_packet": ReviewCriterionStatus.met,
                "signoff_verdict": ReviewCriterionStatus.met,
            },
        ),
    )
    registry.save(Path("artifacts/governance/review_gates.json"))

The registry keeps verdict metadata distinct from the static definitions so the
same criteria can be shared across environments while reviewers update the
state file during governance sign-off meetings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence

import yaml

from src.compliance.workflow import (
    ComplianceWorkflowChecklist,
    ComplianceWorkflowSnapshot,
    ComplianceWorkflowTask,
    WorkflowTaskStatus,
)

__all__ = [
    "ReviewVerdict",
    "ReviewCriterionStatus",
    "ReviewGateCriterion",
    "ReviewGateDefinition",
    "ReviewGateDecision",
    "ReviewGateEntry",
    "ReviewGateRegistry",
]


_STATE_VERSION = 1


class ReviewVerdict(StrEnum):
    """Verdict recorded during a governance review gate."""

    pass_ = "pass"
    warn = "warn"
    fail = "fail"
    waived = "waived"

    @classmethod
    def from_value(cls, value: object | None) -> "ReviewVerdict":
        if isinstance(value, ReviewVerdict):
            return value
        label = str(value or "").strip().lower()
        if not label:
            raise ValueError("review verdict is required")
        if label == "pass":
            return cls.pass_
        if label in {"warn", "warning"}:
            return cls.warn
        if label in {"fail", "failed"}:
            return cls.fail
        if label in {"waived", "waive"}:
            return cls.waived
        raise ValueError(f"unknown review verdict: {value}")


class ReviewCriterionStatus(StrEnum):
    """Status recorded for an individual sign-off criterion."""

    met = "met"
    not_met = "not_met"
    waived = "waived"

    @classmethod
    def from_value(cls, value: object | None) -> "ReviewCriterionStatus":
        if isinstance(value, ReviewCriterionStatus):
            return value
        label = str(value or "").strip().lower().replace("-", "_")
        if label in {"met", "complete", "completed", "pass"}:
            return cls.met
        if label in {"not_met", "missing", "fail", "failed"}:
            return cls.not_met
        if label in {"waived", "waive", "n_a", "na"}:
            return cls.waived
        raise ValueError(f"unknown criterion status: {value}")


_slug_pattern = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = _slug_pattern.sub("_", value.lower()).strip("_")
    return slug or "criterion"


def _normalise_iterable(values: Iterable[str]) -> tuple[str, ...]:
    cleaned = [item.strip() for item in values if str(item).strip()]
    seen: set[str] = set()
    deduped: list[str] = []
    for item in cleaned:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return tuple(deduped)


@dataclass(frozen=True, slots=True)
class ReviewGateCriterion:
    """Single sign-off criterion associated with a review gate."""

    criterion_id: str
    description: str
    mandatory: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.criterion_id,
            "description": self.description,
            "mandatory": self.mandatory,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ReviewGateCriterion":
        description = str(payload.get("description") or "").strip()
        if not description:
            raise ValueError("criterion description is required")
        raw_id = payload.get("id")
        criterion_id = (
            str(raw_id).strip().lower()
            if isinstance(raw_id, (str, bytes)) and str(raw_id).strip()
            else _slugify(description)
        )
        mandatory_raw = payload.get("mandatory")
        mandatory = True if mandatory_raw is None else bool(mandatory_raw)
        return cls(criterion_id=criterion_id, description=description, mandatory=mandatory)


@dataclass(frozen=True, slots=True)
class ReviewGateDefinition:
    """Static definition describing a governance review gate."""

    gate_id: str
    title: str
    description: str
    severity: str = "medium"
    owners: tuple[str, ...] = field(default_factory=tuple)
    criteria: tuple[ReviewGateCriterion, ...] = field(default_factory=tuple)
    artifacts: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "gate_id": self.gate_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "criteria": [criterion.as_dict() for criterion in self.criteria],
        }
        if self.owners:
            payload["owners"] = list(self.owners)
        if self.artifacts:
            payload["artifacts"] = list(self.artifacts)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ReviewGateDefinition":
        gate_id_raw = payload.get("gate_id") or payload.get("id")
        gate_id = str(gate_id_raw or "").strip().lower()
        if not gate_id:
            raise ValueError("gate_id is required for review gates")
        title = str(payload.get("title") or "").strip() or gate_id
        description = str(payload.get("description") or "").strip() or title
        severity = str(payload.get("severity") or "medium").strip().lower() or "medium"
        owners_raw = payload.get("owners")
        owners: tuple[str, ...] = ()
        if isinstance(owners_raw, Sequence) and not isinstance(owners_raw, (str, bytes)):
            owners = _normalise_iterable(str(item) for item in owners_raw)
        criteria_payload = payload.get("criteria")
        criteria: list[ReviewGateCriterion] = []
        if isinstance(criteria_payload, Sequence):
            for item in criteria_payload:
                if isinstance(item, Mapping):
                    criteria.append(ReviewGateCriterion.from_mapping(item))
        artifacts_raw = payload.get("artifacts")
        artifacts: tuple[str, ...] = ()
        if isinstance(artifacts_raw, Sequence) and not isinstance(artifacts_raw, (str, bytes)):
            artifacts = _normalise_iterable(str(item) for item in artifacts_raw)
        return cls(
            gate_id=gate_id,
            title=title,
            description=description,
            severity=severity,
            owners=owners,
            criteria=tuple(criteria),
            artifacts=artifacts,
        )


@dataclass(frozen=True, slots=True)
class ReviewGateDecision:
    """Recorded verdict and metadata for a review gate."""

    gate_id: str
    verdict: ReviewVerdict
    decided_at: datetime
    decided_by: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    criteria_status: Mapping[str, ReviewCriterionStatus] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "gate_id": self.gate_id,
            "verdict": self.verdict.value,
            "decided_at": self.decided_at.isoformat(),
        }
        if self.decided_by:
            payload["decided_by"] = list(self.decided_by)
        if self.notes:
            payload["notes"] = list(self.notes)
        if self.criteria_status:
            payload["criteria_status"] = {
                key: status.value for key, status in self.criteria_status.items()
            }
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ReviewGateDecision":
        gate_id_raw = payload.get("gate_id") or payload.get("id")
        gate_id = str(gate_id_raw or "").strip().lower()
        if not gate_id:
            raise ValueError("gate_id is required for review decisions")
        verdict = ReviewVerdict.from_value(payload.get("verdict"))
        decided_at_raw = payload.get("decided_at") or payload.get("recorded_at")
        if isinstance(decided_at_raw, datetime):
            decided_at = decided_at_raw.astimezone(UTC)
        elif isinstance(decided_at_raw, str):
            decided_at = datetime.fromisoformat(decided_at_raw)
            if decided_at.tzinfo is None:
                decided_at = decided_at.replace(tzinfo=UTC)
            else:
                decided_at = decided_at.astimezone(UTC)
        else:
            decided_at = datetime.now(tz=UTC)
        decided_by_raw = payload.get("decided_by")
        decided_by: tuple[str, ...] = ()
        if isinstance(decided_by_raw, Sequence) and not isinstance(decided_by_raw, (str, bytes)):
            decided_by = _normalise_iterable(str(item) for item in decided_by_raw)
        notes_raw = payload.get("notes")
        notes: tuple[str, ...] = ()
        if isinstance(notes_raw, Sequence) and not isinstance(notes_raw, (str, bytes)):
            notes = tuple(str(item).strip() for item in notes_raw if str(item).strip())
        criteria_raw = payload.get("criteria_status") or {}
        criteria_status: dict[str, ReviewCriterionStatus] = {}
        if isinstance(criteria_raw, Mapping):
            for key, value in criteria_raw.items():
                criterion_key = str(key).strip().lower()
                if not criterion_key:
                    continue
                try:
                    criteria_status[criterion_key] = ReviewCriterionStatus.from_value(value)
                except ValueError:
                    continue
        return cls(
            gate_id=gate_id,
            verdict=verdict,
            decided_at=decided_at,
            decided_by=decided_by,
            notes=notes,
            criteria_status=criteria_status,
        )


@dataclass(frozen=True, slots=True)
class ReviewGateEntry:
    """Merged view of a review gate definition and the recorded decision."""

    definition: ReviewGateDefinition
    decision: ReviewGateDecision | None = None

    def status(self) -> WorkflowTaskStatus:
        return _resolve_task_status(self.definition, self.decision)


def _resolve_task_status(
    definition: ReviewGateDefinition, decision: ReviewGateDecision | None
) -> WorkflowTaskStatus:
    if decision is None:
        return WorkflowTaskStatus.todo

    mandatory_missing = False
    mandatory_pending = False

    for criterion in definition.criteria:
        status = decision.criteria_status.get(criterion.criterion_id)
        if status is None:
            if criterion.mandatory:
                mandatory_pending = True
            continue
        if status is ReviewCriterionStatus.not_met and criterion.mandatory:
            mandatory_missing = True
            break

    if decision.verdict is ReviewVerdict.fail or mandatory_missing:
        return WorkflowTaskStatus.blocked
    if mandatory_pending or decision.verdict is ReviewVerdict.warn:
        return WorkflowTaskStatus.in_progress
    return WorkflowTaskStatus.completed


def _load_yaml(path: Path) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - surfaced by caller
        raise FileNotFoundError(f"review gate definition file not found: {path}") from exc
    except yaml.YAMLError as exc:  # pragma: no cover - configuration issue
        raise ValueError(f"failed to parse review gate definitions: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("review gate definition file must contain a mapping")
    return raw


@dataclass(slots=True)
class ReviewGateRegistry:
    """Registry that merges review gate definitions with recorded decisions."""

    definitions: Mapping[str, ReviewGateDefinition]
    decisions: MutableMapping[str, ReviewGateDecision]

    @classmethod
    def from_payload(
        cls,
        definitions: Iterable[ReviewGateDefinition],
        decisions: Iterable[ReviewGateDecision] | None = None,
    ) -> "ReviewGateRegistry":
        definition_map = {definition.gate_id: definition for definition in definitions}
        decision_map: dict[str, ReviewGateDecision] = {}
        if decisions is not None:
            for decision in decisions:
                if decision.gate_id in definition_map:
                    decision_map[decision.gate_id] = decision
        return cls(definitions=definition_map, decisions=decision_map)

    @classmethod
    def load(
        cls,
        definition_path: Path,
        *,
        state_path: Path | None = None,
    ) -> "ReviewGateRegistry":
        payload = _load_yaml(definition_path)
        gates_payload = payload.get("gates")
        if not isinstance(gates_payload, Sequence):
            raise ValueError("review gate configuration must include a 'gates' sequence")
        definitions = [
            ReviewGateDefinition.from_mapping(item)
            for item in gates_payload
            if isinstance(item, Mapping)
        ]
        decisions: list[ReviewGateDecision] = []
        if state_path is not None and state_path.exists():
            state_raw = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(state_raw, Mapping):
                gates_state = state_raw.get("gates")
                if isinstance(gates_state, Sequence):
                    for item in gates_state:
                        if isinstance(item, Mapping):
                            try:
                                decisions.append(ReviewGateDecision.from_mapping(item))
                            except ValueError:
                                continue
        return cls.from_payload(definitions, decisions)

    def iter_entries(self) -> Iterator[ReviewGateEntry]:
        for gate_id in sorted(self.definitions):
            definition = self.definitions[gate_id]
            decision = self.decisions.get(gate_id)
            yield ReviewGateEntry(definition=definition, decision=decision)

    def get(self, gate_id: str) -> ReviewGateEntry | None:
        definition = self.definitions.get(gate_id)
        if definition is None:
            return None
        return ReviewGateEntry(definition=definition, decision=self.decisions.get(gate_id))

    def record_decision(self, decision: ReviewGateDecision) -> ReviewGateEntry:
        if decision.gate_id not in self.definitions:
            raise KeyError(f"unknown review gate: {decision.gate_id}")
        self.decisions[decision.gate_id] = decision
        return ReviewGateEntry(
            definition=self.definitions[decision.gate_id],
            decision=decision,
        )

    def remove_decision(self, gate_id: str) -> None:
        self.decisions.pop(gate_id, None)

    def to_state_payload(self) -> dict[str, object]:
        return {
            "version": _STATE_VERSION,
            "gates": [decision.as_dict() for decision in self.decisions.values()],
        }

    def save(self, path: Path) -> None:
        payload = self.to_state_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def to_summary(self) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        for entry in self.iter_entries():
            decision = entry.decision
            definition = entry.definition
            status = entry.status()
            criterion_summaries: list[dict[str, object]] = []
            for criterion in definition.criteria:
                status_value: str | None = None
                if decision is not None:
                    recorded = decision.criteria_status.get(criterion.criterion_id)
                    if recorded is not None:
                        status_value = recorded.value
                criterion_summaries.append(
                    {
                        "id": criterion.criterion_id,
                        "description": criterion.description,
                        "mandatory": criterion.mandatory,
                        "status": status_value or ("pending" if decision else "unset"),
                    }
                )
            entry_payload: dict[str, object] = {
                "gate_id": definition.gate_id,
                "title": definition.title,
                "description": definition.description,
                "severity": definition.severity,
                "status": status.value,
                "verdict": decision.verdict.value if decision else "pending",
                "owners": list(definition.owners),
                "artifacts": list(definition.artifacts),
                "criteria": criterion_summaries,
            }
            if decision is not None:
                entry_payload["decided_at"] = decision.decided_at.isoformat()
                entry_payload["decided_by"] = list(decision.decided_by)
                if decision.notes:
                    entry_payload["notes"] = list(decision.notes)
            entries.append(entry_payload)
        return {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "gates": entries,
        }

    def to_markdown(self) -> str:
        lines = ["# Governance Review Gates", ""]
        for entry in self.iter_entries():
            definition = entry.definition
            decision = entry.decision
            status = entry.status()
            verdict_label = decision.verdict.value.upper() if decision else "PENDING"
            lines.append(f"## {definition.title} ({status.value.upper()} / {verdict_label})")
            lines.append(definition.description)
            if definition.owners:
                lines.append("")
                lines.append("**Owners:** " + ", ".join(definition.owners))
            if decision is not None:
                lines.append("**Decided at:** " + decision.decided_at.isoformat())
                if decision.decided_by:
                    lines.append("**Decided by:** " + ", ".join(decision.decided_by))
            if decision is not None and decision.notes:
                lines.append("**Notes:**")
                for note in decision.notes:
                    lines.append(f"- {note}")
            lines.append("")
            lines.append("| Criterion | Mandatory | Status |")
            lines.append("| --- | --- | --- |")
            for criterion in definition.criteria:
                criterion_status = None
                if decision is not None:
                    recorded = decision.criteria_status.get(criterion.criterion_id)
                    if recorded is not None:
                        criterion_status = recorded.value
                recorded_label = (
                    criterion_status.upper()
                    if criterion_status is not None
                    else ("PENDING" if decision is not None else "UNSET")
                )
                mandatory_label = "yes" if criterion.mandatory else "no"
                lines.append(
                    f"| {criterion.description} | {mandatory_label} | {recorded_label} |"
                )
            if definition.artifacts:
                lines.append("")
                lines.append("**Artifacts:**")
                for artifact in definition.artifacts:
                    lines.append(f"- {artifact}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def to_workflow_snapshot(
        self,
        *,
        regulation: str = "AlphaTrade Governance",
        generated_at: datetime | None = None,
    ) -> ComplianceWorkflowSnapshot:
        tasks: list[ComplianceWorkflowTask] = []
        for entry in self.iter_entries():
            definition = entry.definition
            decision = entry.decision
            status = entry.status()
            summary_parts = [f"severity={definition.severity}"]
            summary_parts.append(f"verdict={decision.verdict.value if decision else 'pending'}")
            if decision is not None and decision.notes:
                summary_parts.append("notes=" + ";".join(decision.notes))
            pending_criteria = [
                criterion.criterion_id
                for criterion in definition.criteria
                if decision is None
                or criterion.criterion_id not in decision.criteria_status
            ]
            if pending_criteria:
                summary_parts.append("pending=" + ",".join(pending_criteria))
            metadata: dict[str, object] = {
                "gate_id": definition.gate_id,
                "owners": list(definition.owners),
                "artifacts": list(definition.artifacts),
            }
            if decision is not None:
                metadata["decided_at"] = decision.decided_at.isoformat()
                metadata["decided_by"] = list(decision.decided_by)
            tasks.append(
                ComplianceWorkflowTask(
                    task_id=f"review_gate::{definition.gate_id}",
                    title=definition.title,
                    status=status,
                    summary="; ".join(summary_parts),
                    severity=definition.severity,
                    metadata=metadata,
                )
            )
        if tasks:
            worst_status = max(
                tasks,
                key=lambda task: {
                    WorkflowTaskStatus.completed: 0,
                    WorkflowTaskStatus.in_progress: 1,
                    WorkflowTaskStatus.todo: 2,
                    WorkflowTaskStatus.blocked: 3,
                }[task.status],
            ).status
        else:
            worst_status = WorkflowTaskStatus.todo
        checklist = ComplianceWorkflowChecklist(
            name="Governance Review Gates",
            regulation=regulation,
            status=worst_status,
            tasks=tuple(tasks),
            metadata={"gate_count": len(tasks)},
        )
        generated_ts = generated_at or datetime.now(tz=UTC)
        return ComplianceWorkflowSnapshot(
            status=worst_status,
            generated_at=generated_ts,
            workflows=(checklist,),
            metadata={"source": "review_gates"},
        )
try:  # Python < 3.11 fallback
    from datetime import UTC
except ImportError:  # pragma: no cover - compatibility branch
    from datetime import timezone

    UTC = timezone.utc  # type: ignore[assignment]
