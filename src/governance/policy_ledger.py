"""Policy ledger store and release management helpers for AlphaTrade loop expansion.

This module introduces a lightweight governance ledger that records promotion
stages for routing tactics, the approvals attached to those promotions, and any
threshold overrides that should be applied when DriftSentry evaluates trade
intents.  The intent is to provide a single source of truth that bridges
DecisionDiary evidence with execution guardrails so tactics can graduate from
paper trading into limited live capital without bypassing governance.

The storage format is JSON on disk to keep the dependency footprint minimal
while still giving operators a deterministic artefact they can review during
release sign-off.  The :class:`LedgerReleaseManager` surfaces ergonomic helper
methods the trading runtime can query when deciding whether a tactic is still in
shadow mode, paper trading, or eligible for limited live deployment.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

import json
import logging

logger = logging.getLogger(__name__)

try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for 3.10 runtimes
    UTC = timezone.utc  # type: ignore[assignment]


if TYPE_CHECKING:  # pragma: no cover - typing convenience
    from src.compliance.workflow import ComplianceWorkflowSnapshot


__all__ = [
    "PolicyLedgerStage",
    "PolicyLedgerRecord",
    "PolicyLedgerStore",
    "LedgerReleaseManager",
    "PolicyDelta",
    "PolicyLedgerFeatureFlags",
    "build_policy_governance_workflow",
]


class PolicyLedgerStage(str, Enum):
    """Stages a tactic passes through on its way to limited live capital."""

    EXPERIMENT = "experiment"
    PAPER = "paper"
    PILOT = "pilot"
    LIMITED_LIVE = "limited_live"

    @classmethod
    def from_value(cls, value: str | "PolicyLedgerStage" | None) -> "PolicyLedgerStage":
        if isinstance(value, PolicyLedgerStage):
            return value
        if not value:
            return cls.EXPERIMENT
        normalised = str(value).strip().lower()
        for stage in cls:
            if stage.value == normalised:
                return stage
        raise ValueError(f"Unknown policy ledger stage: {value}")


_STAGE_ORDER: Mapping[PolicyLedgerStage, int] = {
    PolicyLedgerStage.EXPERIMENT: 0,
    PolicyLedgerStage.PAPER: 1,
    PolicyLedgerStage.PILOT: 2,
    PolicyLedgerStage.LIMITED_LIVE: 3,
}


def _validate_stage_progress(old: PolicyLedgerStage, new: PolicyLedgerStage) -> None:
    if _STAGE_ORDER[new] < _STAGE_ORDER[old]:
        raise ValueError(
            "Cannot regress policy ledger stage",
        )


@dataclass(slots=True)
class PolicyLedgerRecord:
    """Ledger entry tying a tactic to its promotion posture."""

    policy_id: str
    tactic_id: str
    stage: PolicyLedgerStage
    approvals: tuple[str, ...] = ()
    evidence_id: str | None = None
    threshold_overrides: Mapping[str, float | str] = field(default_factory=dict)
    policy_delta: PolicyDelta | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    history: tuple[Mapping[str, Any], ...] = ()

    def with_stage(
        self,
        stage: PolicyLedgerStage,
        *,
        approvals: Iterable[str] | None = None,
        evidence_id: str | None = None,
        threshold_overrides: Mapping[str, float | str] | None = None,
        policy_delta: PolicyDelta | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> "PolicyLedgerRecord":
        _validate_stage_progress(self.stage, stage)
        applied_timestamp = timestamp or datetime.now(tz=UTC)
        delta = self.policy_delta if policy_delta is None else _coerce_policy_delta(policy_delta)
        history_entry = {
            "prior_stage": self.stage.value,
            "next_stage": stage.value,
            "applied_at": applied_timestamp.isoformat(),
            "approvals": sorted(tuple(approvals or self.approvals)),
            "evidence_id": evidence_id or self.evidence_id,
        }
        if delta is not None and not delta.is_empty():
            history_entry["policy_delta"] = dict(delta.as_dict())
        new_history = self.history + (history_entry,)
        return PolicyLedgerRecord(
            policy_id=self.policy_id,
            tactic_id=self.tactic_id,
            stage=stage,
            approvals=tuple(sorted(tuple(approvals or self.approvals))),
            evidence_id=evidence_id or self.evidence_id,
            threshold_overrides=dict(threshold_overrides or self.threshold_overrides),
            policy_delta=delta,
            metadata=dict(self.metadata) | (dict(metadata) if metadata else {}),
            created_at=self.created_at,
            updated_at=applied_timestamp,
            history=new_history,
        )

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "policy_id": self.policy_id,
            "tactic_id": self.tactic_id,
            "stage": self.stage.value,
            "approvals": list(self.approvals),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "history": list(self.history),
        }
        if self.evidence_id:
            payload["evidence_id"] = self.evidence_id
        if self.threshold_overrides:
            payload["threshold_overrides"] = dict(self.threshold_overrides)
        if self.policy_delta is not None and not self.policy_delta.is_empty():
            payload["policy_delta"] = dict(self.policy_delta.as_dict())
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def audit_gaps(
        self,
        *,
        expected_stage: "PolicyLedgerStage" | str | None = None,
    ) -> tuple[str, ...]:
        """Return audit prerequisites missing for the expected release stage."""

        stage = (
            PolicyLedgerStage.from_value(expected_stage)
            if expected_stage is not None
            else self.stage
        )
        gaps: list[str] = []

        if stage in (
            PolicyLedgerStage.PAPER,
            PolicyLedgerStage.PILOT,
            PolicyLedgerStage.LIMITED_LIVE,
        ) and not self.evidence_id:
            gaps.append("missing_evidence")

        if stage in (PolicyLedgerStage.PILOT, PolicyLedgerStage.LIMITED_LIVE):
            if not self.approvals:
                gaps.append("missing_approvals")
            elif stage is PolicyLedgerStage.LIMITED_LIVE and len(self.approvals) < 2:
                gaps.append("additional_approval_needed")

        # De-duplicate while preserving order for deterministic payloads.
        return tuple(dict.fromkeys(gaps))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyLedgerRecord":
        policy_id = str(data.get("policy_id"))
        tactic_id = str(data.get("tactic_id"))
        stage = PolicyLedgerStage.from_value(data.get("stage"))
        approvals = tuple(str(value) for value in data.get("approvals", ()))
        evidence_id = data.get("evidence_id")
        threshold_overrides = dict(data.get("threshold_overrides") or {})
        policy_delta_payload = data.get("policy_delta")
        policy_delta = None
        if isinstance(policy_delta_payload, Mapping):
            try:
                policy_delta = PolicyDelta.from_dict(policy_delta_payload)
            except Exception as exc:
                logger.warning(
                    "Skipping policy_delta for %s due to error: %s",
                    policy_id,
                    exc,
                )
        metadata = dict(data.get("metadata") or {})
        created_at_raw = data.get("created_at")
        updated_at_raw = data.get("updated_at")
        created_at = (
            datetime.fromisoformat(created_at_raw)
            if isinstance(created_at_raw, str)
            else datetime.now(tz=UTC)
        )
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        updated_at = (
            datetime.fromisoformat(updated_at_raw)
            if isinstance(updated_at_raw, str)
            else datetime.now(tz=UTC)
        )
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        history_payload = tuple(
            dict(entry)
            for entry in data.get("history", ())
            if isinstance(entry, Mapping)
        )
        return cls(
            policy_id=policy_id,
            tactic_id=tactic_id,
            stage=stage,
            approvals=approvals,
            evidence_id=str(evidence_id) if evidence_id else None,
            threshold_overrides=threshold_overrides,
            policy_delta=policy_delta,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            history=history_payload,
        )


class PolicyLedgerStore:
    """Persistence wrapper providing CRUD for policy ledger records."""

    def __init__(
        self,
        path: str | Path,
        *,
        now: Callable[[], datetime] | None = None,
        lock_timeout: float = 5.0,
        stale_lock_timeout: float = 30.0,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._records: MutableMapping[str, PolicyLedgerRecord] = {}
        self._lock_timeout = max(0.0, float(lock_timeout))
        self._stale_lock_timeout = max(0.0, float(stale_lock_timeout))
        self._load()

    def _load(self) -> None:
        self._records.clear()
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Policy ledger payload at {self._path} is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Policy ledger file must contain a JSON object")
        records_payload = payload.get("records", {})
        if not isinstance(records_payload, Mapping):
            raise ValueError("Policy ledger records payload must be a mapping")
        for policy_id, record_payload in records_payload.items():
            if not isinstance(record_payload, Mapping):
                continue
            try:
                record = PolicyLedgerRecord.from_dict(record_payload)
            except Exception as exc:
                logger.warning("Skipping invalid policy ledger record for %s: %s", policy_id, exc)
                continue
            self._records[policy_id] = record

    def _lock_path(self) -> Path:
        suffix = f"{self._path.suffix}.lock" if self._path.suffix else ".lock"
        return self._path.with_suffix(suffix)

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        if self._lock_timeout == 0:
            # Fail fast if locking disabled by configuration.
            raise TimeoutError("Policy ledger locking disabled via lock_timeout=0")

        lock_path = self._lock_path()
        start = time.monotonic()
        fd: int | None = None

        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"pid={os.getpid()}".encode("ascii", errors="ignore"))
                break
            except FileExistsError:
                now = time.monotonic()
                if now - start >= self._lock_timeout:
                    if self._stale_lock_timeout > 0:
                        try:
                            stat = lock_path.stat()
                        except FileNotFoundError:
                            continue
                        if time.time() - stat.st_mtime >= self._stale_lock_timeout:
                            try:
                                lock_path.unlink()
                            except FileNotFoundError:
                                continue
                            continue
                    raise TimeoutError(
                        f"Failed to acquire policy ledger lock within {self._lock_timeout:.2f}s"
                    )
                time.sleep(0.05)

        try:
            yield
        finally:
            if fd is not None:
                os.close(fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _dump_locked(self) -> None:
        payload = {
            "records": {policy_id: record.as_dict() for policy_id, record in self._records.items()},
            "updated_at": self._now().isoformat(),
        }
        data = json.dumps(payload, indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            delete=False,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_name = handle.name
        os.replace(tmp_name, self._path)

    def _dump(self) -> None:
        with self._exclusive_lock():
            # Reload to merge concurrent updates before writing.
            self._load()
            self._dump_locked()

    def get(self, policy_id: str) -> PolicyLedgerRecord | None:
        return self._records.get(policy_id)

    def iter_records(self) -> tuple[PolicyLedgerRecord, ...]:
        return tuple(self._records.values())

    def upsert(
        self,
        *,
        policy_id: str,
        tactic_id: str,
        stage: PolicyLedgerStage,
        approvals: Iterable[str] = (),
        evidence_id: str | None = None,
        threshold_overrides: Mapping[str, float | str] | None = None,
        policy_delta: PolicyDelta | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PolicyLedgerRecord:
        approvals_tuple = tuple(sorted(str(value) for value in approvals if value))
        delta = _coerce_policy_delta(policy_delta)
        if delta is not None and delta.is_empty():
            delta = None
        if delta is not None and not approvals_tuple:
            raise ValueError("Policy delta updates require reviewer approvals metadata")
        timestamp = self._now()
        with self._exclusive_lock():
            # Refresh in-memory records to avoid clobbering concurrent writers.
            self._load()

            if policy_id in self._records:
                record = self._records[policy_id]
                updated = record.with_stage(
                    stage,
                    approvals=approvals_tuple,
                    evidence_id=evidence_id,
                    threshold_overrides=threshold_overrides,
                    policy_delta=delta,
                    metadata=metadata,
                    timestamp=timestamp,
                )
            else:
                history_entry: MutableMapping[str, Any] = {
                    "prior_stage": None,
                    "next_stage": stage.value,
                    "applied_at": timestamp.isoformat(),
                    "approvals": list(approvals_tuple),
                    "evidence_id": evidence_id,
                }
                if delta is not None:
                    history_entry["policy_delta"] = dict(delta.as_dict())
                updated = PolicyLedgerRecord(
                    policy_id=policy_id,
                    tactic_id=tactic_id,
                    stage=stage,
                    approvals=approvals_tuple,
                    evidence_id=evidence_id,
                    threshold_overrides=dict(threshold_overrides or {}),
                    policy_delta=delta,
                    metadata=dict(metadata or {}),
                    created_at=timestamp,
                    updated_at=timestamp,
                    history=(history_entry,),
                )

            self._records[policy_id] = updated
            self._dump_locked()

        return updated


_DEFAULT_THRESHOLD_PROFILES: Mapping[PolicyLedgerStage, Mapping[str, float | str]] = {
    PolicyLedgerStage.EXPERIMENT: {
        "warn_confidence_floor": 0.85,
        "warn_notional_limit": 10_000.0,
        "block_severity": "warn",
    },
    PolicyLedgerStage.PAPER: {
        "warn_confidence_floor": 0.75,
        "warn_notional_limit": 50_000.0,
        "block_severity": "alert",
    },
    PolicyLedgerStage.PILOT: {
        "warn_confidence_floor": 0.7,
        "warn_notional_limit": 75_000.0,
        "block_severity": "alert",
    },
    PolicyLedgerStage.LIMITED_LIVE: {
        "warn_confidence_floor": 0.6,
        "warn_notional_limit": 100_000.0,
        "block_severity": "alert",
    },
}


class LedgerReleaseManager:
    """Runtime helper that surfaces release posture to execution components."""

    def __init__(
        self,
        store: PolicyLedgerStore,
        *,
        default_stage: PolicyLedgerStage = PolicyLedgerStage.EXPERIMENT,
        stage_thresholds: Mapping[PolicyLedgerStage, Mapping[str, float | str]] | None = None,
        feature_flags: PolicyLedgerFeatureFlags | None = None,
        evidence_resolver: Callable[[str], bool] | None = None,
    ) -> None:
        self._store = store
        self._default_stage = default_stage
        self._stage_thresholds = dict(_DEFAULT_THRESHOLD_PROFILES)
        if stage_thresholds:
            for stage, overrides in stage_thresholds.items():
                self._stage_thresholds[PolicyLedgerStage.from_value(stage)] = dict(overrides)
        self._feature_flags = feature_flags or PolicyLedgerFeatureFlags.from_env()
        self._evidence_resolver = evidence_resolver

    @staticmethod
    def _coverage_stage(record: PolicyLedgerRecord) -> PolicyLedgerStage:
        """Determine the highest stage permitted by audit coverage."""

        if not record.evidence_id:
            return PolicyLedgerStage.EXPERIMENT

        if not record.approvals:
            return PolicyLedgerStage.PAPER

        if len(record.approvals) < 2:
            return PolicyLedgerStage.PILOT

        return PolicyLedgerStage.LIMITED_LIVE

    @staticmethod
    def _audit_gaps(record: PolicyLedgerRecord) -> tuple[str, ...]:
        return record.audit_gaps(expected_stage=record.stage)

    def resolve_stage(self, policy_id: str | None) -> PolicyLedgerStage:
        if not policy_id:
            return self._default_stage
        record = self._store.get(policy_id)
        if record is None:
            return self._default_stage
        coverage_stage = self._coverage_stage(record)
        if _STAGE_ORDER[coverage_stage] < _STAGE_ORDER[record.stage]:
            return coverage_stage
        return record.stage

    def resolve_thresholds(self, policy_id: str | None) -> Mapping[str, float | str]:
        stage = self.resolve_stage(policy_id)
        base = dict(self._stage_thresholds.get(stage, {}))
        record = self._store.get(policy_id) if policy_id else None
        if record is not None and record.threshold_overrides:
            base.update(record.threshold_overrides)
        base["stage"] = stage.value
        return base

    def build_governance_workflow(
        self,
        *,
        regulation: str = "AlphaTrade Governance",
        generated_at: datetime | None = None,
    ) -> "ComplianceWorkflowSnapshot":
        """Construct a governance checklist snapshot from the ledger state."""

        return build_policy_governance_workflow(
            self._store,
            regulation=regulation,
            generated_at=generated_at,
        )

    def describe(self, policy_id: str | None) -> Mapping[str, Any]:
        stage = self.resolve_stage(policy_id)
        thresholds = self.resolve_thresholds(policy_id)
        record = self._store.get(policy_id) if policy_id else None
        payload: MutableMapping[str, Any] = {
            "stage": stage.value,
            "thresholds": dict(thresholds),
        }
        if record is not None:
            coverage_stage = self._coverage_stage(record)
            audit_gaps = self._audit_gaps(record)
            payload["approvals"] = list(record.approvals)
            if record.evidence_id:
                payload["evidence_id"] = record.evidence_id
            payload["updated_at"] = record.updated_at.isoformat()
            payload["declared_stage"] = record.stage.value
            payload["audit_stage"] = coverage_stage.value
            if audit_gaps:
                payload["audit_gaps"] = list(audit_gaps)
            payload["audit_enforced"] = record.stage != stage
        return payload

    def promote(
        self,
        *,
        policy_id: str,
        tactic_id: str,
        stage: PolicyLedgerStage | str,
        approvals: Iterable[str] = (),
        evidence_id: str | None = None,
        threshold_overrides: Mapping[str, float | str] | None = None,
        policy_delta: PolicyDelta | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PolicyLedgerRecord:
        stage_value = PolicyLedgerStage.from_value(stage)
        if self._feature_flags.require_diary_evidence and not evidence_id:
            raise ValueError("DecisionDiary evidence_id is required for ledger promotions")
        if evidence_id and self._evidence_resolver is not None:
            try:
                if not self._evidence_resolver(evidence_id):
                    raise ValueError(
                        f"No DecisionDiary evidence located for {evidence_id}"
                    )
            except Exception as exc:
                raise ValueError(f"Failed to validate DecisionDiary evidence: {exc}") from exc
        record = self._store.upsert(
            policy_id=policy_id,
            tactic_id=tactic_id,
            stage=stage_value,
            approvals=approvals,
            evidence_id=evidence_id,
            threshold_overrides=threshold_overrides,
            policy_delta=policy_delta,
            metadata=metadata,
        )
        logger.info(
            "Policy ledger promotion applied",
            extra={
                "policy_id": policy_id,
                "stage": stage_value.value,
                "approvals": list(record.approvals),
            },
        )
        return record


def build_policy_governance_workflow(
    store: PolicyLedgerStore,
    *,
    regulation: str = "AlphaTrade Governance",
    generated_at: datetime | None = None,
) -> ComplianceWorkflowSnapshot:
    """Build a compliance workflow snapshot linked to policy ledger posture."""

    from src.compliance.workflow import (
        ComplianceWorkflowChecklist,
        ComplianceWorkflowSnapshot,
        ComplianceWorkflowTask,
        WorkflowTaskStatus,
    )

    workflow_status_order: Mapping[WorkflowTaskStatus, int] = {
        WorkflowTaskStatus.completed: 0,
        WorkflowTaskStatus.in_progress: 1,
        WorkflowTaskStatus.todo: 2,
        WorkflowTaskStatus.blocked: 3,
    }

    def _resolve_task_status(record: PolicyLedgerRecord) -> WorkflowTaskStatus:
        if not record.evidence_id:
            return WorkflowTaskStatus.blocked
        if not record.approvals:
            return WorkflowTaskStatus.todo
        if len(record.approvals) < 2:
            return WorkflowTaskStatus.in_progress
        if record.policy_delta is None or record.policy_delta.is_empty():
            return WorkflowTaskStatus.in_progress
        return WorkflowTaskStatus.completed

    generated_ts = generated_at or datetime.now(tz=UTC)
    records = sorted(
        store.iter_records(),
        key=lambda record: (_STAGE_ORDER.get(record.stage, 0), record.policy_id),
    )
    tasks: list[ComplianceWorkflowTask] = []
    for record in records:
        status = _resolve_task_status(record)
        summary_parts = [f"tactic={record.tactic_id}"]
        if record.approvals:
            summary_parts.append("approvals=" + ",".join(record.approvals))
        if record.evidence_id:
            summary_parts.append(f"evidence={record.evidence_id}")
        else:
            summary_parts.append("evidence=missing")
        if record.policy_delta and record.policy_delta.regime:
            summary_parts.append(f"regime={record.policy_delta.regime}")
        severity = "high" if status is WorkflowTaskStatus.blocked else "medium"
        metadata: dict[str, Any] = {
            "policy_id": record.policy_id,
            "stage": record.stage.value,
            "approvals": list(record.approvals),
            "evidence_id": record.evidence_id,
            "updated_at": record.updated_at.isoformat(),
        }
        if record.policy_delta is not None and not record.policy_delta.is_empty():
            metadata["policy_delta"] = dict(record.policy_delta.as_dict())
        task = ComplianceWorkflowTask(
            task_id=f"policy::{record.policy_id}",
            title=f"{record.policy_id} → {record.stage.value}",
            status=status,
            summary="; ".join(summary_parts),
            severity=severity,
            metadata=metadata,
        )
        tasks.append(task)

    if tasks:
        workflow_status = max(tasks, key=lambda task: workflow_status_order[task.status]).status
    else:
        workflow_status = WorkflowTaskStatus.todo

    checklist_metadata: dict[str, Any] = {
        "policies": len(records),
        "completed": sum(1 for task in tasks if task.status is WorkflowTaskStatus.completed),
        "blocked": sum(1 for task in tasks if task.status is WorkflowTaskStatus.blocked),
    }
    checklist = ComplianceWorkflowChecklist(
        name="Policy Ledger Governance",
        regulation=regulation,
        status=workflow_status,
        tasks=tuple(tasks),
        metadata=checklist_metadata,
    )
    snapshot_metadata = {
        "source": "policy_ledger",
        "policy_count": len(records),
    }
    return ComplianceWorkflowSnapshot(
        status=workflow_status,
        generated_at=generated_ts,
        workflows=(checklist,),
        metadata=snapshot_metadata,
    )


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True, frozen=True)
class PolicyDelta:
    """Delta applied to risk policy and router guardrails for a given regime."""

    regime: str | None = None
    regime_confidence: float | None = None
    risk_config: Mapping[str, Any] = field(default_factory=dict)
    router_guardrails: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "risk_config": dict(self.risk_config),
            "router_guardrails": dict(self.router_guardrails),
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if self.regime is not None:
            payload["regime"] = self.regime
        if self.regime_confidence is not None:
            payload["regime_confidence"] = float(self.regime_confidence)
        return payload

    def is_empty(self) -> bool:
        return not (self.risk_config or self.router_guardrails or self.notes or self.metadata)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyDelta":
        regime = payload.get("regime")
        regime_confidence = payload.get("regime_confidence")
        if isinstance(regime_confidence, str):
            try:
                regime_confidence = float(regime_confidence)
            except ValueError:
                regime_confidence = None
        elif regime_confidence is not None:
            try:
                regime_confidence = float(regime_confidence)
            except (TypeError, ValueError):
                regime_confidence = None
        risk_config = dict(payload.get("risk_config") or {})
        router_guardrails = dict(payload.get("router_guardrails") or {})
        notes_payload = payload.get("notes")
        if isinstance(notes_payload, Sequence) and not isinstance(notes_payload, (str, bytes)):
            notes = tuple(str(item).strip() for item in notes_payload if str(item).strip())
        else:
            notes = ()
        metadata = dict(payload.get("metadata") or {})
        return cls(
            regime=str(regime) if regime is not None else None,
            regime_confidence=regime_confidence if isinstance(regime_confidence, float) else None,
            risk_config=risk_config,
            router_guardrails=router_guardrails,
            notes=notes,
            metadata=metadata,
        )


@dataclass(slots=True, frozen=True)
class PolicyLedgerFeatureFlags:
    """Feature switches guarding policy ledger promotions."""

    require_diary_evidence: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "PolicyLedgerFeatureFlags":
        env = env or os.environ
        require_diary_env = env.get("POLICY_LEDGER_REQUIRE_DIARY")
        require_diary = _parse_bool(require_diary_env) if require_diary_env is not None else True
        return cls(require_diary_evidence=require_diary)


def _coerce_policy_delta(
    delta: PolicyDelta | Mapping[str, Any] | None,
) -> PolicyDelta | None:
    if delta is None:
        return None
    if isinstance(delta, PolicyDelta):
        return delta
    if isinstance(delta, Mapping):
        return PolicyDelta.from_dict(delta)
    raise TypeError("policy_delta must be a mapping or PolicyDelta instance")
