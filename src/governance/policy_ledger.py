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

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping

import json
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "PolicyLedgerStage",
    "PolicyLedgerRecord",
    "PolicyLedgerStore",
    "LedgerReleaseManager",
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
        metadata: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> "PolicyLedgerRecord":
        _validate_stage_progress(self.stage, stage)
        applied_timestamp = timestamp or datetime.now(tz=UTC)
        history_entry = {
            "prior_stage": self.stage.value,
            "next_stage": stage.value,
            "applied_at": applied_timestamp.isoformat(),
            "approvals": sorted(tuple(approvals or self.approvals)),
            "evidence_id": evidence_id or self.evidence_id,
        }
        new_history = self.history + (history_entry,)
        return PolicyLedgerRecord(
            policy_id=self.policy_id,
            tactic_id=self.tactic_id,
            stage=stage,
            approvals=tuple(sorted(tuple(approvals or self.approvals))),
            evidence_id=evidence_id or self.evidence_id,
            threshold_overrides=dict(threshold_overrides or self.threshold_overrides),
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
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyLedgerRecord":
        policy_id = str(data.get("policy_id"))
        tactic_id = str(data.get("tactic_id"))
        stage = PolicyLedgerStage.from_value(data.get("stage"))
        approvals = tuple(str(value) for value in data.get("approvals", ()))
        evidence_id = data.get("evidence_id")
        threshold_overrides = dict(data.get("threshold_overrides") or {})
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
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._records: MutableMapping[str, PolicyLedgerRecord] = {}
        self._load()

    def _load(self) -> None:
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

    def _dump(self) -> None:
        payload = {
            "records": {policy_id: record.as_dict() for policy_id, record in self._records.items()},
            "updated_at": self._now().isoformat(),
        }
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True))

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
        metadata: Mapping[str, Any] | None = None,
    ) -> PolicyLedgerRecord:
        approvals_tuple = tuple(sorted(str(value) for value in approvals if value))
        timestamp = self._now()
        if policy_id in self._records:
            record = self._records[policy_id]
            updated = record.with_stage(
                stage,
                approvals=approvals_tuple,
                evidence_id=evidence_id,
                threshold_overrides=threshold_overrides,
                metadata=metadata,
                timestamp=timestamp,
            )
        else:
            updated = PolicyLedgerRecord(
                policy_id=policy_id,
                tactic_id=tactic_id,
                stage=stage,
                approvals=approvals_tuple,
                evidence_id=evidence_id,
                threshold_overrides=dict(threshold_overrides or {}),
                metadata=dict(metadata or {}),
                created_at=timestamp,
                updated_at=timestamp,
                history=(),
            )
        self._records[policy_id] = updated
        self._dump()
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
    ) -> None:
        self._store = store
        self._default_stage = default_stage
        self._stage_thresholds = dict(_DEFAULT_THRESHOLD_PROFILES)
        if stage_thresholds:
            for stage, overrides in stage_thresholds.items():
                self._stage_thresholds[PolicyLedgerStage.from_value(stage)] = dict(overrides)

    def resolve_stage(self, policy_id: str | None) -> PolicyLedgerStage:
        if not policy_id:
            return self._default_stage
        record = self._store.get(policy_id)
        if record is None:
            return self._default_stage
        return record.stage

    def resolve_thresholds(self, policy_id: str | None) -> Mapping[str, float | str]:
        stage = self.resolve_stage(policy_id)
        base = dict(self._stage_thresholds.get(stage, {}))
        record = self._store.get(policy_id) if policy_id else None
        if record is not None and record.threshold_overrides:
            base.update(record.threshold_overrides)
        base["stage"] = stage.value
        return base

    def describe(self, policy_id: str | None) -> Mapping[str, Any]:
        stage = self.resolve_stage(policy_id)
        thresholds = self.resolve_thresholds(policy_id)
        record = self._store.get(policy_id) if policy_id else None
        payload: MutableMapping[str, Any] = {
            "stage": stage.value,
            "thresholds": dict(thresholds),
        }
        if record is not None:
            payload["approvals"] = list(record.approvals)
            if record.evidence_id:
                payload["evidence_id"] = record.evidence_id
            payload["updated_at"] = record.updated_at.isoformat()
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
        metadata: Mapping[str, Any] | None = None,
    ) -> PolicyLedgerRecord:
        stage_value = PolicyLedgerStage.from_value(stage)
        record = self._store.upsert(
            policy_id=policy_id,
            tactic_id=tactic_id,
            stage=stage_value,
            approvals=approvals,
            evidence_id=evidence_id,
            threshold_overrides=threshold_overrides,
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


__all__ = [
    "PolicyLedgerStage",
    "PolicyLedgerRecord",
    "PolicyLedgerStore",
    "LedgerReleaseManager",
]
