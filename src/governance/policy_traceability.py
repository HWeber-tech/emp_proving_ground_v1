"""Traceability helpers for policy ledger promotions.

This module packages governance trace metadata so every policy promotion records
its supporting DecisionDiary slice alongside the code and configuration hashes
that produced the proposal.  The roadmap requires that governance sign-off can
always recover the precise diary context, binary provenance, and config posture
for each ledger entry.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from src.governance.policy_ledger import PolicyDelta, PolicyLedgerStage
from src.understanding.decision_diary import DecisionDiaryEntry, DecisionDiaryStore

__all__ = [
    "PolicyTraceability",
    "build_traceability_metadata",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PolicyTraceability:
    """Structured trace payload attached to policy ledger promotions."""

    policy_id: str
    tactic_id: str
    stage: str
    evidence_id: str
    code_hash: str
    config_hash: str
    diary_slice: Mapping[str, Any]
    generated_at: _dt.datetime

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "policy_id": self.policy_id,
            "tactic_id": self.tactic_id,
            "stage": self.stage,
            "evidence_id": self.evidence_id,
            "code_hash": self.code_hash,
            "config_hash": self.config_hash,
            "generated_at": self.generated_at.astimezone(_dt.timezone.utc).isoformat(),
            "diary_slice": dict(self.diary_slice),
        }
        return payload


_CODE_HASH_CACHE: str | None = None


def _discover_repo_root() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _resolve_code_hash() -> str:
    global _CODE_HASH_CACHE
    if _CODE_HASH_CACHE is not None:
        return _CODE_HASH_CACHE

    repo_root = _discover_repo_root()
    if repo_root is None:
        _CODE_HASH_CACHE = "unknown"
        logger.debug("Unable to resolve repository root for code hash")
        return _CODE_HASH_CACHE

    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        value = output.decode("utf-8").strip()
        _CODE_HASH_CACHE = value or "unknown"
    except Exception:
        logger.debug("Failed to resolve git code hash", exc_info=True)
        _CODE_HASH_CACHE = "unknown"
    return _CODE_HASH_CACHE


def _normalise_sort_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:  # pragma: no cover - fallback for non-serialisable data
        return repr(value)


def _normalise(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalised: MutableMapping[str, Any] = {}
        for key, payload in value.items():
            normalised[str(key)] = _normalise(payload)
        return dict(sorted(normalised.items()))
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if isinstance(value, set):
        normalised_items = [_normalise(item) for item in value]
        return sorted(normalised_items, key=_normalise_sort_key)
    if isinstance(value, _dt.datetime):
        return value.astimezone(_dt.timezone.utc).isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "as_dict") and callable(value.as_dict):  # type: ignore[attr-defined]
        try:
            return _normalise(value.as_dict())
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
    if hasattr(value, "dict") and callable(value.dict):  # type: ignore[attr-defined]
        try:
            return _normalise(value.dict())
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _delta_payload(policy_delta: PolicyDelta | Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if policy_delta is None:
        return None
    if isinstance(policy_delta, PolicyDelta):
        if policy_delta.is_empty():
            return None
        return policy_delta.as_dict()
    if isinstance(policy_delta, Mapping):
        payload = dict(policy_delta)
        if not payload:
            return None
        return payload
    return None


def _compute_config_hash(
    *,
    stage: PolicyLedgerStage,
    threshold_overrides: Mapping[str, Any] | None,
    policy_delta: PolicyDelta | Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> str:
    delta_payload = _delta_payload(policy_delta)
    payload = {
        "stage": stage.value,
        "threshold_overrides": _normalise(threshold_overrides or {}),
        "policy_delta": _normalise(delta_payload) if delta_payload is not None else None,
        "metadata": _normalise(metadata or {}),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_diary_slice(
    *,
    evidence_id: str,
    stage: PolicyLedgerStage,
    entry: DecisionDiaryEntry | None,
    diary_path: Path | None,
    store_available: bool,
) -> Mapping[str, Any]:
    slice_payload: MutableMapping[str, Any] = {
        "entry_id": evidence_id,
        "stage": stage.value,
    }
    if diary_path is not None:
        slice_payload["diary_path"] = diary_path.as_posix()
    if not store_available:
        slice_payload["status"] = "store_unavailable"
        return slice_payload
    if entry is None:
        slice_payload["status"] = "missing_entry"
        return slice_payload
    slice_payload["status"] = "ok"
    entry_payload = entry.as_dict()
    slice_payload["entry"] = entry_payload
    return slice_payload


def build_traceability_metadata(
    *,
    policy_id: str,
    tactic_id: str,
    stage: PolicyLedgerStage,
    evidence_id: str | None,
    diary_store: DecisionDiaryStore | None,
    diary_path: Path | None,
    threshold_overrides: Mapping[str, Any] | None,
    policy_delta: PolicyDelta | Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return a traceability payload for the promotion, if evidence is present."""

    evidence = (evidence_id or "").strip()
    if not evidence:
        return None

    entry: DecisionDiaryEntry | None = None
    if diary_store is not None:
        try:
            entry = diary_store.get(evidence)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to fetch diary entry %s", evidence, exc_info=True)
            entry = None
    diary_slice = _build_diary_slice(
        evidence_id=evidence,
        stage=stage,
        entry=entry,
        diary_path=diary_path,
        store_available=diary_store is not None,
    )
    trace = PolicyTraceability(
        policy_id=policy_id,
        tactic_id=tactic_id,
        stage=stage.value,
        evidence_id=evidence,
        code_hash=_resolve_code_hash(),
        config_hash=_compute_config_hash(
            stage=stage,
            threshold_overrides=threshold_overrides,
            policy_delta=policy_delta,
            metadata=metadata,
        ),
        diary_slice=diary_slice,
        generated_at=_dt.datetime.now(tz=_dt.timezone.utc),
    )
    return trace.as_dict()
