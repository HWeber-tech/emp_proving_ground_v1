"""Rebuild risk policy and router guardrail artifacts from the policy ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.config.risk.risk_config import RiskConfig

from .policy_ledger import (
    LedgerReleaseManager,
    PolicyDelta,
    PolicyLedgerRecord,
    PolicyLedgerStage,
    PolicyLedgerStore,
)


@dataclass(frozen=True)
class PolicyRebuildArtifact:
    """Materialised policy artifact reconstructed from ledger promotions."""

    policy_id: str
    tactic_id: str
    stage: PolicyLedgerStage
    risk_config: RiskConfig
    router_guardrails: Mapping[str, Any]
    approvals: tuple[str, ...]
    evidence_id: str | None
    thresholds: Mapping[str, float | str]
    metadata: Mapping[str, Any]
    policy_delta: PolicyDelta | None
    history: Sequence[Mapping[str, Any]]
    updated_at: datetime

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "policy_id": self.policy_id,
            "tactic_id": self.tactic_id,
            "stage": self.stage.value,
            "approvals": list(self.approvals),
            "risk_config": _serialise_risk_config(self.risk_config),
            "router_guardrails": dict(self.router_guardrails),
            "thresholds": dict(self.thresholds),
            "history": [dict(entry) for entry in self.history],
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }
        if self.evidence_id:
            payload["evidence_id"] = self.evidence_id
        if self.policy_delta is not None and not self.policy_delta.is_empty():
            payload["policy_delta"] = dict(self.policy_delta.as_dict())
        return payload


def _serialise_risk_config(config: RiskConfig) -> Mapping[str, Any]:
    raw = config.dict()
    serialised: MutableMapping[str, Any] = {}
    for key, value in raw.items():
        if hasattr(value, "__float__"):
            try:
                serialised[key] = float(value)
                continue
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                pass
        serialised[key] = value
    return serialised


def _iter_policy_deltas(record: PolicyLedgerRecord) -> Iterable[Mapping[str, Any]]:
    for entry in record.history:
        payload = entry.get("policy_delta")
        if isinstance(payload, Mapping):
            yield payload
    if record.policy_delta is not None and not record.policy_delta.is_empty():
        yield record.policy_delta.as_dict()


def _apply_risk_config(base: RiskConfig, overrides: Mapping[str, Any]) -> RiskConfig:
    if not overrides:
        return base
    payload = dict(base.dict())
    payload.update(overrides)
    return RiskConfig(**payload)


def _merge_guardrails(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Mapping[str, Any]:
    merged = dict(base)
    merged.update(overrides)
    return merged


def rebuild_policy_artifacts(
    store: PolicyLedgerStore,
    *,
    base_config: RiskConfig | None = None,
    default_guardrails: Mapping[str, Any] | None = None,
) -> tuple[PolicyRebuildArtifact, ...]:
    """Replay ledger entries to rebuild enforceable policy artefacts."""

    base_config = base_config or RiskConfig()
    default_guardrails = dict(default_guardrails or {})
    manager = LedgerReleaseManager(store)

    artifacts: list[PolicyRebuildArtifact] = []
    records = sorted(
        store.iter_records(),
        key=lambda record: record.updated_at,
    )
    for record in records:
        config = base_config
        guardrails = dict(default_guardrails)
        for delta_payload in _iter_policy_deltas(record):
            delta_config = dict(delta_payload.get("risk_config") or {})
            if delta_config:
                config = _apply_risk_config(config, delta_config)
            guardrail_overrides = dict(delta_payload.get("router_guardrails") or {})
            if guardrail_overrides:
                guardrails = _merge_guardrails(guardrails, guardrail_overrides)
        thresholds = manager.resolve_thresholds(record.policy_id)
        artifact = PolicyRebuildArtifact(
            policy_id=record.policy_id,
            tactic_id=record.tactic_id,
            stage=record.stage,
            risk_config=config,
            router_guardrails=guardrails,
            approvals=record.approvals,
            evidence_id=record.evidence_id,
            thresholds=thresholds,
            metadata=record.metadata,
            policy_delta=record.policy_delta,
            history=record.history,
            updated_at=record.updated_at,
        )
        artifacts.append(artifact)
    return tuple(artifacts)


__all__ = ["PolicyRebuildArtifact", "rebuild_policy_artifacts"]
