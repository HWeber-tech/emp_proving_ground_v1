"""Helpers to reconstruct policy phenotypes from the governance ledger.

`PolicyRebuildArtifact` captures the materialised configuration produced by
`rebuild_policy_artifacts`.  This module layers deterministic hashing and
selection helpers on top so operators can reference immutable phenotypes when
replaying ledger state or comparing runtime payloads.

The roadmap item for `make rebuild-policy HASH=...` requires a reproducible
artifact keyed by a hash so we can locate the exact policy posture registered in
the ledger.  The hash is derived from a canonical snapshot of the risk config,
guardrails, thresholds, approvals, and supporting metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, MutableMapping, Sequence

from src.config.risk.risk_config import RiskConfig

from .policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from .policy_rebuilder import PolicyRebuildArtifact, rebuild_policy_artifacts

__all__ = [
    "PolicyPhenotype",
    "build_policy_phenotypes",
    "select_policy_phenotype",
]


def _serialise_risk_config(config: RiskConfig) -> Mapping[str, Any]:
    """Coerce a `RiskConfig` into plain JSON-serialisable types."""

    payload: MutableMapping[str, Any] = {}
    for key, value in config.dict().items():  # type: ignore[no-untyped-call]
        if hasattr(value, "__float__"):
            try:
                payload[key] = float(value)
                continue
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                pass
        payload[key] = value
    return payload


def _hash_payload(payload: Mapping[str, Any]) -> str:
    """Generate a deterministic hash for the phenotype payload."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyPhenotype:
    """Digestible representation of a policy reconstructed from the ledger."""

    policy_id: str
    policy_hash: str
    tactic_id: str
    stage: PolicyLedgerStage
    approvals: tuple[str, ...]
    evidence_id: str | None
    risk_config: Mapping[str, Any]
    router_guardrails: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    metadata: Mapping[str, Any]
    history: Sequence[Mapping[str, Any]]
    updated_at: datetime

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "tactic_id": self.tactic_id,
            "stage": self.stage.value,
            "approvals": list(self.approvals),
            "risk_config": dict(self.risk_config),
            "router_guardrails": dict(self.router_guardrails),
            "thresholds": dict(self.thresholds),
            "metadata": dict(self.metadata),
            "history": [dict(entry) for entry in self.history],
            "updated_at": self.updated_at.isoformat(),
        }
        if self.evidence_id:
            payload["evidence_id"] = self.evidence_id
        return payload

    @classmethod
    def from_artifact(cls, artifact: PolicyRebuildArtifact) -> "PolicyPhenotype":
        risk_config = _serialise_risk_config(artifact.risk_config)
        router_guardrails = dict(artifact.router_guardrails)
        thresholds = dict(artifact.thresholds)
        metadata = dict(artifact.metadata)
        approvals = tuple(artifact.approvals)
        history = tuple(dict(entry) for entry in artifact.history)

        hash_payload: MutableMapping[str, Any] = {
            "policy_id": artifact.policy_id,
            "tactic_id": artifact.tactic_id,
            "stage": artifact.stage.value,
            "approvals": list(approvals),
            "risk_config": risk_config,
            "router_guardrails": router_guardrails,
            "thresholds": thresholds,
            "metadata": metadata,
            "history": list(history),
        }
        if artifact.evidence_id:
            hash_payload["evidence_id"] = artifact.evidence_id

        policy_hash = _hash_payload(hash_payload)

        return cls(
            policy_id=artifact.policy_id,
            policy_hash=policy_hash,
            tactic_id=artifact.tactic_id,
            stage=artifact.stage,
            approvals=approvals,
            evidence_id=artifact.evidence_id,
            risk_config=risk_config,
            router_guardrails=router_guardrails,
            thresholds=thresholds,
            metadata=metadata,
            history=history,
            updated_at=artifact.updated_at,
        )


def build_policy_phenotypes(
    store: PolicyLedgerStore,
    *,
    base_config: RiskConfig | None = None,
    default_guardrails: Mapping[str, Any] | None = None,
) -> tuple[PolicyPhenotype, ...]:
    """Replay the ledger and convert rebuild artifacts into phenotypes."""

    artifacts = rebuild_policy_artifacts(
        store,
        base_config=base_config,
        default_guardrails=default_guardrails,
    )
    return tuple(PolicyPhenotype.from_artifact(artifact) for artifact in artifacts)


def select_policy_phenotype(
    phenotypes: Sequence[PolicyPhenotype],
    *,
    policy_hash: str | None = None,
    policy_id: str | None = None,
) -> PolicyPhenotype:
    """Locate a phenotype by hash or policy identifier.

    Hash lookup takes precedence when both selectors are provided.
    """

    if policy_hash:
        normalised = policy_hash.strip().lower()
        for phenotype in phenotypes:
            if phenotype.policy_hash.lower() == normalised:
                return phenotype
        raise LookupError(f"No policy phenotype found for hash {policy_hash!r}")

    if policy_id:
        normalised_id = policy_id.strip()
        for phenotype in phenotypes:
            if phenotype.policy_id == normalised_id:
                return phenotype
        raise LookupError(f"No policy phenotype found for policy_id {policy_id!r}")

    raise ValueError("Either policy_hash or policy_id must be provided")

