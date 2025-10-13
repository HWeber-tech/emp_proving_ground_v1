"""Rebuild trading runtime configuration deterministically from the policy ledger.

The governance roadmap requires operators to regenerate the exact runtime
configuration that promoted a strategy.  This module layers deterministic JSON
serialisation on top of the policy phenotype helpers so the resulting payload is
byte-identical every time ``rebuild_strategy`` is invoked for a given
``policy_hash``.

The helper mirrors the ``rebuild_policy`` pipeline by replaying the governance
ledger, selecting the requested phenotype by its immutable hash, and producing a
canonical JSON document that captures the risk configuration, router guardrails,
thresholds, approvals, and supporting metadata.  The JSON bytes are hashed using
SHA-256 so callers can verify integrity or persist the blob alongside other
artifacts without worrying about hidden ordering differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from src.config.risk.risk_config import RiskConfig
from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.governance.policy_phenotype import (
    PolicyPhenotype,
    build_policy_phenotypes,
    select_policy_phenotype,
)

__all__ = [
    "StrategyRuntimeConfig",
    "rebuild_strategy",
]


_DEFAULT_LEDGER_PATH = Path("artifacts/governance/policy_ledger.json")


def _coerce_risk_config(value: RiskConfig | Mapping[str, Any] | None) -> RiskConfig | None:
    """Accept ``RiskConfig`` instances or simple mappings for convenience."""

    if value is None:
        return None
    if isinstance(value, RiskConfig):
        return value
    if isinstance(value, Mapping):
        return RiskConfig(**dict(value))
    raise TypeError("base_config must be a RiskConfig or mapping")


def _canonical_payload(phenotype: PolicyPhenotype) -> Mapping[str, Any]:
    """Convert a phenotype into the canonical runtime configuration payload."""

    payload: MutableMapping[str, Any] = {
        "policy_id": phenotype.policy_id,
        "policy_hash": phenotype.policy_hash,
        "tactic_id": phenotype.tactic_id,
        "stage": phenotype.stage.value,
        "approvals": list(phenotype.approvals),
        "risk_config": dict(phenotype.risk_config),
        "router_guardrails": dict(phenotype.router_guardrails),
        "thresholds": dict(phenotype.thresholds),
        "metadata": dict(phenotype.metadata),
        "history": [dict(entry) for entry in phenotype.history],
        "updated_at": phenotype.updated_at.isoformat(),
    }
    if phenotype.evidence_id:
        payload["evidence_id"] = phenotype.evidence_id
    return payload


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialise payload to canonical UTF-8 JSON bytes."""

    text = json.dumps(  # ``sort_keys`` guarantees deterministic byte order.
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return text.encode("utf-8")


@dataclass(frozen=True, slots=True)
class StrategyRuntimeConfig:
    """Byte-identical runtime configuration regenerated from the ledger."""

    policy_id: str
    policy_hash: str
    tactic_id: str
    stage: PolicyLedgerStage
    approvals: tuple[str, ...]
    risk_config: Mapping[str, Any]
    router_guardrails: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    metadata: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    updated_at: datetime
    evidence_id: str | None
    payload: Mapping[str, Any]
    json_bytes: bytes
    digest: str

    def as_dict(self) -> Mapping[str, Any]:
        """Return the canonical payload as a plain mapping."""

        return dict(self.payload)

    def json(self, *, indent: int | None = None) -> str:
        """Return the configuration as a JSON string.

        When ``indent`` is omitted the method reuses the byte-identical payload
        generated during construction.  Providing an ``indent`` re-serialises the
        payload with pretty-printing while preserving key ordering guarantees.
        """

        if indent is None:
            return self.json_bytes.decode("utf-8")
        return json.dumps(self.payload, sort_keys=True, indent=indent)


def rebuild_strategy(
    policy_hash: str,
    *,
    store: PolicyLedgerStore | None = None,
    ledger_path: str | Path | None = None,
    base_config: RiskConfig | Mapping[str, Any] | None = None,
    default_guardrails: Mapping[str, Any] | None = None,
) -> StrategyRuntimeConfig:
    """Rebuild a strategy's runtime configuration from the governance ledger.

    Parameters
    ----------
    policy_hash:
        Immutable hash emitted by the policy phenotype pipeline.
    store:
        Optional pre-instantiated ``PolicyLedgerStore``.  When omitted the
        function loads the store from ``ledger_path`` (or the default artifacts
        location).
    ledger_path:
        Location of the ledger JSON artifact used when ``store`` is not
        provided.
    base_config:
        Optional baseline ``RiskConfig`` (or mapping) applied prior to replaying
        ledger deltas.
    default_guardrails:
        Optional router guardrail defaults merged before applying recorded
        policy deltas.

    Returns
    -------
    StrategyRuntimeConfig
        Dataclass containing the canonical payload, canonical JSON bytes, and
        the SHA-256 digest for integrity checks.
    """

    if not policy_hash or not policy_hash.strip():
        raise ValueError("policy_hash must be provided")

    ledger_store = store
    if ledger_store is None:
        path = Path(ledger_path) if ledger_path is not None else _DEFAULT_LEDGER_PATH
        ledger_store = PolicyLedgerStore(path)

    phenotypes = build_policy_phenotypes(
        ledger_store,
        base_config=_coerce_risk_config(base_config),
        default_guardrails=default_guardrails,
    )
    phenotype = select_policy_phenotype(phenotypes, policy_hash=policy_hash)

    payload = _canonical_payload(phenotype)
    json_bytes = _canonical_json_bytes(payload)
    digest = sha256(json_bytes).hexdigest()

    history: tuple[Mapping[str, Any], ...] = tuple(dict(entry) for entry in phenotype.history)

    return StrategyRuntimeConfig(
        policy_id=phenotype.policy_id,
        policy_hash=phenotype.policy_hash,
        tactic_id=phenotype.tactic_id,
        stage=phenotype.stage,
        approvals=tuple(phenotype.approvals),
        risk_config=dict(phenotype.risk_config),
        router_guardrails=dict(phenotype.router_guardrails),
        thresholds=dict(phenotype.thresholds),
        metadata=dict(phenotype.metadata),
        history=history,
        updated_at=phenotype.updated_at,
        evidence_id=phenotype.evidence_id,
        payload=payload,
        json_bytes=json_bytes,
        digest=digest,
    )
