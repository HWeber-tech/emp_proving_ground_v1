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

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
import math
import numbers
from pathlib import Path
import re
from typing import Any, Mapping, MutableMapping, Sequence, cast

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


_DOCKER_TAG_MAX_LENGTH = 128
_DOCKER_TAG_SANITISE_PATTERN = re.compile(r"[^a-z0-9_.-]")


def _sanitise_tag(value: str) -> str:
    """Normalise arbitrary text into a Docker tag-safe token."""

    lowered = value.strip().lower()
    if not lowered:
        return "unknown"
    sanitised = _DOCKER_TAG_SANITISE_PATTERN.sub("-", lowered)
    trimmed = sanitised[:_DOCKER_TAG_MAX_LENGTH]
    return trimmed or "unknown"


def _build_docker_image_tags(policy_hash: str, config_fingerprint: str) -> Mapping[str, str]:
    """Return reproducible Docker tags derived from governance artifacts."""

    policy_token = _sanitise_tag(policy_hash)
    fingerprint_token = _sanitise_tag(config_fingerprint)
    short_policy = policy_token[:12] or policy_token
    short_fingerprint = fingerprint_token[:12] or fingerprint_token

    return {
        "policy_hash": f"policy-{policy_token}",
        "config_fingerprint": f"config-{fingerprint_token}",
        "release": f"release-{short_policy}-{short_fingerprint}",
    }


def _coerce_risk_config(value: RiskConfig | Mapping[str, Any] | None) -> RiskConfig | None:
    """Accept ``RiskConfig`` instances or simple mappings for convenience."""

    if value is None:
        return None
    if isinstance(value, RiskConfig):
        return value
    if isinstance(value, Mapping):
        return RiskConfig(**dict(value))
    raise TypeError("base_config must be a RiskConfig or mapping")


def _sort_key(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return repr(value)


def _normalise_json_value(value: Any) -> Any:
    """Normalise heterogenous values into JSON-stable primitives."""

    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Decimal values must be finite for canonical JSON payloads")
        return float(value)

    if isinstance(value, numbers.Integral):  # Handles ints without converting bools (already returned)
        return int(value)

    if isinstance(value, numbers.Real):
        normalised = float(value)
        if not math.isfinite(normalised):
            raise ValueError("Float values must be finite for canonical JSON payloads")
        return normalised

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Enum):
        return _normalise_json_value(value.value)

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return _normalise_json_value(asdict(value))

    if hasattr(value, "_asdict"):
        try:
            return _normalise_json_value(value._asdict())  # type: ignore[attr-defined]
        except TypeError:
            pass

    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for key, item in value.items():
            key_str = str(key)
            items.append((key_str, _normalise_json_value(item)))
        items.sort(key=lambda pair: pair[0])
        return {key: item for key, item in items}

    if isinstance(value, (set, frozenset)):
        normalised_items = [_normalise_json_value(item) for item in value]
        return sorted(normalised_items, key=_sort_key)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_json_value(item) for item in value]

    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _normalise_json_value(vars(value))

    raise TypeError(f"Unsupported value type {type(value)!r} for canonical runtime payloads")


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
    return cast(Mapping[str, Any], _normalise_json_value(payload))


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
    runtime_config: Mapping[str, Any]
    runtime_json_bytes: bytes
    runtime_digest: str
    runtime_config_path: str
    docker_image_tags: Mapping[str, str]
    payload: Mapping[str, Any]
    json_bytes: bytes
    digest: str

    def as_dict(self) -> Mapping[str, Any]:
        """Return the canonical payload as a plain mapping."""

        canonical_text = self.json_bytes.decode("utf-8")
        return cast(Mapping[str, Any], json.loads(canonical_text))

    def json(self, *, indent: int | None = None) -> str:
        """Return the configuration as a JSON string.

        When ``indent`` is omitted the method reuses the byte-identical payload
        generated during construction.  Providing an ``indent`` re-serialises the
        payload with pretty-printing while preserving key ordering guarantees.
        """

        canonical_text = self.json_bytes.decode("utf-8")
        if indent is None:
            return canonical_text
        payload_dict = json.loads(canonical_text)
        return json.dumps(payload_dict, sort_keys=True, indent=indent, ensure_ascii=False)


def rebuild_strategy(
    policy_hash: str,
    *,
    store: PolicyLedgerStore | None = None,
    ledger_path: str | Path | None = None,
    base_config: RiskConfig | Mapping[str, Any] | None = None,
    default_guardrails: Mapping[str, Any] | None = None,
    runtime_output_path: str | Path | None = None,
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
        ledger_source_path = path
    else:
        ledger_source_path = ledger_store.path

    phenotypes = build_policy_phenotypes(
        ledger_store,
        base_config=_coerce_risk_config(base_config),
        default_guardrails=default_guardrails,
    )
    phenotype = select_policy_phenotype(phenotypes, policy_hash=policy_hash)

    runtime_output = (
        Path(runtime_output_path)
        if runtime_output_path is not None
        else ledger_source_path.with_name("runtime_config.json")
    ).expanduser().resolve()

    runtime_payload_raw = _canonical_payload(phenotype)
    runtime_json_bytes = _canonical_json_bytes(runtime_payload_raw)
    runtime_payload = cast(Mapping[str, Any], json.loads(runtime_json_bytes.decode("utf-8")))
    runtime_digest = sha256(runtime_json_bytes).hexdigest()

    enriched_payload: MutableMapping[str, Any] = dict(runtime_payload)
    enriched_payload["runtime_config"] = runtime_payload
    enriched_payload["runtime_digest"] = runtime_digest
    enriched_payload["runtime_config_path"] = str(runtime_output)
    enriched_payload["docker_image_tags"] = _build_docker_image_tags(
        phenotype.policy_hash,
        runtime_digest,
    )

    json_bytes = _canonical_json_bytes(enriched_payload)
    digest = sha256(json_bytes).hexdigest()

    approvals_payload = cast(Sequence[Any], runtime_payload["approvals"])
    risk_config_payload = cast(Mapping[str, Any], runtime_payload["risk_config"])
    guardrails_payload = cast(Mapping[str, Any], runtime_payload["router_guardrails"])
    thresholds_payload = cast(Mapping[str, Any], runtime_payload["thresholds"])
    metadata_payload = cast(Mapping[str, Any], runtime_payload["metadata"])
    history_payload = cast(Sequence[Any], runtime_payload["history"])

    approvals = tuple(str(item) for item in approvals_payload)
    risk_config = dict(risk_config_payload)
    router_guardrails = dict(guardrails_payload)
    thresholds = dict(thresholds_payload)
    metadata = dict(metadata_payload)
    history = tuple(dict(cast(Mapping[str, Any], entry)) for entry in history_payload)

    return StrategyRuntimeConfig(
        policy_id=phenotype.policy_id,
        policy_hash=phenotype.policy_hash,
        tactic_id=phenotype.tactic_id,
        stage=phenotype.stage,
        approvals=approvals,
        risk_config=risk_config,
        router_guardrails=router_guardrails,
        thresholds=thresholds,
        metadata=metadata,
        history=history,
        updated_at=phenotype.updated_at,
        evidence_id=phenotype.evidence_id,
        runtime_config=dict(runtime_payload),
        runtime_json_bytes=runtime_json_bytes,
        runtime_digest=runtime_digest,
        runtime_config_path=str(runtime_output),
        docker_image_tags=enriched_payload["docker_image_tags"],
        payload=dict(enriched_payload),
        json_bytes=json_bytes,
        digest=digest,
    )
