"""Contracts for strategy genotypes and phenotypes.

The adaptation roadmap requires a structured contract describing how strategy
blueprints (genotypes) materialise into runtime-ready payloads (phenotypes).
This module provides immutable dataclasses for:

* Feature blueprints that describe the signals required by a tactic.
* Execution topology definitions that capture the routing or execution shape.
* Risk templates that combine baseline :class:`~src.config.risk.risk_config.RiskConfig`
  constructs with deterministic overrides.
* Tunable parameters that mutation/evolution operators can adjust safely.

The :class:`StrategyGenotype` orchestrates these components and offers a
:meth:`~StrategyGenotype.realise` helper that enforces validation before
producing a :class:`StrategyPhenotype`.  The resulting phenotype is a frozen
snapshot with resolved parameters that downstream governance and runtime layers
can serialise without ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Sequence

from src.config.risk.risk_config import RiskConfig

__all__ = [
    "StrategyFeature",
    "StrategyExecutionTopology",
    "StrategyRiskTemplate",
    "StrategyTunable",
    "StrategyGenotype",
    "StrategyPhenotype",
]


def _normalise_identifier(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _freeze_mapping(
    mapping: Mapping[Any, Any] | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    if mapping is None:
        return MappingProxyType({})
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{field_name} must be provided as a mapping")
    payload: dict[str, Any] = {}
    for key, value in mapping.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError(f"{field_name} contains blank keys")
        payload[key_text] = value
    return MappingProxyType(payload)


def _freeze_sequence(values: Sequence[Any] | None) -> tuple[str, ...]:
    if not values:
        return ()
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return tuple(unique)


def _coerce_bound(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise TypeError(f"{field_name} must be numeric") from exc


def _ensure_numeric(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} requires a numeric value") from exc


@dataclass(frozen=True, slots=True)
class StrategyFeature:
    """Feature blueprint describing a signal required by a strategy.

    Every feature must carry a falsifiable economic hypothesis plus references
    to CI tests that exercise the mechanism so governance can link behaviour to
    verified evidence.
    """

    name: str
    inputs: Sequence[str] = field(default_factory=tuple)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None
    economic_hypothesis: str = field(default="")
    ci_tests: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = _normalise_identifier(self.name, field_name="Feature name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "inputs", _freeze_sequence(self.inputs))
        params = _freeze_mapping(self.parameters, field_name=f"parameters for {name}")
        object.__setattr__(self, "parameters", params)
        if self.description is not None:
            description = str(self.description).strip()
            object.__setattr__(self, "description", description or None)

        hypothesis = str(self.economic_hypothesis).strip()
        if not hypothesis:
            raise ValueError(f"StrategyFeature {name!r} requires a non-empty economic_hypothesis")
        object.__setattr__(self, "economic_hypothesis", hypothesis)

        ci_tests = _freeze_sequence(self.ci_tests)
        if not ci_tests:
            raise ValueError(f"StrategyFeature {name!r} must declare at least one CI test reference")
        object.__setattr__(self, "ci_tests", ci_tests)

    def with_parameters(self, overrides: Mapping[str, Any] | None) -> "StrategyFeature":
        if not overrides:
            return self
        merged = dict(self.parameters)
        for key, value in overrides.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            merged[key_text] = value
        return StrategyFeature(
            name=self.name,
            inputs=self.inputs,
            parameters=merged,
            description=self.description,
            economic_hypothesis=self.economic_hypothesis,
            ci_tests=self.ci_tests,
        )

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "name": self.name,
            "inputs": list(self.inputs),
            "parameters": dict(self.parameters),
            "economic_hypothesis": self.economic_hypothesis,
            "ci_tests": list(self.ci_tests),
        }
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class StrategyExecutionTopology:
    """Execution topology contract describing routing/execution shape."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    version: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        name = _normalise_identifier(self.name, field_name="Topology name")
        object.__setattr__(self, "name", name)
        params = _freeze_mapping(self.parameters, field_name=f"parameters for {name} topology")
        object.__setattr__(self, "parameters", params)
        if self.version is not None:
            version = str(self.version).strip()
            object.__setattr__(self, "version", version or None)
        if self.description is not None:
            description = str(self.description).strip()
            object.__setattr__(self, "description", description or None)

    def with_parameters(self, overrides: Mapping[str, Any] | None) -> "StrategyExecutionTopology":
        if not overrides:
            return self
        if not isinstance(overrides, Mapping):
            raise TypeError("Topology overrides must be provided as a mapping")
        merged = dict(self.parameters)
        for key, value in overrides.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            merged[key_text] = value
        return StrategyExecutionTopology(
            name=self.name,
            parameters=merged,
            version=self.version,
            description=self.description,
        )

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "name": self.name,
            "parameters": dict(self.parameters),
        }
        if self.version:
            payload["version"] = self.version
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class StrategyRiskTemplate:
    """Risk template pairing a baseline config with deterministic overrides."""

    template_id: str
    base_config: RiskConfig | Mapping[str, Any] | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        template_id = _normalise_identifier(self.template_id, field_name="Risk template id")
        object.__setattr__(self, "template_id", template_id)

        base = self.base_config
        if base is None:
            base_cfg = RiskConfig()
        elif isinstance(base, RiskConfig):
            base_cfg = base
        elif isinstance(base, Mapping):
            base_cfg = RiskConfig(**dict(base))
        else:  # pragma: no cover - defensive guard
            raise TypeError("base_config must be a RiskConfig or mapping")
        object.__setattr__(self, "base_config", base_cfg)

        overrides = _freeze_mapping(
            self.overrides,
            field_name=f"risk overrides for template {template_id}",
        )
        object.__setattr__(self, "overrides", overrides)

        if self.description is not None:
            description = str(self.description).strip()
            object.__setattr__(self, "description", description or None)

    def materialise(self, overrides: Mapping[str, Any] | None = None) -> RiskConfig:
        payload = dict(self.base_config.dict())
        payload.update(self.overrides)
        if overrides:
            for key, value in overrides.items():
                key_text = str(key).strip()
                if not key_text:
                    continue
                payload[key_text] = value
        return RiskConfig(**payload)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "template_id": self.template_id,
            "overrides": dict(self.overrides),
        }
        payload["base_config"] = self.base_config.dict()
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class StrategyTunable:
    """Tunable parameter definition used by mutation operators."""

    name: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    description: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = _normalise_identifier(self.name, field_name="Tunable name")
        object.__setattr__(self, "name", name)
        min_value = _coerce_bound(self.minimum, field_name=f"minimum bound for {name}")
        max_value = _coerce_bound(self.maximum, field_name=f"maximum bound for {name}")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError(
                f"Tunable {name} minimum bound {min_value} exceeds maximum {max_value}"
            )
        object.__setattr__(self, "minimum", min_value)
        object.__setattr__(self, "maximum", max_value)
        if self.description is not None:
            description = str(self.description).strip()
            object.__setattr__(self, "description", description or None)
        object.__setattr__(self, "tags", _freeze_sequence(self.tags))

    def resolve(self, overrides: Mapping[str, Any] | None = None) -> Any:
        value = self.default
        if overrides and self.name in overrides:
            value = overrides[self.name]
        numeric = None
        if self.minimum is not None or self.maximum is not None:
            numeric = _ensure_numeric(value, field_name=f"Tunable {self.name}")
        if numeric is not None:
            if self.minimum is not None and numeric < self.minimum - 1e-12:
                raise ValueError(
                    f"Tunable {self.name} value {numeric} is below minimum {self.minimum}"
                )
            if self.maximum is not None and numeric > self.maximum + 1e-12:
                raise ValueError(
                    f"Tunable {self.name} value {numeric} exceeds maximum {self.maximum}"
                )
        return value

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "name": self.name,
            "default": self.default,
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.description:
            payload["description"] = self.description
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


@dataclass(frozen=True, slots=True)
class StrategyGenotype:
    """Immutable strategy blueprint combining features, topology, and risk."""

    strategy_id: str
    features: Sequence[StrategyFeature]
    execution_topology: StrategyExecutionTopology
    risk_template: StrategyRiskTemplate
    tunables: Sequence[StrategyTunable] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _tunable_index: Mapping[str, StrategyTunable] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        strategy_id = _normalise_identifier(self.strategy_id, field_name="strategy_id")
        object.__setattr__(self, "strategy_id", strategy_id)

        features = tuple(self.features)
        if not features:
            raise ValueError("StrategyGenotype requires at least one feature")
        seen_features: set[str] = set()
        for feature in features:
            if feature.name in seen_features:
                raise ValueError(f"Duplicate feature name {feature.name!r} in genotype")
            seen_features.add(feature.name)
        object.__setattr__(self, "features", features)

        tunables = tuple(self.tunables)
        index: dict[str, StrategyTunable] = {}
        for tunable in tunables:
            if tunable.name in index:
                raise ValueError(f"Duplicate tunable name {tunable.name!r} in genotype")
            index[tunable.name] = tunable
        object.__setattr__(self, "tunables", tunables)
        object.__setattr__(self, "_tunable_index", MappingProxyType(index))

        metadata = _freeze_mapping(self.metadata, field_name=f"metadata for {strategy_id}")
        object.__setattr__(self, "metadata", metadata)

    def resolve_tunables(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        overrides = overrides or {}
        resolved: dict[str, Any] = {}
        for name, tunable in self._tunable_index.items():
            resolved[name] = tunable.resolve(overrides)
        unknown_overrides = set(overrides) - set(resolved)
        if unknown_overrides:
            unknown = ", ".join(sorted(unknown_overrides))
            raise KeyError(f"Unknown tunable overrides: {unknown}")
        return resolved

    def _realise_features(
        self,
        overrides: Mapping[str, Mapping[str, Any]] | None,
    ) -> tuple[StrategyFeature, ...]:
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, Mapping):
            raise TypeError("Feature overrides must be provided as a mapping")
        realised: list[StrategyFeature] = []
        feature_index = {feature.name: feature for feature in self.features}
        unknown_overrides = set(overrides) - set(feature_index)
        if unknown_overrides:
            unknown = ", ".join(sorted(unknown_overrides))
            raise KeyError(f"Unknown feature overrides: {unknown}")
        for feature in self.features:
            params = overrides.get(feature.name)
            if params is not None and not isinstance(params, Mapping):
                raise TypeError(
                    f"Feature overrides for {feature.name!r} must be provided as a mapping"
                )
            realised.append(feature.with_parameters(params))
        return tuple(realised)

    def realise(
        self,
        *,
        tunable_overrides: Mapping[str, Any] | None = None,
        risk_overrides: Mapping[str, Any] | None = None,
        feature_parameter_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        topology_overrides: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StrategyPhenotype":
        resolved_features = self._realise_features(feature_parameter_overrides)
        resolved_tunables = self.resolve_tunables(tunable_overrides)
        topology = self.execution_topology.with_parameters(topology_overrides)
        risk_config = self.risk_template.materialise(risk_overrides)

        merged_metadata = dict(self.metadata)
        if metadata:
            for key, value in metadata.items():
                key_text = str(key).strip()
                if not key_text:
                    continue
                merged_metadata[key_text] = value

        return StrategyPhenotype(
            strategy_id=self.strategy_id,
            features=resolved_features,
            execution_topology=topology,
            risk_config=risk_config,
            tunable_values=resolved_tunables,
            metadata=merged_metadata,
        )

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "strategy_id": self.strategy_id,
            "features": [feature.as_dict() for feature in self.features],
            "execution_topology": self.execution_topology.as_dict(),
            "risk_template": self.risk_template.as_dict(),
            "tunables": [tunable.as_dict() for tunable in self.tunables],
            "metadata": dict(self.metadata),
        }
        return payload


@dataclass(frozen=True, slots=True)
class StrategyPhenotype:
    """Runtime-ready snapshot derived from a strategy genotype."""

    strategy_id: str
    features: Sequence[StrategyFeature]
    execution_topology: StrategyExecutionTopology
    risk_config: RiskConfig
    tunable_values: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strategy_id = _normalise_identifier(self.strategy_id, field_name="strategy_id")
        object.__setattr__(self, "strategy_id", strategy_id)
        object.__setattr__(self, "features", tuple(self.features))
        tunables = _freeze_mapping(
            self.tunable_values, field_name=f"tunable values for {strategy_id}"
        )
        object.__setattr__(self, "tunable_values", tunables)
        metadata = _freeze_mapping(self.metadata, field_name=f"metadata for {strategy_id} phenotype")
        object.__setattr__(self, "metadata", metadata)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "features": [feature.as_dict() for feature in self.features],
            "execution_topology": self.execution_topology.as_dict(),
            "risk_config": self.risk_config.dict(),
            "tunables": dict(self.tunable_values),
            "metadata": dict(self.metadata),
        }

    def risk_summary(self) -> Mapping[str, Decimal | bool | int | float]:
        """Return a lightweight view of risk parameters for telemetry surfaces."""

        summary: dict[str, Decimal | bool | int | float] = {}
        for key, value in self.risk_config.dict().items():
            if isinstance(value, (Decimal, bool, int, float)):
                summary[key] = value
        return summary
