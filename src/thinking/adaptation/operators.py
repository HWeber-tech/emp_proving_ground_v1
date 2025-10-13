"""Genotype-level adaptation operators.

Implements roadmap operators that mutate :class:`StrategyGenotype` instances
without mutating the original blueprint.  Each operator returns a
``GenotypeOperatorResult`` describing the updated genotype and lightweight
metadata about the mutation so orchestration layers can persist lineage and
governance evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Sequence

from .strategy_contracts import (
    StrategyExecutionTopology,
    StrategyFeature,
    StrategyGenotype,
    StrategyRiskTemplate,
)

__all__ = [
    "GenotypeOperatorResult",
    "op_add_feature",
    "op_drop_feature",
    "op_swap_execution_topology",
    "op_tighten_risk",
]


def _normalise_mapping(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be provided as a mapping")
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        cleaned[key_text] = value
    return MappingProxyType(cleaned)


def _ensure_feature(definition: StrategyFeature | Mapping[str, Any]) -> StrategyFeature:
    if isinstance(definition, StrategyFeature):
        return definition
    if isinstance(definition, Mapping):
        return StrategyFeature(**dict(definition))
    raise TypeError("feature must be a StrategyFeature or mapping definition")


def _ensure_topology(
    definition: StrategyExecutionTopology | Mapping[str, Any]
) -> StrategyExecutionTopology:
    if isinstance(definition, StrategyExecutionTopology):
        return definition
    if isinstance(definition, Mapping):
        return StrategyExecutionTopology(**dict(definition))
    raise TypeError("execution topology must be StrategyExecutionTopology or mapping")


def _copy_genotype(
    genotype: StrategyGenotype,
    *,
    features: Sequence[StrategyFeature] | None = None,
    execution_topology: StrategyExecutionTopology | None = None,
    risk_template: StrategyRiskTemplate | None = None,
) -> StrategyGenotype:
    return StrategyGenotype(
        strategy_id=genotype.strategy_id,
        features=tuple(features) if features is not None else genotype.features,
        execution_topology=(
            execution_topology if execution_topology is not None else genotype.execution_topology
        ),
        risk_template=risk_template if risk_template is not None else genotype.risk_template,
        tunables=genotype.tunables,
        metadata=genotype.metadata,
    )


def _require_position(position: int | None, upper_bound: int) -> int:
    if position is None:
        return upper_bound
    if not isinstance(position, int):
        raise TypeError("position must be an integer")
    if position < 0 or position > upper_bound:
        raise IndexError("position is out of range for feature list")
    return position


def _to_decimal(value: Any, *, field_name: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation as exc:  # pragma: no cover - defensive guard
            raise TypeError(f"{field_name} string must be numeric") from exc
    raise TypeError(f"{field_name} must be numeric")


def _quantise_decimal(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _build_risk_template(
    template: StrategyRiskTemplate,
    overrides: Mapping[str, Any],
) -> StrategyRiskTemplate:
    cloned = StrategyRiskTemplate(
        template_id=template.template_id,
        base_config=template.base_config,
        overrides=dict(overrides),
        description=template.description,
    )
    # Force validation early to surface invalid overrides.
    cloned.materialise()
    return cloned


@dataclass(frozen=True, slots=True)
class GenotypeOperatorResult:
    action: str
    genotype: StrategyGenotype
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _normalise_mapping(self.metadata))

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "action": self.action,
            "metadata": dict(self.metadata),
            "genotype": self.genotype.as_dict(),
        }


def op_add_feature(
    genotype: StrategyGenotype,
    feature: StrategyFeature | Mapping[str, Any],
    *,
    position: int | None = None,
    replace: bool = False,
) -> GenotypeOperatorResult:
    """Return a copy of ``genotype`` with ``feature`` inserted."""

    new_feature = _ensure_feature(feature)
    features = list(genotype.features)
    existing_index: int | None = None
    for index, current in enumerate(features):
        if current.name == new_feature.name:
            existing_index = index
            break

    replaced_feature: StrategyFeature | None = None
    if existing_index is not None:
        if not replace:
            raise ValueError(
                f"Feature {new_feature.name!r} already exists; pass replace=True to overwrite",
            )
        replaced_feature = features.pop(existing_index)
        if position is None:
            position = existing_index

    insert_index = _require_position(position, len(features))
    features.insert(insert_index, new_feature)
    updated = _copy_genotype(genotype, features=features)

    metadata: MutableMapping[str, Any] = {
        "feature": new_feature.name,
        "position": insert_index,
    }
    if replaced_feature is not None:
        metadata["replaced_feature"] = replaced_feature.name

    return GenotypeOperatorResult(
        action="op_add_feature",
        genotype=updated,
        metadata=metadata,
    )


def op_drop_feature(
    genotype: StrategyGenotype,
    feature_name: str,
    *,
    missing_ok: bool = False,
) -> GenotypeOperatorResult:
    """Drop ``feature_name`` from ``genotype`` when present."""

    name = str(feature_name).strip()
    if not name:
        raise ValueError("feature_name must be a non-empty string")

    features = list(genotype.features)
    target_index: int | None = None
    for index, feature in enumerate(features):
        if feature.name == name:
            target_index = index
            break

    if target_index is None:
        if missing_ok:
            return GenotypeOperatorResult(
                action="op_drop_feature",
                genotype=genotype,
                metadata={"feature": name, "dropped": False},
            )
        raise KeyError(f"Feature {name!r} not found in genotype")

    if len(features) <= 1:
        raise ValueError("Cannot drop the only feature in a genotype")

    removed = features.pop(target_index)
    updated = _copy_genotype(genotype, features=features)
    metadata = {
        "feature": removed.name,
        "previous_position": target_index,
        "dropped": True,
    }
    return GenotypeOperatorResult(
        action="op_drop_feature",
        genotype=updated,
        metadata=metadata,
    )


def op_swap_execution_topology(
    genotype: StrategyGenotype,
    topology: StrategyExecutionTopology | Mapping[str, Any],
    *,
    parameter_overrides: Mapping[str, Any] | None = None,
) -> GenotypeOperatorResult:
    """Replace the execution topology associated with ``genotype``."""

    new_topology = _ensure_topology(topology)
    if parameter_overrides:
        new_topology = new_topology.with_parameters(parameter_overrides)

    previous = genotype.execution_topology
    updated = _copy_genotype(genotype, execution_topology=new_topology)

    metadata: MutableMapping[str, Any] = {
        "previous_topology": previous.name,
        "new_topology": new_topology.name,
    }
    if previous.version:
        metadata["previous_version"] = previous.version
    if new_topology.version:
        metadata["new_version"] = new_topology.version

    return GenotypeOperatorResult(
        action="op_swap_execution_topology",
        genotype=updated,
        metadata=metadata,
    )


def op_tighten_risk(
    genotype: StrategyGenotype,
    *,
    scale: float = 0.75,
    floors: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> GenotypeOperatorResult:
    """Apply conservative scaling to key risk parameters."""

    if scale <= 0 or scale > 1:
        raise ValueError("scale must be within (0, 1]")

    risk_template = genotype.risk_template
    current_overrides: dict[str, Any] = dict(risk_template.overrides)
    changes: dict[str, Mapping[str, Any]] = {}

    scaling_fields = (
        "max_risk_per_trade_pct",
        "max_total_exposure_pct",
        "max_drawdown_pct",
        "target_volatility_pct",
        "max_leverage",
        "max_volatility_leverage",
    )

    floor_values: Mapping[str, Decimal] | None
    if floors is None:
        floor_values = None
    else:
        floor_values = {
            str(key).strip(): _to_decimal(value, field_name=f"floor for {key}")
            for key, value in floors.items()
            if str(key).strip()
        }

    for field in scaling_fields:
        base_value: Any
        if field in current_overrides:
            base_value = current_overrides[field]
        else:
            base_value = getattr(risk_template.base_config, field, None)
        if base_value is None:
            continue
        numeric = _to_decimal(base_value, field_name=field)
        scaled = _quantise_decimal(numeric * Decimal(str(scale)))
        if floor_values and field in floor_values and scaled < floor_values[field]:
            scaled = floor_values[field]
        if scaled != numeric or field not in current_overrides:
            current_overrides[field] = scaled
            changes[field] = {
                "previous": str(numeric),
                "new": str(scaled),
            }

    if overrides:
        if not isinstance(overrides, Mapping):
            raise TypeError("overrides must be provided as a mapping")
        for key, value in overrides.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            previous_value = current_overrides.get(key_text)
            current_overrides[key_text] = value
            changes[key_text] = {
                "previous": str(previous_value) if previous_value is not None else None,
                "new": str(value),
                "manual": True,
            }

    updated_template = _build_risk_template(risk_template, current_overrides)
    updated = _copy_genotype(genotype, risk_template=updated_template)

    metadata: MutableMapping[str, Any] = {
        "scale": scale,
        "updated_fields": [
            {
                "field": field,
                **details,
            }
            for field, details in sorted(changes.items())
        ],
    }
    if floor_values:
        metadata["floors"] = {key: str(value) for key, value in floor_values.items()}

    return GenotypeOperatorResult(
        action="op_tighten_risk",
        genotype=updated,
        metadata=metadata,
    )
