"""Configuration schema for `UnderstandingRouter` fast-weight adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from src.understanding.router import FeatureGate, FastWeightAdapter, HebbianConfig


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_float(value: object | None) -> float:
    if value is None:
        raise ValueError("value cannot be None")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"cannot coerce {value!r} to float") from exc


def _coerce_optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot coerce {value!r} to float") from exc


def _as_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _parse_datetime(value: object | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"expires_at must be an ISO date string, got {value!r}")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid ISO datetime: {value!r}") from exc


@dataclass(frozen=True)
class FeatureGateDefinition:
    """Declarative feature gate definition for adapter configuration."""

    feature: str
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "FeatureGateDefinition":
        mapping = _as_mapping(raw, context="feature_gate")
        feature = str(mapping.get("feature", "")).strip()
        if not feature:
            raise ValueError("feature_gate requires a non-empty 'feature'")
        minimum = _coerce_optional_float(mapping.get("minimum"))
        maximum = _coerce_optional_float(mapping.get("maximum"))
        return cls(feature=feature, minimum=minimum, maximum=maximum)

    def build(self) -> FeatureGate:
        return FeatureGate(feature=self.feature, minimum=self.minimum, maximum=self.maximum)

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {"feature": self.feature}
        if self.minimum is not None:
            payload["minimum"] = float(self.minimum)
        if self.maximum is not None:
            payload["maximum"] = float(self.maximum)
        return payload


@dataclass(frozen=True)
class HebbianAdapterDefinition:
    """Hebbian parameter set for adaptive fast-weight updates."""

    feature: str
    learning_rate: float = 0.2
    decay: float = 0.1
    baseline: float = 1.0
    floor: float = 0.0
    ceiling: float | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "HebbianAdapterDefinition":
        mapping = _as_mapping(raw, context="hebbian")
        feature = str(mapping.get("feature", "")).strip()
        if not feature:
            raise ValueError("hebbian definition requires a 'feature'")
        return cls(
            feature=feature,
            learning_rate=_coerce_float(mapping.get("learning_rate", 0.2)),
            decay=_coerce_float(mapping.get("decay", 0.1)),
            baseline=_coerce_float(mapping.get("baseline", 1.0)),
            floor=_coerce_float(mapping.get("floor", 0.0)),
            ceiling=_coerce_optional_float(mapping.get("ceiling")),
        )

    def build(self) -> HebbianConfig:
        return HebbianConfig(
            feature=self.feature,
            learning_rate=self.learning_rate,
            decay=self.decay,
            baseline=self.baseline,
            floor=self.floor,
            ceiling=self.ceiling,
        )

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "feature": self.feature,
            "learning_rate": float(self.learning_rate),
            "decay": float(self.decay),
            "baseline": float(self.baseline),
            "floor": float(self.floor),
        }
        if self.ceiling is not None:
            payload["ceiling"] = float(self.ceiling)
        return payload


@dataclass(frozen=True)
class UnderstandingAdapterConfig:
    """Declarative adapter configuration bound to the router."""

    adapter_id: str
    tactic_id: str
    rationale: str
    multiplier: float | None = None
    feature_gates: tuple[FeatureGateDefinition, ...] = ()
    required_flags: Mapping[str, bool] = field(default_factory=dict)
    expires_at: datetime | None = None
    hebbian: HebbianAdapterDefinition | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "UnderstandingAdapterConfig":
        mapping = _as_mapping(raw, context="adapter")
        adapter_id = str(mapping.get("adapter_id", "")).strip()
        tactic_id = str(mapping.get("tactic_id", "")).strip()
        rationale = str(mapping.get("rationale", "")).strip()
        if not adapter_id:
            raise ValueError("adapter requires an 'adapter_id'")
        if not tactic_id:
            raise ValueError("adapter requires a 'tactic_id'")
        if not rationale:
            raise ValueError("adapter requires a 'rationale'")

        feature_gates_raw = mapping.get("feature_gates", ())
        if isinstance(feature_gates_raw, Sequence) and not isinstance(feature_gates_raw, (str, bytes)):
            feature_gates = tuple(
                FeatureGateDefinition.from_mapping(item) for item in feature_gates_raw
            )
        else:
            feature_gates = ()

        flags_raw = mapping.get("required_flags", {})
        if isinstance(flags_raw, Mapping):
            required_flags = {str(flag): _coerce_bool(val, default=True) for flag, val in flags_raw.items()}
        else:
            required_flags = {}

        hebbian_raw = mapping.get("hebbian")
        hebbian = (
            HebbianAdapterDefinition.from_mapping(hebbian_raw)
            if isinstance(hebbian_raw, Mapping)
            else None
        )

        multiplier_value = mapping.get("multiplier")
        multiplier = None if multiplier_value is None else float(multiplier_value)

        expires_at = _parse_datetime(mapping.get("expires_at"))

        return cls(
            adapter_id=adapter_id,
            tactic_id=tactic_id,
            rationale=rationale,
            multiplier=multiplier,
            feature_gates=feature_gates,
            required_flags=required_flags,
            expires_at=expires_at,
            hebbian=hebbian,
        )

    def build(self) -> FastWeightAdapter:
        return FastWeightAdapter(
            adapter_id=self.adapter_id,
            tactic_id=self.tactic_id,
            rationale=self.rationale,
            multiplier=self.multiplier,
            feature_gates=tuple(gate.build() for gate in self.feature_gates),
            required_flags=dict(self.required_flags),
            expires_at=self.expires_at,
            hebbian=self.hebbian.build() if self.hebbian else None,
        )

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "adapter_id": self.adapter_id,
            "tactic_id": self.tactic_id,
            "rationale": self.rationale,
            "feature_gates": [gate.as_dict() for gate in self.feature_gates],
            "required_flags": dict(self.required_flags),
        }
        if self.multiplier is not None:
            payload["multiplier"] = float(self.multiplier)
        if self.expires_at is not None:
            payload["expires_at"] = self.expires_at.isoformat()
        if self.hebbian is not None:
            payload["hebbian"] = dict(self.hebbian.as_dict())
        return payload


@dataclass(frozen=True)
class TierDefaults:
    """Default fast-weight posture for a deployment tier."""

    fast_weights_enabled: bool = False
    enabled_adapters: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "TierDefaults":
        mapping = _as_mapping(raw, context="tier_defaults")
        enabled = mapping.get("enabled_adapters", ())
        if isinstance(enabled, Sequence) and not isinstance(enabled, (str, bytes)):
            adapter_ids = tuple(str(item).strip() for item in enabled if str(item).strip())
        else:
            adapter_ids = ()
        return cls(
            fast_weights_enabled=_coerce_bool(mapping.get("fast_weights_enabled", False)),
            enabled_adapters=adapter_ids,
        )

    def as_dict(self) -> Mapping[str, object]:
        return {
            "fast_weights_enabled": self.fast_weights_enabled,
            "enabled_adapters": list(self.enabled_adapters),
        }


@dataclass(frozen=True)
class UnderstandingRouterConfig:
    """Typed configuration bundle for the understanding router."""

    feature_flag: str = "fast_weights_live"
    default_fast_weights_enabled: bool = False
    adapters: tuple[UnderstandingAdapterConfig, ...] = ()
    tier_defaults: Mapping[str, TierDefaults] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "UnderstandingRouterConfig":
        mapping = _as_mapping(raw, context="understanding_router_config")

        adapters_raw = mapping.get("adapters", ())
        if isinstance(adapters_raw, Sequence) and not isinstance(adapters_raw, (str, bytes)):
            adapters = tuple(
                UnderstandingAdapterConfig.from_mapping(item) for item in adapters_raw
            )
        else:
            adapters = ()

        tier_defaults_raw = mapping.get("tier_defaults", {})
        if isinstance(tier_defaults_raw, Mapping):
            tier_defaults = {
                str(tier).strip().lower(): TierDefaults.from_mapping(config)
                for tier, config in tier_defaults_raw.items()
                if isinstance(config, Mapping)
            }
        else:
            tier_defaults = {}

        feature_flag = str(mapping.get("feature_flag", "fast_weights_live")).strip() or "fast_weights_live"
        default_enabled = _coerce_bool(mapping.get("default_fast_weights_enabled", False))

        return cls(
            feature_flag=feature_flag,
            default_fast_weights_enabled=default_enabled,
            adapters=adapters,
            tier_defaults=tier_defaults,
        )

    def build_adapters(self) -> Mapping[str, FastWeightAdapter]:
        return {cfg.adapter_id: cfg.build() for cfg in self.adapters}

    def for_tier(self, tier: str | None) -> TierDefaults:
        if not tier:
            return TierDefaults(fast_weights_enabled=self.default_fast_weights_enabled)
        return self.tier_defaults.get(tier.strip().lower(), TierDefaults(fast_weights_enabled=self.default_fast_weights_enabled))

    def as_dict(self) -> Mapping[str, object]:
        return {
            "feature_flag": self.feature_flag,
            "default_fast_weights_enabled": self.default_fast_weights_enabled,
            "adapters": [cfg.as_dict() for cfg in self.adapters],
            "tier_defaults": {
                name: defaults.as_dict() for name, defaults in self.tier_defaults.items()
            },
        }


__all__ = [
    "FeatureGateDefinition",
    "HebbianAdapterDefinition",
    "TierDefaults",
    "UnderstandingAdapterConfig",
    "UnderstandingRouterConfig",
]

