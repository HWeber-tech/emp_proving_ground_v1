"""Tiny Recursive Model surrogate used for production TRM inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import math

from .encoder import _FEATURE_NAMES
from .types import StrategyEncoding, StrategyInference


@dataclass(slots=True)
class _LinearHead:
    weights: Mapping[str, float]
    bias: float
    clip: float | None = None

    def evaluate(self, features: Mapping[str, float], *, apply_clip: bool = True) -> float:
        score = self.bias
        for name, weight in self.weights.items():
            score += weight * float(features.get(name, 0.0))
        if apply_clip and self.clip is not None:
            limit = abs(self.clip)
            if score > limit:
                return limit
            if score < -limit:
                return -limit
        return score


@dataclass(slots=True)
class _HeadConfig:
    head: _LinearHead
    clip: float | None
    affine_scale: float = 1.0
    affine_bias: float = 0.0
    temperature: float | None = None

    def evaluate(self, features: Mapping[str, float]) -> float:
        score = self.head.evaluate(features, apply_clip=False)
        score = score * self.affine_scale + self.affine_bias
        if self.clip is not None:
            limit = abs(self.clip)
            if score > limit:
                return limit
            if score < -limit:
                return -limit
        return score


def _sigmoid(value: float) -> float:
    if value >= 0:
        expo = math.exp(-value)
        return 1.0 / (1.0 + expo)
    expo = math.exp(value)
    return expo / (1.0 + expo)


_QUANTILE_LEVELS: tuple[float, ...] = (0.25, 0.5, 0.75)
_DEFAULT_QUANTILE_BIAS: Mapping[float, float] = {
    0.25: -0.05,
    0.5: 0.0,
    0.75: 0.05,
}


_DEFAULT_MODEL_SPEC = {
    "feature_names": _FEATURE_NAMES,
    "heads": {
        "weight_adjust": {
            "weights": {
                "count_log": 0.04,
                "mean_pnl_scaled": -0.9,
                "pnl_std_scaled": -0.35,
                "risk_rate": -0.25,
                "win_rate": 0.35,
                "loss_rate": -0.45,
                "volatility_mean": -0.15,
                "spread_mean_pips": -0.05,
                "belief_confidence_mean": 0.5,
                "pnl_trend_scaled": 0.6,
                "drawdown_ratio": -0.3,
            },
            "bias": 0.0,
            "clip": 0.3,
        },
        "flag": {
            "weights": {
                "risk_rate": 2.4,
                "loss_rate": 1.6,
                "mean_pnl_scaled": -1.1,
                "pnl_std_scaled": -0.8,
                "drawdown_ratio": 0.9,
            },
            "bias": -0.3,
        },
        "experiment": {
            "weights": {
                "volatility_mean": 1.0,
                "pnl_trend_scaled": 0.8,
                "spread_mean_pips": -0.2,
                "belief_confidence_mean": 0.4,
            },
            "bias": -0.15,
        },
        "confidence": {
            "weights": {
                "win_rate": 1.4,
                "loss_rate": -0.8,
                "risk_rate": -0.9,
                "belief_confidence_mean": 0.8,
                "mean_pnl_scaled": 0.6,
            },
            "bias": 0.1,
        },
        "quantile": {
            "quantiles": {
                "q25": {
                    "shared": "weight_adjust",
                    "affine": {"scale": 1.0, "bias": _DEFAULT_QUANTILE_BIAS[0.25]},
                },
                "q50": {
                    "shared": "weight_adjust",
                    "affine": {"scale": 1.0, "bias": _DEFAULT_QUANTILE_BIAS[0.5]},
                },
                "q75": {
                    "shared": "weight_adjust",
                    "affine": {"scale": 1.0, "bias": _DEFAULT_QUANTILE_BIAS[0.75]},
                },
            }
        },
    },
}


_HEAD_NAMES: tuple[str, ...] = (
    "weight_adjust",
    "flag",
    "experiment",
    "confidence",
)


def _clone_head(head: _LinearHead) -> _LinearHead:
    return _LinearHead(weights=dict(head.weights), bias=head.bias, clip=head.clip)


def _clone_head_config(config: _HeadConfig) -> _HeadConfig:
    return _HeadConfig(
        head=_clone_head(config.head),
        clip=config.clip,
        affine_scale=config.affine_scale,
        affine_bias=config.affine_bias,
        temperature=config.temperature,
    )


def _build_linear_head(spec: Mapping[str, Any]) -> _LinearHead:
    weights_raw = spec.get("weights")
    if not isinstance(weights_raw, Mapping):
        raise ValueError("head spec must include a 'weights' mapping")
    weights = {str(name): float(value) for name, value in weights_raw.items()}
    bias_raw = spec.get("bias", 0.0)
    clip_raw = spec.get("clip")
    clip = float(clip_raw) if clip_raw is not None else None
    return _LinearHead(weights=weights, bias=float(bias_raw), clip=clip)


def _parse_affine(spec: Mapping[str, Any]) -> tuple[float, float]:
    block = spec.get("affine")
    if not isinstance(block, Mapping):
        return 1.0, 0.0
    scale_raw = block.get("scale", 1.0)
    bias_raw = block.get("bias", 0.0)
    try:
        scale = float(scale_raw)
    except (TypeError, ValueError):
        scale = 1.0
    try:
        bias = float(bias_raw)
    except (TypeError, ValueError):
        bias = 0.0
    return scale, bias


def _parse_temperature(value: Any) -> float | None:
    if value is None:
        return None
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return None
    return temperature if temperature > 0 else None


class TRMModel:
    """Lightweight surrogate model producing strategy-level inferences."""

    def __init__(self, spec: dict, *, temperature: float = 1.0) -> None:
        self._spec = spec

        heads_spec = spec.get("heads") or {}
        missing = [name for name in _HEAD_NAMES if name not in heads_spec]
        if missing:
            raise ValueError(f"model spec missing heads: {', '.join(missing)}")

        self._default_head_configs: dict[str, _HeadConfig] = {}
        available_heads: dict[str, _LinearHead] = {}
        for name in _HEAD_NAMES:
            base_head = _build_linear_head(heads_spec[name])
            config = _HeadConfig(
                head=_clone_head(base_head),
                clip=base_head.clip,
            )
            self._default_head_configs[name] = config
            available_heads[name] = base_head

        shared_heads_spec = spec.get("shared_heads") or {}
        for name, shared_spec in shared_heads_spec.items():
            shared_head = _build_linear_head(shared_spec)
            available_heads[str(name)] = shared_head

        self._default_quantile_configs = self._build_quantile_configs(
            heads_spec.get("quantile"),
            available_heads,
            defaults=None,
        )
        for level, config in self._default_quantile_configs.items():
            for alias in _quantile_aliases(level):
                available_heads[alias] = _clone_head(config.head)

        self._strategy_domains: dict[str, str] = {
            str(strategy): str(domain)
            for strategy, domain in (spec.get("strategy_domains") or {}).items()
            if strategy and domain
        }

        domain_definitions = (
            spec.get("domain_heads")
            or spec.get("domains")
            or {}
        )
        self._domain_head_configs: dict[str, dict[str, _HeadConfig]] = {}
        self._domain_quantile_configs: dict[str, dict[float, _HeadConfig]] = {}

        for domain_name, domain_spec in domain_definitions.items():
            if not isinstance(domain_spec, Mapping):
                continue
            domain_key = str(domain_name)
            strategies = domain_spec.get("strategies") or ()
            for strategy in strategies:
                if strategy:
                    self._strategy_domains[str(strategy)] = domain_key

            combined_head_specs: dict[str, Mapping[str, Any]] = {}
            heads_block = domain_spec.get("heads")
            if isinstance(heads_block, Mapping):
                combined_head_specs.update(
                    {str(key): value for key, value in heads_block.items() if isinstance(value, Mapping)}
                )
            for head_name in _HEAD_NAMES:
                head_block = domain_spec.get(head_name)
                if isinstance(head_block, Mapping) and head_name not in combined_head_specs:
                    combined_head_specs[head_name] = head_block

            domain_head_configs: dict[str, _HeadConfig] = {}
            for head_name, head_spec in combined_head_specs.items():
                if head_name not in _HEAD_NAMES:
                    continue
                config = self._build_head_config(
                    head_name,
                    head_spec,
                    available_heads,
                )
                domain_head_configs[head_name] = config
            self._domain_head_configs[domain_key] = domain_head_configs

            quantile_block = domain_spec.get("quantile")
            if quantile_block is None and isinstance(heads_block, Mapping):
                candidate = heads_block.get("quantile")
                if isinstance(candidate, Mapping):
                    quantile_block = candidate
            if isinstance(quantile_block, Mapping):
                quantile_configs = self._build_quantile_configs(
                    quantile_block,
                    available_heads,
                    defaults=self._default_quantile_configs,
                )
                self._domain_quantile_configs[domain_key] = quantile_configs

        self._temperature = max(0.1, float(temperature))
        encoded = json.dumps(spec, sort_keys=True)
        self._model_hash = sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def model_hash(self) -> str:
        return self._model_hash

    @classmethod
    def load(cls, path: Path | None = None, *, temperature: float = 1.0) -> "TRMModel":
        if path and path.exists():
            spec = json.loads(path.read_text())
        else:
            spec = _DEFAULT_MODEL_SPEC
        return cls(spec, temperature=temperature)

    def infer(self, encoding: StrategyEncoding) -> StrategyInference:
        features = encoding.features
        domain = self._strategy_domains.get(encoding.strategy_id)

        weight_config = self._resolve_head_config("weight_adjust", domain)
        flag_config = self._resolve_head_config("flag", domain)
        experiment_config = self._resolve_head_config("experiment", domain)
        confidence_config = self._resolve_head_config("confidence", domain)
        quantile_configs = self._resolve_quantile_configs(domain)

        weight_delta = float(weight_config.evaluate(features))
        flag_score = float(flag_config.evaluate(features))
        experiment_score = float(experiment_config.evaluate(features))
        confidence_score = float(confidence_config.evaluate(features))

        flag_temperature = flag_config.temperature or 1.0
        experiment_temperature = experiment_config.temperature or 1.0
        confidence_temperature = confidence_config.temperature or 1.0

        flag_probability = _sigmoid(flag_score / max(flag_temperature, 1e-6))
        experiment_probability = _sigmoid(experiment_score / max(experiment_temperature, 1e-6))
        combined_confidence_temperature = self._temperature * max(confidence_temperature, 1e-6)
        confidence = _sigmoid(confidence_score / combined_confidence_temperature)

        clip_value = weight_config.clip if weight_config.clip is not None else weight_config.head.clip
        if clip_value is not None:
            clip_abs = abs(clip_value) or 0.3
            weight_strength = min(0.99, abs(weight_delta) / clip_abs)
            confidence = max(confidence, weight_strength)
        confidence = max(confidence, flag_probability, experiment_probability)
        confidence = min(max(confidence, 0.0), 0.999)

        quantiles: dict[str, float] = {}
        q25_config = quantile_configs[0.25]
        q50_config = quantile_configs[0.5]
        q75_config = quantile_configs[0.75]
        q25 = float(q25_config.evaluate(features))
        q50 = float(q50_config.evaluate(features))
        q75 = float(q75_config.evaluate(features))
        if q50 < q25:
            q50 = q25
        if q50 > q75:
            q50 = q75
        if q25 > q75:
            q25, q75 = q75, q25
        quantiles["q25"] = q25
        quantiles["q50"] = q50
        quantiles["q75"] = q75

        return StrategyInference(
            strategy_id=encoding.strategy_id,
            weight_delta=weight_delta,
            flag_probability=flag_probability,
            experiment_probability=experiment_probability,
            confidence=confidence,
            quantiles=quantiles,
        )

    def _resolve_head_config(self, head_name: str, domain: str | None) -> _HeadConfig:
        if domain:
            domain_config = self._domain_head_configs.get(domain)
            if domain_config and head_name in domain_config:
                return domain_config[head_name]
        return self._default_head_configs[head_name]

    def _resolve_quantile_configs(self, domain: str | None) -> Mapping[float, _HeadConfig]:
        if domain:
            domain_config = self._domain_quantile_configs.get(domain)
            if domain_config:
                return domain_config
        return self._default_quantile_configs

    def _build_head_config(
        self,
        head_name: str,
        head_spec: Mapping[str, Any],
        available_heads: Mapping[str, _LinearHead],
        *,
        default_head: _LinearHead | None = None,
    ) -> _HeadConfig:
        base_head: _LinearHead

        if "weights" in head_spec:
            base_head = _build_linear_head(head_spec)
        else:
            shared_ref = head_spec.get("shared")
            if isinstance(shared_ref, str):
                shared_head = available_heads.get(shared_ref)
                if shared_head is None:
                    raise ValueError(f"unknown shared head '{shared_ref}' for {head_name}")
                base_head = _clone_head(shared_head)
            else:
                reference = default_head
                if reference is None:
                    reference = self._default_head_configs[head_name].head
                base_head = _clone_head(reference)

            if "bias" in head_spec:
                try:
                    base_head.bias = float(head_spec["bias"])
                except (TypeError, ValueError):
                    pass
            if "weights" in head_spec:
                weights_override = head_spec.get("weights")
                if isinstance(weights_override, Mapping):
                    base_head.weights = {
                        str(name): float(value)
                        for name, value in weights_override.items()
                    }
            if "clip" in head_spec:
                clip_override = head_spec.get("clip")
                base_head.clip = float(clip_override) if clip_override is not None else None

        scale, bias = _parse_affine(head_spec)
        clip_value = base_head.clip
        if "clip" in head_spec:
            clip_override = head_spec.get("clip")
            clip_value = float(clip_override) if clip_override is not None else None

        temperature = _parse_temperature(head_spec.get("temperature"))

        return _HeadConfig(
            head=base_head,
            clip=clip_value,
            affine_scale=scale,
            affine_bias=bias,
            temperature=temperature,
        )

    def _build_quantile_configs(
        self,
        quantile_spec: Mapping[str, Any] | None,
        available_heads: Mapping[str, _LinearHead],
        *,
        defaults: Mapping[float, _HeadConfig] | None,
    ) -> dict[float, _HeadConfig]:
        fallback = available_heads.get("weight_adjust")
        if fallback is None:
            raise ValueError("quantile head requires 'weight_adjust' base head")

        configs: dict[float, _HeadConfig] = {}
        entries: dict[float, Mapping[str, Any]] = {}
        if isinstance(quantile_spec, Mapping):
            candidate = quantile_spec.get("quantiles") if isinstance(quantile_spec.get("quantiles"), Mapping) else None
            source = candidate if isinstance(candidate, Mapping) else quantile_spec
            for key, value in source.items():
                tau = _parse_quantile_key(key)
                if tau is None or not isinstance(value, Mapping):
                    continue
                entries[tau] = value

        for level in _QUANTILE_LEVELS:
            if level in entries:
                head_spec = entries[level]
                config = self._build_head_config(
                    f"quantile@{level}",
                    head_spec,
                    available_heads,
                    default_head=fallback,
                )
            elif defaults and level in defaults:
                config = _clone_head_config(defaults[level])
            else:
                head_spec = {
                    "shared": "weight_adjust",
                    "affine": {
                        "scale": 1.0,
                        "bias": _DEFAULT_QUANTILE_BIAS.get(level, 0.0),
                    },
                }
                config = self._build_head_config(
                    f"quantile@{level}",
                    head_spec,
                    available_heads,
                    default_head=fallback,
                )
            configs[level] = config
        return configs


__all__ = ["TRMModel"]


def _parse_quantile_key(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        candidate = float(value)
        if math.isnan(candidate) or math.isinf(candidate):
            return None
    elif isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return None
        if token.startswith("quantile"):
            token = token[len("quantile") :]
        if token.startswith("@"):
            token = token[1:]
        if token.startswith("q"):
            token = token[1:]
        token = token.replace("%", "")
        token = token.strip()
        if not token:
            return None
        try:
            candidate = float(token)
        except ValueError:
            return None
        if candidate > 1.0:
            candidate /= 100.0
    else:
        return None
    candidate = round(candidate, 2)
    if candidate in _QUANTILE_LEVELS:
        return candidate
    return None


def _quantile_aliases(level: float) -> tuple[str, ...]:
    percent = int(round(level * 100))
    canonical = f"{level:.2f}"
    trimmed = canonical.rstrip("0").rstrip(".")
    aliases = {
        f"quantile@{canonical}",
        f"quantile@{trimmed}",
        f"quantile@{percent}",
        f"quantile@q{percent}",
    }
    return tuple(sorted(aliases))
