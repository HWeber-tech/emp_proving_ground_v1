"""Tiny Recursive Model surrogate used for production TRM inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import math

from .encoder import _FEATURE_NAMES
from .types import StrategyEncoding, StrategyInference


@dataclass(slots=True)
class _LinearHead:
    weights: Mapping[str, float]
    bias: float
    clip: float | None = None

    def evaluate(self, features: Mapping[str, float]) -> float:
        score = self.bias
        for name, weight in self.weights.items():
            score += weight * float(features.get(name, 0.0))
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
    },
}


class TRMModel:
    """Lightweight surrogate model producing strategy-level inferences."""

    def __init__(self, spec: dict, *, temperature: float = 1.0) -> None:
        self._spec = spec
        heads = spec.get("heads", {})
        self._weight_head = _LinearHead(**heads["weight_adjust"])
        self._flag_head = _LinearHead(**heads["flag"])
        self._experiment_head = _LinearHead(**heads["experiment"])
        self._confidence_head = _LinearHead(**heads["confidence"])
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
        weight_delta = float(self._weight_head.evaluate(features))
        flag_score = float(self._flag_head.evaluate(features))
        experiment_score = float(self._experiment_head.evaluate(features))
        confidence_raw = float(self._confidence_head.evaluate(features)) / self._temperature

        flag_probability = _sigmoid(flag_score)
        experiment_probability = _sigmoid(experiment_score)
        confidence = _sigmoid(confidence_raw)

        if self._weight_head.clip:
            clip_value = abs(self._weight_head.clip) or 0.3
            weight_strength = min(0.99, abs(weight_delta) / clip_value)
            confidence = max(confidence, weight_strength)
        confidence = max(confidence, flag_probability, experiment_probability)
        confidence = min(max(confidence, 0.0), 0.999)

        return StrategyInference(
            strategy_id=encoding.strategy_id,
            weight_delta=weight_delta,
            flag_probability=flag_probability,
            experiment_probability=experiment_probability,
            confidence=confidence,
        )


__all__ = ["TRMModel"]
