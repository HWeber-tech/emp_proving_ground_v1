"""Market regime classification for execution model selection.

The high-impact roadmap calls for a blended classifier that inspects
volatility, liquidity, and sentiment telemetry so execution can adjust its
behaviour before orders hit the street.  This module implements a lightweight
scoring model that can run inside unit tests while still giving the runtime a
structured regime assessment to work with.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from src.data_foundation.config.execution_config import MarketRegimeModel

__all__ = [
    "MarketRegime",
    "MarketRegimeSignals",
    "MarketRegimeAssessment",
    "classify_market_regime",
    "apply_regime_adjustment",
]


class MarketRegime(str, Enum):
    """Enumerates the execution regimes considered by the roadmap classifier."""

    CALM = "calm"
    BALANCED = "balanced"
    STRESSED = "stressed"
    DISLOCATED = "dislocated"


@dataclass(slots=True, frozen=True)
class MarketRegimeSignals:
    """Normalised telemetry required by the classifier."""

    realised_volatility: float
    order_book_liquidity: float
    sentiment_score: float


@dataclass(slots=True, frozen=True)
class MarketRegimeAssessment:
    """Structured result returned by :func:`classify_market_regime`."""

    regime: MarketRegime
    score: float
    drivers: Mapping[str, float]
    risk_multiplier: float

    def as_dict(self) -> dict[str, float | str]:
        payload: dict[str, float | str] = {
            "regime": self.regime.value,
            "score": self.score,
            "risk_multiplier": self.risk_multiplier,
        }
        payload.update({key: float(value) for key, value in self.drivers.items()})
        return payload


def classify_market_regime(
    signals: MarketRegimeSignals, model: MarketRegimeModel
) -> MarketRegimeAssessment:
    """Blend volatility, liquidity, and sentiment to classify the market regime."""

    volatility_pressure = _normalise(
        signals.realised_volatility,
        low=model.calm_volatility,
        high=model.dislocated_volatility,
    )
    liquidity_pressure = _liquidity_pressure(
        signals.order_book_liquidity,
        low=model.low_liquidity_ratio,
        high=model.high_liquidity_ratio,
    )
    sentiment_pressure = _sentiment_pressure(
        signals.sentiment_score,
        negative=model.negative_sentiment,
        positive=model.positive_sentiment,
    )

    vol_w, liq_w, sent_w = model.normalised_weights()
    score = vol_w * volatility_pressure + liq_w * liquidity_pressure + sent_w * sentiment_pressure

    calm_threshold = _clamp(model.calm_score_threshold, 0.0, 1.0)
    balanced_threshold = max(calm_threshold, _clamp(model.balanced_score_threshold, 0.0, 1.0))
    stressed_threshold = max(balanced_threshold, _clamp(model.stressed_score_threshold, 0.0, 1.0))

    if score <= calm_threshold:
        regime = MarketRegime.CALM
    elif score <= balanced_threshold:
        regime = MarketRegime.BALANCED
    elif score <= stressed_threshold:
        regime = MarketRegime.STRESSED
    else:
        regime = MarketRegime.DISLOCATED

    drivers = {
        "volatility_pressure": volatility_pressure,
        "liquidity_pressure": liquidity_pressure,
        "sentiment_pressure": sentiment_pressure,
    }
    multiplier = _resolve_multiplier(regime, model.risk_multipliers)

    return MarketRegimeAssessment(
        regime=regime,
        score=score,
        drivers=drivers,
        risk_multiplier=multiplier,
    )


def apply_regime_adjustment(
    slippage_bps: float, assessment: MarketRegimeAssessment
) -> float:
    """Scale a slippage estimate using the regime multiplier."""

    return max(0.0, float(slippage_bps) * float(assessment.risk_multiplier))


def _normalise(value: float, *, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low), 0.0, 1.0)


def _liquidity_pressure(value: float, *, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    if value >= high:
        return 0.0
    if value <= low:
        return 1.0
    return (high - value) / (high - low)


def _sentiment_pressure(value: float, *, negative: float, positive: float) -> float:
    if positive <= negative:
        return 1.0
    if value >= positive:
        return 0.0
    if value <= negative:
        return 1.0
    return (positive - value) / (positive - negative)


def _clamp(value: float, low: float, high: float) -> float:
    if low > high:
        low, high = high, low
    return max(low, min(high, value))


def _resolve_multiplier(
    regime: MarketRegime, multipliers: Mapping[str, float] | None
) -> float:
    defaults = {
        MarketRegime.CALM: 0.85,
        MarketRegime.BALANCED: 1.0,
        MarketRegime.STRESSED: 1.25,
        MarketRegime.DISLOCATED: 1.6,
    }
    if multipliers:
        for key in (regime.value, regime.name, regime.value.upper()):
            if key in multipliers:
                try:
                    return max(0.0, float(multipliers[key]))
                except Exception:  # pragma: no cover - defensive
                    break
    return defaults[regime]
