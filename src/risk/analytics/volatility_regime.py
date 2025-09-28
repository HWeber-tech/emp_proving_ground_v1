"""Volatility regime classification utilities for risk sizing.

This module fulfils the roadmap requirement to enrich the risk stack with a
volatility regime classifier that can throttle or expand exposure based on the
observed environment.  The classifier intentionally keeps the implementation
lightweight so it can run inside CI and research notebooks without optional
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, MutableMapping, Sequence

from .volatility_target import calculate_realised_volatility

__all__ = [
    "VolatilityRegime",
    "VolatilityRegimeThresholds",
    "VolatilityRegimeAssessment",
    "classify_volatility_regime",
]


class VolatilityRegime(str, Enum):
    """Discrete regimes describing realised volatility conditions."""

    LOW = "low_volatility"
    NORMAL = "normal"
    HIGH = "high_volatility"
    EXTREME = "extreme_volatility"


@dataclass(slots=True, frozen=True)
class VolatilityRegimeThresholds:
    """Thresholds that partition realised volatility into regimes."""

    low: float
    high: float
    extreme: float

    def as_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high, "extreme": self.extreme}


@dataclass(slots=True, frozen=True)
class VolatilityRegimeAssessment:
    """Result returned by :func:`classify_volatility_regime`."""

    regime: VolatilityRegime
    realised_volatility: float
    target_volatility: float
    thresholds: VolatilityRegimeThresholds
    risk_multiplier: float

    def as_dict(self) -> dict[str, float | str]:
        payload: MutableMapping[str, float | str] = {
            "regime": self.regime.value,
            "realised_volatility": self.realised_volatility,
            "target_volatility": self.target_volatility,
            "risk_multiplier": self.risk_multiplier,
        }
        payload.update(self.thresholds.as_dict())
        return dict(payload)


def classify_volatility_regime(
    returns: Sequence[float] | Iterable[float],
    *,
    target_volatility: float | None = None,
    window: int | None = None,
    annualisation_factor: float = 1.0,
    low_ratio: float = 0.75,
    high_ratio: float = 1.25,
    extreme_ratio: float = 2.0,
    risk_multipliers: Mapping[str | VolatilityRegime, float] | None = None,
) -> VolatilityRegimeAssessment:
    """Classify the realised volatility regime for ``returns``.

    Parameters
    ----------
    returns:
        Iterable of returns expressed as decimal fractions.
    target_volatility:
        Baseline volatility expectation used to derive regime thresholds.  When
        omitted the realised volatility becomes the baseline.
    window:
        Optional lookback window passed to
        :func:`calculate_realised_volatility`.
    annualisation_factor:
        Annualisation scalar for realised volatility.
    low_ratio, high_ratio, extreme_ratio:
        Multipliers applied to the baseline volatility to determine regime
        thresholds.  They must satisfy ``0 < low_ratio < high_ratio <
        extreme_ratio``.
    risk_multipliers:
        Optional mapping overriding the default exposure multipliers applied to
        each regime.  Keys can be either :class:`VolatilityRegime` values or
        their string representation.
    """

    realised = calculate_realised_volatility(
        returns,
        window=window,
        annualisation_factor=annualisation_factor,
    )

    baseline = float(target_volatility) if target_volatility else realised
    baseline = max(baseline, 1e-9)

    ratios = [low_ratio, high_ratio, extreme_ratio]
    if any(r <= 0 for r in ratios) or not (low_ratio < high_ratio < extreme_ratio):
        raise ValueError("volatility regime ratios must satisfy 0 < low < high < extreme")

    thresholds = VolatilityRegimeThresholds(
        low=baseline * low_ratio,
        high=baseline * high_ratio,
        extreme=baseline * extreme_ratio,
    )

    regime: VolatilityRegime
    if realised <= thresholds.low:
        regime = VolatilityRegime.LOW
    elif realised <= thresholds.high:
        regime = VolatilityRegime.NORMAL
    elif realised <= thresholds.extreme:
        regime = VolatilityRegime.HIGH
    else:
        regime = VolatilityRegime.EXTREME

    default_multipliers: dict[VolatilityRegime, float] = {
        VolatilityRegime.LOW: 1.25,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.HIGH: 0.65,
        VolatilityRegime.EXTREME: 0.35,
    }

    multiplier = _resolve_multiplier(regime, risk_multipliers, default_multipliers)
    return VolatilityRegimeAssessment(
        regime=regime,
        realised_volatility=realised,
        target_volatility=baseline,
        thresholds=thresholds,
        risk_multiplier=multiplier,
    )


def _resolve_multiplier(
    regime: VolatilityRegime,
    overrides: Mapping[str | VolatilityRegime, float] | None,
    defaults: Mapping[VolatilityRegime, float],
) -> float:
    if overrides:
        key = regime
        if key in overrides:
            return max(0.0, float(overrides[key]))
        if regime.value in overrides:
            return max(0.0, float(overrides[regime.value]))
    return max(0.0, float(defaults.get(regime, 1.0)))
