"""Volatility-target sizing helpers supporting the high-impact roadmap.

The high-impact roadmap calls for volatility-aware position sizing that can
translate realised volatility observations into actionable notional targets.
This module keeps the implementation dependency-light so it can run inside the
existing CI footprint while providing a reusable result container for
reporting, telemetry, and risk enforcement layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from typing_extensions import Self

import numpy as np

__all__ = [
    "VolatilityTargetAllocation",
    "calculate_realised_volatility",
    "determine_target_allocation",
]


@dataclass(slots=True)
class VolatilityTargetAllocation:
    """Summary of a volatility-target sizing decision."""

    target_notional: float
    leverage: float
    target_volatility: float
    realised_volatility: float
    volatility_regime: str | None = None
    risk_multiplier: float | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        """Return a JSON-serialisable representation of the allocation."""

        return {
            "target_notional": self.target_notional,
            "leverage": self.leverage,
            "target_volatility": self.target_volatility,
            "realised_volatility": self.realised_volatility,
            "volatility_regime": self.volatility_regime,
            "risk_multiplier": self.risk_multiplier,
        }

    def with_adjustment(
        self,
        *,
        target_notional: float | None = None,
        leverage: float | None = None,
        volatility_regime: str | None = None,
        risk_multiplier: float | None = None,
    ) -> Self:
        """Return a copy updated with the supplied adjustments."""

        return VolatilityTargetAllocation(
            target_notional=target_notional if target_notional is not None else self.target_notional,
            leverage=leverage if leverage is not None else self.leverage,
            target_volatility=self.target_volatility,
            realised_volatility=self.realised_volatility,
            volatility_regime=volatility_regime if volatility_regime is not None else self.volatility_regime,
            risk_multiplier=risk_multiplier if risk_multiplier is not None else self.risk_multiplier,
        )


def _normalise_series(
    series: Sequence[float] | Iterable[float], *, window: int | None = None
) -> np.ndarray:
    values = np.asarray(list(series), dtype=float)
    if values.size == 0:
        raise ValueError("volatility series must contain at least one element")
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("volatility series must contain finite observations")
    if window is not None and window > 0 and values.size > window:
        values = values[-window:]
    return values


def calculate_realised_volatility(
    series: Sequence[float] | Iterable[float],
    *,
    window: int | None = None,
    annualisation_factor: float = 1.0,
) -> float:
    """Compute realised volatility for a returns series.

    Parameters
    ----------
    series:
        Iterable of returns expressed as decimal fractions (e.g. 0.01 == 1%).
    window:
        Optional lookback window. When provided the trailing ``window`` samples
        are used. When omitted the full series is considered.
    annualisation_factor:
        Scalar applied to the standard deviation result. Passing ``sqrt(252)``
        transforms daily volatility into an annualised estimate.
    """

    windowed = _normalise_series(series, window=window)
    ddof = 1 if windowed.size > 1 else 0
    realised = float(np.std(windowed, ddof=ddof))
    realised *= float(max(0.0, annualisation_factor))
    return realised


def determine_target_allocation(
    *,
    capital: float,
    target_volatility: float,
    realised_volatility: float,
    max_leverage: float = 3.0,
) -> VolatilityTargetAllocation:
    """Translate realised volatility into a volatility-target notional.

    The allocation scales exposure so that the expected realised volatility of
    the resulting position approaches ``target_volatility``. Exposure is capped
    by ``max_leverage`` to ensure the sizing decision remains bounded even when
    the realised volatility is extremely low.
    """

    capital = float(capital)
    target_volatility = float(target_volatility)
    realised_volatility = float(realised_volatility)
    max_leverage = max(0.0, float(max_leverage))

    if capital <= 0.0 or target_volatility <= 0.0 or realised_volatility < 0.0:
        return VolatilityTargetAllocation(
            target_notional=0.0,
            leverage=0.0,
            target_volatility=target_volatility,
            realised_volatility=realised_volatility,
        )

    if realised_volatility == 0.0:
        leverage = max_leverage
    else:
        leverage = target_volatility / realised_volatility
        leverage = max(0.0, min(leverage, max_leverage))

    target_notional = capital * leverage
    return VolatilityTargetAllocation(
        target_notional=target_notional,
        leverage=leverage,
        target_volatility=target_volatility,
        realised_volatility=realised_volatility,
    )
