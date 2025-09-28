"""Market regime detection utilities for risk-aware execution.

This module translates raw price/return series into coarse-grained market
regime classifications that can be consumed by risk and sizing components.
The implementation intentionally mirrors the high-impact roadmap requirement
for volatility-informed risk throttling by:

* Classifying regimes (``calm``, ``normal``, ``storm``) using realised
  volatility estimates aligned to the ``VolConfig`` thresholds.
* Emitting regime-aware risk multipliers sourced from the sizing
  configuration so downstream components can attenuate exposure.
* Respecting optional regime gates (block/attenuate) defined in the
  volatility config.

The detector works purely on numeric inputs (``Mapping`` payloads containing
``close`` prices, ``prices`` or direct return series) to stay dependency-light
and test-friendly.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC, Sequence as SequenceABC
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.data_foundation.config.sizing_config import SizingConfig, load_sizing_config
from src.data_foundation.config.vol_config import load_vol_config
from src.sensory.what.volatility_engine import VolConfig

__all__ = [
    "RegimeLabel",
    "MarketRegimeResult",
    "MarketRegimeDetector",
]


@dataclass(frozen=True)
class RegimeLabel:
    """Light-weight representation of a regime label."""

    value: str

    def __str__(self) -> str:  # pragma: no cover - trivial helper
        return self.value


@dataclass(frozen=True)
class MarketRegimeResult:
    """Output returned by :class:`MarketRegimeDetector`."""

    regime: RegimeLabel
    confidence: float
    realised_volatility: float
    annualised_volatility: float
    sample_size: int
    risk_multiplier: float
    blocked: bool
    timestamp: datetime
    diagnostics: Mapping[str, float]

    @property
    def metadata(self) -> Mapping[str, float]:  # pragma: no cover - backwards compat
        return self.diagnostics


class MarketRegimeDetector:
    """Classify market regimes from return series.

    Parameters
    ----------
    vol_config:
        Optional :class:`VolConfig` override.  If omitted the canonical
        configuration is loaded from ``config/vol/vol_engine.yaml``.
    sizing_config:
        Optional :class:`SizingConfig` override providing regime multipliers.
    periods_per_year:
        Annualisation factor applied when converting realised volatility to an
        annualised metric.  Defaults to 252 (daily bars).
    """

    def __init__(
        self,
        *,
        vol_config: VolConfig | None = None,
        sizing_config: SizingConfig | None = None,
        periods_per_year: float = 252.0,
    ) -> None:
        self._vol_config = vol_config or load_vol_config()
        self._sizing_config = sizing_config or load_sizing_config()
        regime_multipliers = self._sizing_config.regime_multipliers or {}
        # Normalise multipliers to lower-case keys and guard against negatives.
        self._regime_multipliers = {
            str(name).lower(): max(0.0, float(mult))
            for name, mult in regime_multipliers.items()
        }
        if "unknown" not in self._regime_multipliers:
            self._regime_multipliers.setdefault("unknown", 1.0)
        self._default_multiplier = self._regime_multipliers.get("normal", 1.0)
        self._annualisation_factor = max(1.0, float(periods_per_year))

    # ------------------------------------------------------------------
    def detect_regime(
        self,
        data: Mapping[str, object] | Sequence[float] | np.ndarray,
    ) -> MarketRegimeResult:
        """Analyse market data and return a :class:`MarketRegimeResult`."""

        returns = self._extract_returns(data)
        sample_size = int(len(returns))
        timestamp = datetime.now(tz=timezone.utc)

        if sample_size < 2:
            return MarketRegimeResult(
                regime=RegimeLabel("unknown"),
                confidence=0.0,
                realised_volatility=0.0,
                annualised_volatility=0.0,
                sample_size=sample_size,
                risk_multiplier=self._regime_multipliers.get("unknown", 1.0),
                blocked=False,
                timestamp=timestamp,
                diagnostics={"sample_size": float(sample_size)},
            )

        realised_vol = float(np.std(returns, ddof=1))
        annualisation = self._resolve_periods_per_year(data)
        annualised_vol = realised_vol * math.sqrt(annualisation)

        regime_value = self._classify_regime(annualised_vol)
        multiplier = self._resolve_multiplier(regime_value)
        blocked = False

        if self._vol_config.use_regime_gate:
            if regime_value == self._vol_config.block_regime:
                gate_mode = (self._vol_config.gate_mode or "block").lower()
                if gate_mode == "block":
                    multiplier = 0.0
                    blocked = True
                else:  # attenuate
                    multiplier *= float(self._vol_config.attenuation_factor)
        multiplier = max(0.0, multiplier)

        confidence = self._calculate_confidence(annualised_vol, sample_size)

        diagnostics = {
            "annualised_volatility": annualised_vol,
            "realised_volatility": realised_vol,
            "sample_size": float(sample_size),
            "calm_threshold": float(self._vol_config.calm_thr),
            "storm_threshold": float(self._vol_config.storm_thr),
        }

        return MarketRegimeResult(
            regime=RegimeLabel(regime_value),
            confidence=confidence,
            realised_volatility=realised_vol,
            annualised_volatility=annualised_vol,
            sample_size=sample_size,
            risk_multiplier=multiplier,
            blocked=blocked,
            timestamp=timestamp,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    def _extract_returns(
        self,
        data: Mapping[str, object] | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        if isinstance(data, Mapping):
            if "returns" in data and data["returns"] is not None:
                raw_returns = np.asarray(list(cast_iterable(data["returns"])), dtype=float)
                return self._sanitize_series(raw_returns)
            for key in ("close", "prices", "price"):
                if key in data and data[key] is not None:
                    prices = np.asarray(list(cast_iterable(data[key])), dtype=float)
                    return self._prices_to_returns(prices)
            raise ValueError("Mapping payload must contain 'returns' or price series")

        array = np.asarray(list(cast_iterable(data)), dtype=float)
        return self._prices_to_returns(array)

    @staticmethod
    def _sanitize_series(series: np.ndarray) -> np.ndarray:
        if series.size == 0:
            return series
        mask = np.isfinite(series)
        return series[mask]

    def _prices_to_returns(self, prices: np.ndarray) -> np.ndarray:
        if prices.size < 2:
            return np.asarray([], dtype=float)
        clipped = np.clip(prices.astype(float), 1e-12, None)
        log_prices = np.log(clipped)
        returns = np.diff(log_prices)
        return self._sanitize_series(returns)

    def _classify_regime(self, annualised_vol: float) -> str:
        calm_thr = float(self._vol_config.calm_thr)
        storm_thr = float(self._vol_config.storm_thr)
        if annualised_vol <= calm_thr:
            return "calm"
        if annualised_vol >= storm_thr:
            return "storm"
        return "normal"

    def _resolve_multiplier(self, regime: str) -> float:
        return self._regime_multipliers.get(regime.lower(), self._default_multiplier)

    def _resolve_periods_per_year(
        self, data: Mapping[str, object] | Sequence[float] | np.ndarray
    ) -> float:
        if isinstance(data, Mapping):
            candidate = data.get("periods_per_year") or data.get("annualisation_factor")
            try:
                if candidate is not None:
                    value = float(candidate)
                    if value > 0:
                        return value
            except (TypeError, ValueError):
                pass
        return self._annualisation_factor

    def _calculate_confidence(self, annualised_vol: float, sample_size: int) -> float:
        calm_thr = float(self._vol_config.calm_thr)
        storm_thr = float(self._vol_config.storm_thr)
        window_hint = max(5, int(round(self._vol_config.rv_window_minutes / max(self._vol_config.bar_interval_minutes, 1))))
        size_component = min(1.0, sample_size / max(window_hint, 1))

        if annualised_vol <= calm_thr:
            distance_component = min(1.0, max(0.0, calm_thr - annualised_vol) / max(calm_thr, 1e-6) + 0.1)
        elif annualised_vol >= storm_thr:
            distance_component = min(1.0, max(0.0, annualised_vol - storm_thr) / max(storm_thr, 1e-6) + 0.1)
        else:
            mid = (calm_thr + storm_thr) / 2.0
            half_span = max((storm_thr - calm_thr) / 2.0, 1e-6)
            distance_component = max(0.0, 1.0 - abs(annualised_vol - mid) / half_span)

        confidence = 0.6 * size_component + 0.4 * distance_component
        return float(max(0.05, min(1.0, confidence)))


def cast_iterable(value: object) -> Iterable[float]:
    """Coerce heterogeneous iterable inputs to a flat iterable of floats."""

    if isinstance(value, (list, tuple, np.ndarray)):
        return value  # type: ignore[return-value]
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)  # type: ignore[return-value]
    if isinstance(value, IterableABC) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)  # type: ignore[return-value]
    raise TypeError("Value must be an iterable of numeric observations")

