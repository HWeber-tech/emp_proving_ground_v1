"""Gamma exposure analysis utilities for the WHEN sensory dimension.

These helpers turn raw option position data into calibrated summaries that the
``WhenSensor`` can feed into its temporal intelligence signal.  The roadmap and
encyclopaedia emphasise that the WHEN organ should understand option dealer
positioning – particularly gamma "pin risk" around the current spot price – so
that operators know when liquidity providers are likely to defend strikes or
accelerate moves.  This module packages the analytics so they can be shared
across sensors, notebooks, and potential runtime integrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "GammaExposureAnalyzer",
    "GammaExposureAnalyzerConfig",
    "GammaStrikeProfile",
    "GammaExposureDataset",
    "GammaExposureSummary",
]


@dataclass(slots=True)
class GammaExposureAnalyzerConfig:
    """Configuration knobs for gamma exposure analysis."""

    near_fraction: float = 0.01
    """Fraction of the underlying price considered "near the money"."""

    pressure_normalizer: float = 1.0e5
    """Scale factor that converts absolute gamma into a [0, 1] pressure score."""

    minimum_spot: float = 1e-6
    """Guardrail to avoid divide-by-zero when the spot price is missing."""

    min_total_gamma: float = 1e-9
    """Threshold below which the summary is considered empty."""


@dataclass(slots=True)
class GammaExposureSummary:
    """Structured view of gamma positioning around the current spot price."""

    as_of: datetime
    symbol: str
    spot_price: float
    net_gamma: float
    total_abs_gamma: float
    near_gamma: float
    far_gamma: float
    pin_risk_score: float
    gamma_pressure: float
    flip_risk: bool
    dominant_strikes: tuple[GammaStrikeProfile, ...] = field(default_factory=tuple)

    @property
    def has_data(self) -> bool:
        return self.total_abs_gamma > 0.0

    @property
    def impact_score(self) -> float:
        """Return a 0–1 score describing how impactful the gamma posture is."""

        base = 0.6 * self.pin_risk_score + 0.4 * self.gamma_pressure
        return max(0.0, min(1.0, base))

    @property
    def primary_strike(self) -> GammaStrikeProfile | None:
        """Return the strike contributing the most absolute gamma."""

        return self.dominant_strikes[0] if self.dominant_strikes else None

    @classmethod
    def empty(cls, *, as_of: datetime, symbol: str, spot_price: float) -> "GammaExposureSummary":
        return cls(
            as_of=as_of,
            symbol=symbol,
            spot_price=spot_price,
            net_gamma=0.0,
            total_abs_gamma=0.0,
            near_gamma=0.0,
            far_gamma=0.0,
            pin_risk_score=0.0,
            gamma_pressure=0.0,
            flip_risk=False,
            dominant_strikes=(),
        )


@dataclass(slots=True)
class GammaStrikeProfile:
    """Aggregated gamma contribution for an individual strike."""

    strike: float
    net_gamma: float
    abs_gamma: float
    distance: float
    share_of_total: float

    @property
    def side(self) -> str:
        """Return ``long`` when dealers are long gamma at the strike, else ``short``."""

        return "long" if self.net_gamma >= 0 else "short"


class GammaExposureAnalyzer:
    """Convert raw option positions into :class:`GammaExposureSummary`."""

    def __init__(self, config: GammaExposureAnalyzerConfig | None = None) -> None:
        self._config = config or GammaExposureAnalyzerConfig()

    def summarise(
        self,
        positions: pd.DataFrame | None,
        *,
        spot_price: float | None = None,
        as_of: datetime | None = None,
        symbol: str | None = None,
    ) -> GammaExposureSummary:
        """Summarise gamma positioning.

        Args:
            positions: Option position snapshot.  Requires ``strike`` and
                ``gamma`` columns with optional ``open_interest`` and
                ``contract_multiplier`` fields.
            spot_price: The current underlying price.  If omitted, attempts to
                infer from the ``underlying_price``/``spot`` column or the
                average strike.
            as_of: Timestamp of the snapshot.  Defaults to ``datetime.utcnow``
                if no column is available.
            symbol: Instrument symbol.  Falls back to ``UNKNOWN`` when not
                supplied.
        """

        if positions is None or positions.empty:
            return GammaExposureSummary.empty(
                as_of=as_of or datetime.utcnow().replace(tzinfo=timezone.utc),
                symbol=symbol or "UNKNOWN",
                spot_price=float(spot_price or 0.0),
            )

        df = positions.copy()
        if "strike" not in df or "gamma" not in df:
            raise ValueError("positions must include 'strike' and 'gamma' columns")

        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
        df = df.dropna(subset=["strike", "gamma"])

        if df.empty:
            return GammaExposureSummary.empty(
                as_of=as_of or datetime.utcnow().replace(tzinfo=timezone.utc),
                symbol=symbol or "UNKNOWN",
                spot_price=float(spot_price or 0.0),
            )

        if as_of is None:
            if "timestamp" in df:
                as_of = pd.to_datetime(df["timestamp"].iloc[-1], utc=True, errors="coerce")
            if as_of is None or pd.isna(as_of):
                as_of = datetime.utcnow().replace(tzinfo=timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        if symbol is None and "symbol" in df:
            symbol = str(df["symbol"].iloc[-1])
        symbol = symbol or "UNKNOWN"

        inferred_spot = spot_price
        if inferred_spot is None:
            for column in ("underlying_price", "spot", "close", "reference_price"):
                if column in df:
                    candidate = pd.to_numeric(df[column], errors="coerce").dropna()
                    if not candidate.empty:
                        inferred_spot = float(candidate.iloc[-1])
                        break
        if inferred_spot is None:
            inferred_spot = float(df["strike"].median())

        spot_price = float(max(self._config.minimum_spot, inferred_spot))

        open_interest_raw = df.get("open_interest")
        if isinstance(open_interest_raw, pd.Series):
            open_interest = pd.to_numeric(open_interest_raw, errors="coerce").fillna(1.0)
        else:
            open_interest = pd.Series(1.0, index=df.index, dtype=float)

        multiplier_raw = df.get("contract_multiplier")
        if isinstance(multiplier_raw, pd.Series):
            multiplier = pd.to_numeric(multiplier_raw, errors="coerce").fillna(1.0)
        else:
            multiplier = pd.Series(1.0, index=df.index, dtype=float)

        weighted_gamma = df["gamma"].to_numpy(dtype=float) * open_interest.to_numpy(dtype=float)
        weighted_gamma = weighted_gamma * multiplier.to_numpy(dtype=float)

        total_abs_gamma = float(np.abs(weighted_gamma).sum())
        if total_abs_gamma < self._config.min_total_gamma:
            return GammaExposureSummary.empty(as_of=as_of, symbol=symbol, spot_price=spot_price)

        distance = np.abs(df["strike"].to_numpy(dtype=float) - spot_price)
        near_cutoff = max(self._config.minimum_spot * 0.25, spot_price * self._config.near_fraction)
        near_mask = distance <= near_cutoff

        near_gamma = float(np.abs(weighted_gamma[near_mask]).sum())
        far_gamma = float(total_abs_gamma - near_gamma)
        net_gamma = float(weighted_gamma.sum())

        strike_frame = pd.DataFrame(
            {
                "strike": df["strike"].to_numpy(dtype=float),
                "weighted_gamma": weighted_gamma,
            }
        )
        grouped = (
            strike_frame.groupby("strike", as_index=False)
            .agg(weighted_gamma=("weighted_gamma", "sum"))
            if not strike_frame.empty
            else pd.DataFrame(columns=["strike", "weighted_gamma"])
        )

        dominant_strikes: tuple[GammaStrikeProfile, ...]
        if grouped.empty:
            dominant_strikes = ()
        else:
            grouped["abs_gamma"] = grouped["weighted_gamma"].abs()
            grouped["distance"] = (grouped["strike"] - spot_price).abs()
            grouped["share"] = grouped["abs_gamma"] / total_abs_gamma
            grouped = grouped.sort_values(["abs_gamma", "distance"], ascending=[False, True])
            top_rows = grouped.head(5)[
                ["strike", "weighted_gamma", "abs_gamma", "distance", "share"]
            ].to_numpy(dtype=float)
            dominant_strikes = tuple(
                GammaStrikeProfile(
                    strike=float(strike),
                    net_gamma=float(weighted_gamma_value),
                    abs_gamma=float(abs_gamma_value),
                    distance=float(distance_value),
                    share_of_total=float(share_value),
                )
                for strike, weighted_gamma_value, abs_gamma_value, distance_value, share_value in top_rows.tolist()
            )

        below_gamma = float(weighted_gamma[df["strike"].to_numpy(dtype=float) < spot_price].sum())
        above_gamma = float(weighted_gamma[df["strike"].to_numpy(dtype=float) > spot_price].sum())
        flip_risk = below_gamma * above_gamma < 0

        density_component = near_gamma / total_abs_gamma
        balance_component = 1.0 - min(1.0, abs(net_gamma) / total_abs_gamma)
        pin_risk_score = max(0.0, min(1.0, density_component * balance_component))

        gamma_pressure = max(
            0.0,
            min(1.0, total_abs_gamma / self._config.pressure_normalizer),
        )

        return GammaExposureSummary(
            as_of=as_of,
            symbol=symbol,
            spot_price=spot_price,
            net_gamma=net_gamma,
            total_abs_gamma=total_abs_gamma,
            near_gamma=near_gamma,
            far_gamma=far_gamma,
            pin_risk_score=pin_risk_score,
            gamma_pressure=gamma_pressure,
            flip_risk=flip_risk,
            dominant_strikes=dominant_strikes,
        )


class GammaExposureDataset:
    """On-disk cache of illustrative gamma exposure data."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        analyzer: GammaExposureAnalyzer | None = None,
        config: GammaExposureAnalyzerConfig | None = None,
    ) -> None:
        self._config = config or GammaExposureAnalyzerConfig()
        self._analyzer = analyzer or GammaExposureAnalyzer(self._config)

        data = frame.copy()
        if "timestamp" in data:
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        self._frame = data

    @classmethod
    def from_csv(cls, path: str | Path | None = None) -> "GammaExposureDataset":
        if path is None:
            path = Path(__file__).resolve().parent / "assets" / "options_gamma.csv"
        frame = pd.read_csv(path)
        return cls(frame)

    def summarise(
        self,
        symbol: str,
        *,
        as_of: datetime,
        spot_price: float | None = None,
        lookback: timedelta = timedelta(days=1),
    ) -> GammaExposureSummary:
        as_of = as_of.astimezone(timezone.utc)
        window_start = as_of - lookback
        mask = (self._frame.get("symbol") == symbol) if "symbol" in self._frame else True
        if isinstance(mask, Iterable):
            filtered = self._frame[mask]
        else:
            filtered = self._frame

        if "timestamp" in filtered:
            mask = (filtered["timestamp"] >= window_start) & (filtered["timestamp"] <= as_of)
            filtered = filtered[mask]

        if filtered.empty:
            return GammaExposureSummary.empty(
                as_of=as_of, symbol=symbol, spot_price=float(spot_price or 0.0)
            )

        return self._analyzer.summarise(
            filtered,
            spot_price=spot_price,
            as_of=as_of,
            symbol=symbol,
        )
