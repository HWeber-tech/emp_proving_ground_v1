"""Analyse option surfaces for structural risk signals.

This module powers the Options Surface Monitor roadmap item.  It tracks
implied-volatility skew, open-interest concentration, dealer gamma posture, and
delta imbalances so operators can quickly understand option market structure
around the underlying price.  The monitor accepts a pandas ``DataFrame`` of
option quotes/positions and produces a structured summary suitable for
dashboards or observability pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.sensory.when.gamma_exposure import (
    GammaExposureAnalyzer,
    GammaExposureAnalyzerConfig,
    GammaExposureSummary,
)

UTC = timezone.utc

__all__ = [
    "DeltaImbalanceSnapshot",
    "ImpliedVolSkewSnapshot",
    "OpenInterestWall",
    "OptionsSurfaceMonitor",
    "OptionsSurfaceMonitorConfig",
    "OptionsSurfaceSummary",
]


@dataclass(slots=True)
class OptionsSurfaceMonitorConfig:
    """Configuration for :class:`OptionsSurfaceMonitor`."""

    gamma_config: GammaExposureAnalyzerConfig = GammaExposureAnalyzerConfig()
    max_open_interest_walls: int = 3
    iv_skew_threshold: float = 0.005
    near_moneyness_band: float = 0.15


@dataclass(slots=True)
class OpenInterestWall:
    """Represents a strike with concentrated open interest."""

    strike: float
    open_interest: float
    distance: float
    relative_distance: float
    dominant_side: str | None

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "strike": self.strike,
            "open_interest": self.open_interest,
            "distance": self.distance,
            "relative_distance": self.relative_distance,
            "dominant_side": self.dominant_side,
        }


@dataclass(slots=True)
class ImpliedVolSkewSnapshot:
    """Summary of call/put implied-volatility skew."""

    call_otm_iv: float | None
    put_otm_iv: float | None
    skew: float | None
    direction: str | None
    samples: int

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "call_otm_iv": self.call_otm_iv,
            "put_otm_iv": self.put_otm_iv,
            "skew": self.skew,
            "direction": self.direction,
            "samples": self.samples,
        }


@dataclass(slots=True)
class DeltaImbalanceSnapshot:
    """Summary of dealer delta imbalance across the surface."""

    net_delta: float | None
    call_delta: float | None
    put_delta: float | None
    normalised: float | None
    direction: str | None

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "net_delta": self.net_delta,
            "call_delta": self.call_delta,
            "put_delta": self.put_delta,
            "normalised": self.normalised,
            "direction": self.direction,
        }


@dataclass(slots=True)
class OptionsSurfaceSummary:
    """Structured analytics describing the option surface."""

    as_of: datetime
    symbol: str
    spot_price: float
    iv_skew: ImpliedVolSkewSnapshot
    open_interest_walls: tuple[OpenInterestWall, ...]
    gamma_exposure: GammaExposureSummary
    delta_imbalance: DeltaImbalanceSnapshot
    metadata: Mapping[str, Any]

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(),
            "symbol": self.symbol,
            "spot_price": self.spot_price,
            "iv_skew": self.iv_skew.as_dict(),
            "open_interest_walls": [wall.as_dict() for wall in self.open_interest_walls],
            "gamma_exposure": {
                "as_of": self.gamma_exposure.as_of.astimezone(UTC).isoformat(),
                "symbol": self.gamma_exposure.symbol,
                "spot_price": self.gamma_exposure.spot_price,
                "net_gamma": self.gamma_exposure.net_gamma,
                "total_abs_gamma": self.gamma_exposure.total_abs_gamma,
                "near_gamma": self.gamma_exposure.near_gamma,
                "far_gamma": self.gamma_exposure.far_gamma,
                "pin_risk_score": self.gamma_exposure.pin_risk_score,
                "gamma_pressure": self.gamma_exposure.gamma_pressure,
                "flip_risk": self.gamma_exposure.flip_risk,
                "dominant_strikes": [
                    {
                        "strike": strike_profile.strike,
                        "net_gamma": strike_profile.net_gamma,
                        "abs_gamma": strike_profile.abs_gamma,
                        "distance": strike_profile.distance,
                        "share_of_total": strike_profile.share_of_total,
                        "side": strike_profile.side,
                    }
                    for strike_profile in self.gamma_exposure.dominant_strikes
                ],
            },
            "delta_imbalance": self.delta_imbalance.as_dict(),
            "metadata": dict(self.metadata),
        }


class OptionsSurfaceMonitor:
    """Analyse option surfaces for structural signals."""

    def __init__(
        self,
        config: OptionsSurfaceMonitorConfig | None = None,
        *,
        gamma_analyzer: GammaExposureAnalyzer | None = None,
    ) -> None:
        self._config = config or OptionsSurfaceMonitorConfig()
        self._gamma_analyzer = gamma_analyzer or GammaExposureAnalyzer(self._config.gamma_config)

    def summarise(
        self,
        positions: pd.DataFrame | Mapping[str, Sequence[Any]] | None,
        *,
        spot_price: float | None = None,
        symbol: str | None = None,
        as_of: datetime | None = None,
    ) -> OptionsSurfaceSummary:
        frame = self._normalise_positions(positions)
        resolved_spot = self._resolve_spot(frame, spot_price)
        resolved_symbol = self._resolve_symbol(frame, symbol)
        timestamp = self._resolve_timestamp(frame, as_of)

        iv_skew = self._compute_iv_skew(frame, resolved_spot)
        oi_walls = self._compute_open_interest_walls(frame, resolved_spot)
        gamma_summary = self._compute_gamma_summary(frame, resolved_spot, resolved_symbol, timestamp)
        delta_imbalance = self._compute_delta_imbalance(frame)

        metadata = {
            "rows": int(frame.shape[0]),
            "columns": list(frame.columns),
            "has_gamma": bool("gamma" in frame.columns),
            "has_implied_vol": bool(self._resolve_iv_column(frame) is not None),
        }

        return OptionsSurfaceSummary(
            as_of=timestamp,
            symbol=resolved_symbol,
            spot_price=resolved_spot,
            iv_skew=iv_skew,
            open_interest_walls=oi_walls,
            gamma_exposure=gamma_summary,
            delta_imbalance=delta_imbalance,
            metadata=metadata,
        )

    def _normalise_positions(
        self, positions: pd.DataFrame | Mapping[str, Sequence[Any]] | None
    ) -> pd.DataFrame:
        if positions is None:
            return pd.DataFrame()
        if isinstance(positions, pd.DataFrame):
            return positions.copy()
        if isinstance(positions, Mapping):
            return pd.DataFrame(positions)
        raise TypeError("positions must be a pandas DataFrame or mapping of sequences")

    def _resolve_spot(self, frame: pd.DataFrame, spot_price: float | None) -> float:
        if spot_price is not None and np.isfinite(spot_price):
            return float(spot_price)
        for column in ("underlying_price", "spot", "reference_price", "close"):
            if column in frame:
                series = pd.to_numeric(frame[column], errors="coerce").dropna()
                if not series.empty:
                    return float(series.iloc[-1])
        strike_series = pd.to_numeric(frame.get("strike"), errors="coerce").dropna()
        if not strike_series.empty:
            return float(strike_series.median())
        return 0.0

    def _resolve_symbol(self, frame: pd.DataFrame, symbol: str | None) -> str:
        if symbol:
            return str(symbol)
        if "symbol" in frame and not frame.empty:
            tail = frame["symbol"].dropna()
            if not tail.empty:
                value = tail.iloc[-1]
                if isinstance(value, str):
                    return value
        return "UNKNOWN"

    def _resolve_timestamp(
        self, frame: pd.DataFrame, as_of: datetime | None
    ) -> datetime:
        if as_of is not None:
            return self._ensure_datetime(as_of)
        if "timestamp" in frame:
            ts_series = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
            if not ts_series.empty:
                return ts_series.iloc[-1].to_pydatetime()
        if "as_of" in frame:
            ts_series = pd.to_datetime(frame["as_of"], utc=True, errors="coerce").dropna()
            if not ts_series.empty:
                return ts_series.iloc[-1].to_pydatetime()
        return datetime.now(tz=UTC)

    def _ensure_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _compute_iv_skew(self, frame: pd.DataFrame, spot_price: float) -> ImpliedVolSkewSnapshot:
        iv_column = self._resolve_iv_column(frame)
        option_type_col = self._resolve_option_type_column(frame)
        if iv_column is None or option_type_col is None or spot_price <= 0.0:
            return ImpliedVolSkewSnapshot(None, None, None, None, 0)

        df = frame[["strike", iv_column, option_type_col, "open_interest"]].copy()
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df[iv_column] = pd.to_numeric(df[iv_column], errors="coerce")
        df["open_interest"] = pd.to_numeric(df.get("open_interest", 1.0), errors="coerce").fillna(0.0)
        df = df.dropna(subset=["strike", iv_column])
        if df.empty:
            return ImpliedVolSkewSnapshot(None, None, None, None, 0)

        df[option_type_col] = df[option_type_col].astype(str).str.lower()
        df = df[df[option_type_col].isin({"c", "call", "p", "put"})]
        if df.empty:
            return ImpliedVolSkewSnapshot(None, None, None, None, 0)

        moneyness = df["strike"].to_numpy(dtype=float) / spot_price
        df = df.assign(moneyness=moneyness)

        near_band = self._config.near_moneyness_band
        lower_bound = 1.0 - near_band
        upper_bound = 1.0 + near_band
        df = df[(df["moneyness"] >= lower_bound) & (df["moneyness"] <= upper_bound)]
        if df.empty:
            return ImpliedVolSkewSnapshot(None, None, None, None, 0)

        call_mask = df[option_type_col].isin({"c", "call"})
        put_mask = df[option_type_col].isin({"p", "put"})

        call_df = df[call_mask & (df["strike"] >= spot_price)]
        put_df = df[put_mask & (df["strike"] <= spot_price)]

        call_iv = self._weighted_average(call_df, iv_column, "open_interest")
        put_iv = self._weighted_average(put_df, iv_column, "open_interest")
        samples = int(call_df.shape[0] + put_df.shape[0])

        if call_iv is None or put_iv is None:
            return ImpliedVolSkewSnapshot(call_iv, put_iv, None, None, samples)

        skew = call_iv - put_iv
        direction = self._categorise_skew(skew)
        return ImpliedVolSkewSnapshot(call_iv, put_iv, skew, direction, samples)

    def _resolve_iv_column(self, frame: pd.DataFrame) -> str | None:
        for column in ("implied_volatility", "iv", "sigma", "volatility"):
            if column in frame.columns:
                return column
        return None

    def _resolve_option_type_column(self, frame: pd.DataFrame) -> str | None:
        for column in ("option_type", "type", "side"):
            if column in frame.columns:
                return column
        return None

    def _weighted_average(
        self, frame: pd.DataFrame, value_column: str, weight_column: str
    ) -> float | None:
        if frame.empty:
            return None
        values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
        weights = (
            pd.to_numeric(frame.get(weight_column, 1.0), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if np.allclose(weights.sum(), 0.0):
            return None
        return float(np.average(values, weights=weights))

    def _categorise_skew(self, skew: float) -> str:
        if np.isnan(skew):
            return "neutral"
        threshold = self._config.iv_skew_threshold
        if skew > threshold:
            return "call"
        if skew < -threshold:
            return "put"
        return "neutral"

    def _compute_open_interest_walls(
        self, frame: pd.DataFrame, spot_price: float
    ) -> tuple[OpenInterestWall, ...]:
        if "strike" not in frame or "open_interest" not in frame:
            return tuple()
        strikes = pd.to_numeric(frame["strike"], errors="coerce")
        open_interest = pd.to_numeric(frame["open_interest"], errors="coerce")
        valid_mask = ~(strikes.isna() | open_interest.isna())
        if not valid_mask.any():
            return tuple()

        df = frame.loc[valid_mask, ["strike", "open_interest"]].copy()
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")

        grouped = (
            df.groupby("strike", as_index=False)["open_interest"].sum().sort_values(
                "open_interest", ascending=False
            )
        )
        if grouped.empty:
            return tuple()

        grouped = grouped.head(self._config.max_open_interest_walls)
        option_type_col = self._resolve_option_type_column(frame)

        walls: list[OpenInterestWall] = []
        strike_numeric = pd.to_numeric(frame.get("strike"), errors="coerce")
        for _, row in grouped.iterrows():
            strike_value = float(row["strike"])
            total_oi = float(row["open_interest"])
            distance = abs(strike_value - spot_price)
            relative = distance / spot_price if spot_price else 0.0
            dominant_side: str | None = None
            if option_type_col is not None:
                subset_mask = np.isclose(strike_numeric, strike_value, rtol=0.0, atol=1e-6)
                subset = frame.loc[subset_mask, [option_type_col, "open_interest"]].copy()
                if not subset.empty:
                    subset[option_type_col] = (
                        subset[option_type_col]
                        .astype(str)
                        .str.lower()
                        .map({"c": "call", "call": "call", "p": "put", "put": "put"})
                    )
                    subset = subset.dropna(subset=[option_type_col])
                    if not subset.empty:
                        by_side = (
                            subset.groupby(option_type_col)["open_interest"].sum().sort_values(ascending=False)
                        )
                        if not by_side.empty:
                            dominant_side = str(by_side.index[0])
            walls.append(
                OpenInterestWall(
                    strike=strike_value,
                    open_interest=total_oi,
                    distance=distance,
                    relative_distance=relative,
                    dominant_side=dominant_side,
                )
            )
        return tuple(walls)

    def _compute_gamma_summary(
        self,
        frame: pd.DataFrame,
        spot_price: float,
        symbol: str,
        as_of: datetime,
    ) -> GammaExposureSummary:
        if "gamma" not in frame or "strike" not in frame:
            return GammaExposureSummary.empty(as_of=as_of, symbol=symbol, spot_price=spot_price)
        try:
            payload = frame.copy()
            if "contract_multiplier" not in payload:
                payload = payload.copy()
                payload["contract_multiplier"] = 1.0
            return self._gamma_analyzer.summarise(
                payload,
                spot_price=spot_price,
                symbol=symbol,
                as_of=as_of,
            )
        except Exception:
            return GammaExposureSummary.empty(as_of=as_of, symbol=symbol, spot_price=spot_price)

    def _compute_delta_imbalance(self, frame: pd.DataFrame) -> DeltaImbalanceSnapshot:
        if "delta" not in frame:
            return DeltaImbalanceSnapshot(None, None, None, None, None)

        delta = pd.to_numeric(frame["delta"], errors="coerce")
        open_interest = pd.to_numeric(frame.get("open_interest", 1.0), errors="coerce").fillna(0.0)
        multiplier = pd.to_numeric(frame.get("contract_multiplier", 1.0), errors="coerce").fillna(1.0)

        valid_mask = ~(delta.isna() | open_interest.isna() | multiplier.isna())
        if not valid_mask.any():
            return DeltaImbalanceSnapshot(None, None, None, None, None)

        weighted_delta = delta[valid_mask].to_numpy(dtype=float)
        contracts = open_interest[valid_mask].to_numpy(dtype=float) * multiplier[valid_mask].to_numpy(dtype=float)
        exposures = weighted_delta * contracts

        option_type_col = self._resolve_option_type_column(frame)
        call_delta = None
        put_delta = None
        if option_type_col is not None:
            types = (
                frame.loc[valid_mask, option_type_col]
                .astype(str)
                .str.lower()
                .map({"c": "call", "call": "call", "p": "put", "put": "put"})
                .to_numpy()
            )
            call_mask = types == "call"
            put_mask = types == "put"
            call_delta = (
                float(exposures[call_mask].sum()) if call_mask.any() else None
            )
            put_delta = (
                float(exposures[put_mask].sum()) if put_mask.any() else None
            )

        net_delta = float(exposures.sum())
        denominator = float(np.abs(exposures).sum())
        normalised = net_delta / denominator if denominator else None
        direction = None
        if normalised is not None:
            if normalised > self._config.iv_skew_threshold:
                direction = "call"
            elif normalised < -self._config.iv_skew_threshold:
                direction = "put"
            else:
                direction = "neutral"

        return DeltaImbalanceSnapshot(net_delta, call_delta, put_delta, normalised, direction)
