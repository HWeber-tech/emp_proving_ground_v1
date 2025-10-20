from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


UTC = timezone.utc


@dataclass(slots=True)
class VolConfig:
    bar_interval_minutes: int = 5
    daily_fit_lookback_days: int = 500
    rv_window_minutes: int = 60
    blend_weight: float = 0.7
    calm_thr: float = 0.08
    storm_thr: float = 0.18
    risk_budget_per_trade: float = 0.003
    k_stop: float = 1.3
    var_confidence: float = 0.95
    ewma_lambda: float = 0.94
    use_regime_gate: bool = False
    block_regime: str = "storm"
    gate_mode: str = "block"
    attenuation_factor: float = 0.3
    brake_scale: float = 0.7


@dataclass(slots=True)
class VolatilitySurfaceSlice:
    """Single tenor/moneyness slice of the volatility surface."""

    tenor_minutes: int
    atm_vol: float
    otm_put_vol: float
    otm_call_vol: float
    skew: float
    term_structure_slope: float
    sample_count: int

    def as_dict(self) -> Mapping[str, float | int]:
        return {
            "tenor_minutes": self.tenor_minutes,
            "atm_vol": self.atm_vol,
            "otm_put_vol": self.otm_put_vol,
            "otm_call_vol": self.otm_call_vol,
            "skew": self.skew,
            "term_structure_slope": self.term_structure_slope,
            "sample_count": self.sample_count,
        }


@dataclass(slots=True)
class VolatilitySurfaceSnapshot:
    """Structured view of the real-time volatility surface."""

    as_of: datetime
    symbol: str
    spot_price: float
    slices: tuple[VolatilitySurfaceSlice, ...]
    surface: pd.DataFrame
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "as_of": self.as_of.astimezone(UTC).isoformat(),
            "symbol": self.symbol,
            "spot_price": self.spot_price,
            "slices": [slice_.as_dict() for slice_ in self.slices],
            "surface": self.surface.to_dict(orient="index"),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class FlowAggressionMetrics:
    """Order-flow aggression snapshot derived from recent activity."""

    buy_volume: float
    sell_volume: float
    net_flow: float
    imbalance: float
    aggression_score: float
    dominant_side: str
    price_impact: float
    flow_velocity: float
    volatility_correlation: float | None
    sample_count: int
    window_minutes: float
    total_volume: float

    def as_dict(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "net_flow": self.net_flow,
            "imbalance": self.imbalance,
            "aggression_score": self.aggression_score,
            "dominant_side": self.dominant_side,
            "price_impact": self.price_impact,
            "flow_velocity": self.flow_velocity,
            "sample_count": self.sample_count,
            "window_minutes": self.window_minutes,
            "total_volume": self.total_volume,
        }
        if self.volatility_correlation is not None and not np.isnan(self.volatility_correlation):
            payload["volatility_correlation"] = self.volatility_correlation
        else:
            payload["volatility_correlation"] = None
        return payload


@dataclass(slots=True)
class VolatilityTopologySnapshot:
    """Composite snapshot combining surface topology and flow aggression."""

    surface: VolatilitySurfaceSnapshot
    flow_aggression: FlowAggressionMetrics
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "surface": self.surface.as_dict(),
            "flow_aggression": self.flow_aggression.as_dict(),
            "metadata": dict(self.metadata),
        }


class VolatilityEngine:
    """Construct real-time volatility surfaces and flow aggression metrics."""

    def __init__(self, config: VolConfig | None = None) -> None:
        self.config: VolConfig = config or VolConfig()

    # Public API -----------------------------------------------------------------
    def map_topology(
        self,
        market_data: pd.DataFrame | Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None,
        *,
        symbol: str | None = None,
        as_of: datetime | None = None,
    ) -> VolatilityTopologySnapshot:
        frame = self._normalise_frame(market_data)
        closes = self._resolve_closes(frame)
        returns = self._compute_returns(closes)

        resolved_symbol = self._resolve_symbol(frame, symbol)
        timestamp = self._resolve_timestamp(frame, as_of)
        spot_price = float(closes.iloc[-1]) if not closes.empty else 0.0

        surface_snapshot = self.compute_surface(
            frame,
            closes=closes,
            returns=returns,
            symbol=resolved_symbol,
            as_of=timestamp,
            spot_price=spot_price,
        )
        flow_metrics = self.compute_flow_aggression(frame, returns=returns)

        metadata = {
            "rows": int(frame.shape[0]),
            "has_volume": bool("volume" in frame.columns),
            "has_signed_volume": bool("signed_volume" in frame.columns),
            "config": asdict(self.config),
        }

        return VolatilityTopologySnapshot(
            surface=surface_snapshot,
            flow_aggression=flow_metrics,
            metadata=metadata,
        )

    def compute_surface(
        self,
        frame: pd.DataFrame,
        *,
        closes: pd.Series | None = None,
        returns: pd.Series | None = None,
        symbol: str | None = None,
        as_of: datetime | None = None,
        spot_price: float | None = None,
    ) -> VolatilitySurfaceSnapshot:
        closes = closes if closes is not None else self._resolve_closes(frame)
        returns = returns if returns is not None else self._compute_returns(closes)

        tenors = self._surface_tenors()
        slices = self._build_surface_slices(tenors, returns)
        surface_frame = self._build_surface_frame(slices)

        metadata = {
            "tenors_minutes": list(tenors),
            "columns": list(surface_frame.columns),
            "return_samples": int(returns.shape[0]),
        }

        return VolatilitySurfaceSnapshot(
            as_of=as_of or self._resolve_timestamp(frame, None),
            symbol=symbol or self._resolve_symbol(frame, None),
            spot_price=spot_price if spot_price is not None else float(closes.iloc[-1]) if not closes.empty else 0.0,
            slices=slices,
            surface=surface_frame,
            metadata=metadata,
        )

    def compute_flow_aggression(
        self,
        frame: pd.DataFrame,
        *,
        returns: pd.Series | None = None,
    ) -> FlowAggressionMetrics:
        returns = returns if returns is not None else self._compute_returns(self._resolve_closes(frame))
        volume = self._coerce_series(frame.get("volume"), default=0.0, length=len(frame))
        signed_volume = self._coerce_series(frame.get("signed_volume"), default=np.nan, length=len(frame))

        if signed_volume.isna().all():
            direction = self._infer_flow_direction(frame)
            signed_volume = volume * direction
        signed_volume = signed_volume.fillna(0.0)

        buy_volume = float(signed_volume.clip(lower=0.0).sum())
        sell_volume = float((-signed_volume.clip(upper=0.0)).sum())
        gross_flow = buy_volume + sell_volume
        net_flow = buy_volume - sell_volume

        imbalance = float(net_flow / gross_flow) if gross_flow > 0 else 0.0
        aggression_score = float(min(1.0, abs(net_flow) / (gross_flow + 1e-9)) * min(1.0, abs(imbalance) * 1.5))
        dominant_side = "neutral"
        if imbalance > 0.05:
            dominant_side = "buy"
        elif imbalance < -0.05:
            dominant_side = "sell"

        price_impact = self._compute_price_impact(frame)
        duration_minutes = self._compute_window_minutes(frame)
        flow_velocity = float(net_flow / (duration_minutes + 1e-9)) if duration_minutes > 0 else 0.0

        volatility_correlation = None
        if not returns.empty and signed_volume.shape[0] >= returns.shape[0] and returns.shape[0] > 2:
            tail_signed = signed_volume.iloc[-returns.shape[0]:]
            try:
                abs_returns = np.abs(returns.to_numpy())
                abs_flow = np.abs(tail_signed.to_numpy())
                if np.std(abs_returns) > 1e-12 and np.std(abs_flow) > 1e-12:
                    corr = np.corrcoef(abs_returns, abs_flow)
                    volatility_correlation = float(corr[0, 1]) if corr.shape == (2, 2) else None
                else:
                    volatility_correlation = None
            except Exception:
                volatility_correlation = None

        return FlowAggressionMetrics(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            net_flow=net_flow,
            imbalance=imbalance,
            aggression_score=aggression_score,
            dominant_side=dominant_side,
            price_impact=price_impact,
            flow_velocity=flow_velocity,
            volatility_correlation=volatility_correlation,
            sample_count=int(len(frame)),
            window_minutes=duration_minutes,
            total_volume=gross_flow,
        )

    # Internal helpers -----------------------------------------------------------
    def _normalise_frame(
        self,
        data: pd.DataFrame | Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any]] | None,
    ) -> pd.DataFrame:
        if data is None:
            return self._empty_frame()
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif isinstance(data, Mapping):
            try:
                frame = pd.DataFrame(data)
            except ValueError:
                frame = pd.DataFrame([dict(data)])
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            try:
                frame = pd.DataFrame(list(data))
            except Exception as exc:  # pragma: no cover - defensive
                raise TypeError("Unable to convert sequence to DataFrame") from exc
        else:
            raise TypeError("Unsupported market data format")

        if frame.empty:
            return self._empty_frame()

        columns = {col.lower(): col for col in frame.columns}
        close_col = self._resolve_column(frame, columns, ["close", "price", "last", "mid"])
        volume_col = self._resolve_column(frame, columns, ["volume", "qty", "size"])
        ts_col = self._resolve_column(frame, columns, ["timestamp", "time", "as_of"])

        if close_col is None:
            frame["close"] = 0.0
        elif close_col != "close":
            frame["close"] = frame.pop(close_col)

        if volume_col is not None and volume_col != "volume":
            frame["volume"] = frame.pop(volume_col)

        if "volume" not in frame:
            frame["volume"] = 0.0

        if ts_col is not None and ts_col != "timestamp":
            frame["timestamp"] = frame.pop(ts_col)

        if "timestamp" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        else:
            timestamps = pd.date_range(
                end=pd.Timestamp.now(tz=UTC),
                periods=len(frame),
                freq=f"{max(1, self.config.bar_interval_minutes)}min",
            )
            frame["timestamp"] = timestamps

        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame["close"] = self._coerce_series(frame["close"], default=0.0, length=len(frame))
        frame["volume"] = self._coerce_series(frame["volume"], default=0.0, length=len(frame))

        if "signed_volume" in frame:
            frame["signed_volume"] = self._coerce_series(frame["signed_volume"], default=0.0, length=len(frame))

        return frame

    def _empty_frame(self) -> pd.DataFrame:
        timestamp = pd.date_range(end=pd.Timestamp.now(tz=UTC), periods=1, freq=f"{max(1, self.config.bar_interval_minutes)}min")
        return pd.DataFrame({
            "timestamp": timestamp,
            "close": pd.Series([0.0], dtype=float),
            "volume": pd.Series([0.0], dtype=float),
        })

    @staticmethod
    def _coerce_series(series: Any, *, default: float, length: int) -> pd.Series:
        if isinstance(series, pd.Series):
            result = series.astype(float, copy=True)
        elif isinstance(series, Iterable) and not isinstance(series, (str, bytes, bytearray)):
            result = pd.Series(list(series), dtype=float)
        else:
            result = pd.Series([default] * length, dtype=float)
        if result.shape[0] != length:
            result = result.reindex(range(length), fill_value=default)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(default)
        return result.astype(float)

    @staticmethod
    def _resolve_column(
        frame: pd.DataFrame,
        lookup: Mapping[str, str],
        candidates: Sequence[str],
    ) -> str | None:
        for candidate in candidates:
            if candidate in lookup:
                return lookup[candidate]
        for column in frame.columns:
            if column.lower() in candidates:
                return column
        return None

    def _resolve_closes(self, frame: pd.DataFrame) -> pd.Series:
        closes = frame.get("close")
        if isinstance(closes, pd.Series):
            series = closes.astype(float)
        else:
            series = self._coerce_series(closes, default=0.0, length=len(frame))
        return series.replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _compute_returns(closes: pd.Series) -> pd.Series:
        if closes is None or closes.empty:
            return pd.Series(dtype=float)
        shifted = closes.shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.log(closes / shifted)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        return returns.astype(float)

    def _surface_tenors(self) -> tuple[int, ...]:
        base = max(1, int(self.config.bar_interval_minutes))
        rv = max(base, int(self.config.rv_window_minutes))
        tenors = sorted({base, rv, rv * 2, rv * 4})
        return tuple(tenors)

    def _annualisation_factor(self) -> float:
        minutes_per_session = 390  # 6.5 hours trading day
        bars_per_year = (252 * minutes_per_session) / max(1, self.config.bar_interval_minutes)
        return float(np.sqrt(bars_per_year))

    def _build_surface_slices(
        self,
        tenors: Sequence[int],
        returns: pd.Series,
    ) -> tuple[VolatilitySurfaceSlice, ...]:
        if returns is None:
            returns = pd.Series(dtype=float)

        annualisation = self._annualisation_factor()
        slices: list[VolatilitySurfaceSlice] = []
        previous_atm = None
        for tenor in tenors:
            window = max(2, int(round(tenor / max(1, self.config.bar_interval_minutes))))
            window_returns = returns.tail(window)
            sample_count = int(window_returns.shape[0])
            if sample_count < 2 or window_returns.std(ddof=0) == 0:
                base_vol = 0.0
                skew = 0.0
            else:
                base_vol = float(window_returns.std(ddof=0) * annualisation)
                skew = float(window_returns.skew()) if sample_count >= 3 else 0.0

            put_scale = 1.0 + max(0.0, -skew) * 0.35
            call_scale = 1.0 + max(0.0, skew) * 0.35

            term_slope = 0.0 if previous_atm is None else base_vol - previous_atm
            previous_atm = base_vol

            slices.append(
                VolatilitySurfaceSlice(
                    tenor_minutes=int(tenor),
                    atm_vol=base_vol,
                    otm_put_vol=base_vol * put_scale,
                    otm_call_vol=base_vol * call_scale,
                    skew=skew,
                    term_structure_slope=term_slope,
                    sample_count=sample_count,
                )
            )

        return tuple(slices)

    @staticmethod
    def _build_surface_frame(slices: Sequence[VolatilitySurfaceSlice]) -> pd.DataFrame:
        if not slices:
            return pd.DataFrame(columns=["atm_vol", "otm_put_vol", "otm_call_vol", "skew", "term_structure_slope", "sample_count"])
        data = {
            "atm_vol": [slice_.atm_vol for slice_ in slices],
            "otm_put_vol": [slice_.otm_put_vol for slice_ in slices],
            "otm_call_vol": [slice_.otm_call_vol for slice_ in slices],
            "skew": [slice_.skew for slice_ in slices],
            "term_structure_slope": [slice_.term_structure_slope for slice_ in slices],
            "sample_count": [slice_.sample_count for slice_ in slices],
        }
        index = pd.Index([slice_.tenor_minutes for slice_ in slices], name="tenor_minutes")
        frame = pd.DataFrame(data, index=index)
        return frame

    @staticmethod
    def _resolve_symbol(frame: pd.DataFrame, override: str | None) -> str:
        if override:
            return override
        for column in ("symbol", "ticker", "asset"):
            if column in frame.columns:
                value = frame[column].dropna()
                if not value.empty:
                    return str(value.iloc[-1])
        return "UNKNOWN"

    @staticmethod
    def _resolve_timestamp(frame: pd.DataFrame, override: datetime | None) -> datetime:
        if override is not None:
            return override.astimezone(UTC) if override.tzinfo else override.replace(tzinfo=UTC)
        if "timestamp" in frame.columns and not frame["timestamp"].dropna().empty:
            ts = frame["timestamp"].dropna().iloc[-1]
            if isinstance(ts, pd.Timestamp):
                return ts.to_pydatetime().astimezone(UTC)
            if isinstance(ts, datetime):
                return ts.astimezone(UTC) if ts.tzinfo else ts.replace(tzinfo=UTC)
        return datetime.now(tz=UTC)

    @staticmethod
    def _infer_flow_direction(frame: pd.DataFrame) -> pd.Series:
        closes = frame.get("close", pd.Series(dtype=float)).astype(float)
        if closes.shape[0] == 0:
            return pd.Series([0.0] * len(frame), dtype=float)
        diff = closes.diff().fillna(0.0)
        raw = np.sign(diff.to_numpy())
        direction = pd.Series(raw, index=closes.index, dtype=float)
        return direction.fillna(0.0)

    @staticmethod
    def _compute_price_impact(frame: pd.DataFrame) -> float:
        closes = frame.get("close")
        if closes is None or len(closes) < 2:
            return 0.0
        series = pd.Series(closes, dtype=float)
        if series.iloc[0] == 0:
            return 0.0
        return float((series.iloc[-1] - series.iloc[0]) / series.iloc[0])

    @staticmethod
    def _compute_window_minutes(frame: pd.DataFrame) -> float:
        if "timestamp" not in frame or frame["timestamp"].dropna().shape[0] < 2:
            return 0.0
        timestamps = frame["timestamp"].dropna()
        start = timestamps.iloc[0]
        end = timestamps.iloc[-1]
        if isinstance(start, pd.Timestamp):
            start_dt = start.to_pydatetime()
        elif isinstance(start, datetime):
            start_dt = start
        else:
            return 0.0
        if isinstance(end, pd.Timestamp):
            end_dt = end.to_pydatetime()
        elif isinstance(end, datetime):
            end_dt = end
        else:
            return 0.0
        return float((end_dt - start_dt).total_seconds() / 60.0)


__all__ = [
    "FlowAggressionMetrics",
    "VolConfig",
    "VolatilityEngine",
    "VolatilitySurfaceSlice",
    "VolatilitySurfaceSnapshot",
    "VolatilityTopologySnapshot",
]
