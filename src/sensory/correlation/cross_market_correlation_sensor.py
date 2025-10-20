"""Cross-market correlation sensor.

Continuously estimates lag/lead relationships between related instruments by
tracking rolling correlations across venues and symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping

import math
import pandas as pd

from src.sensory.lineage import build_lineage_record
from src.sensory.signals import SensorSignal

__all__ = ["CrossMarketCorrelationSensor"]


@dataclass(slots=True)
class _RelationshipState:
    correlation: float
    lag: float
    confidence: float
    samples: int
    updated_at: datetime


class CrossMarketCorrelationSensor:
    """Estimate cross-market lag/lead relationships in rolling windows."""

    def __init__(
        self,
        window: int = 240,
        *,
        max_lag: int = 5,
        min_overlap: int = 20,
        significance_threshold: float = 0.45,
        smoothing: float = 0.35,
        max_relationships: int = 5,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        if min_overlap <= 0:
            raise ValueError("min_overlap must be positive")
        if not (0.0 <= significance_threshold <= 1.0):
            raise ValueError("significance_threshold must be between 0 and 1")
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError("smoothing must be between 0 and 1")
        if max_relationships <= 0:
            raise ValueError("max_relationships must be positive")

        self._window = int(window)
        self._max_lag = int(max_lag)
        self._min_overlap = int(min_overlap)
        self._threshold = float(significance_threshold)
        self._smoothing = float(smoothing)
        self._max_relationships = int(max_relationships)
        self._history: dict[str, _RelationshipState] = {}

    # ------------------------------------------------------------------
    def process(self, df: pd.DataFrame | None) -> list[SensorSignal]:
        timestamp = self._resolve_timestamp(df)
        if df is None or df.empty:
            return [self._default_signal(timestamp, reason="insufficient_market_data")]

        panel = self._build_price_panel(df)
        if panel is None or panel.shape[1] < 2:
            return [self._default_signal(timestamp, reason="insufficient_series")]

        returns = panel.pct_change().dropna(how="all")
        if returns.empty or returns.shape[1] < 2:
            return [self._default_signal(timestamp, reason="insufficient_variation")]

        returns = returns.tail(self._window + self._max_lag + 1)
        frequency_seconds = self._infer_frequency_seconds(returns.index)

        relationships = self._analyse_relationships(returns, timestamp, frequency_seconds)
        if not relationships:
            return [self._default_signal(timestamp, reason="no_significant_relationships")]

        top_relationship = relationships[0]
        strength = float(top_relationship.get("smoothed_correlation", 0.0))
        confidence = float(top_relationship.get("confidence", 0.1))
        state = "active" if abs(strength) >= self._threshold else "weak"

        lineage = build_lineage_record(
            "CORRELATION",
            "sensory.correlation",
            inputs={
                "window": self._window,
                "max_lag": self._max_lag,
                "threshold": self._threshold,
                "series": list(panel.columns),
                "pairs_evaluated": len(relationships),
            },
            outputs={
                "strength": strength,
                "confidence": confidence,
                "state": state,
            },
            telemetry={
                "frequency_seconds": frequency_seconds,
                "top_relationship": top_relationship,
            },
            metadata={
                "timestamp": timestamp.isoformat(),
                "mode": "lag_lead_analysis",
                "relationships": relationships,
            },
        )

        quality = {
            "source": "sensory.correlation",
            "timestamp": timestamp.isoformat(),
            "confidence": confidence,
            "strength": strength,
            "state": state,
            "series": list(panel.columns),
            "samples": int(returns.shape[0]),
        }

        if frequency_seconds is not None:
            quality["frequency_seconds"] = float(frequency_seconds)

        metadata: dict[str, Any] = {
            "source": "sensory.correlation",
            "window": self._window,
            "max_lag": self._max_lag,
            "threshold": self._threshold,
            "state": state,
            "relationships": relationships,
            "quality": quality,
            "lineage": lineage.as_dict(),
        }

        value: dict[str, Any] = {
            "strength": strength,
            "confidence": confidence,
            "state": state,
            "relationships": relationships,
            "window": self._window,
            "max_lag": self._max_lag,
        }
        if frequency_seconds is not None:
            value["frequency_seconds"] = float(frequency_seconds)

        signal = SensorSignal(
            signal_type="CORRELATION",
            value=value,
            confidence=confidence,
            metadata=metadata,
            lineage=lineage,
            timestamp=timestamp,
        )
        return [signal]

    # ------------------------------------------------------------------
    def _analyse_relationships(
        self,
        returns: pd.DataFrame,
        timestamp: datetime,
        frequency_seconds: float | None,
    ) -> list[dict[str, Any]]:
        columns = list(returns.columns)
        if len(columns) < 2:
            return []

        relationships: list[dict[str, Any]] = []
        for idx, name_a in enumerate(columns[:-1]):
            series_a = returns[name_a].dropna()
            if series_a.empty:
                continue
            for name_b in columns[idx + 1 :]:
                series_b = returns[name_b].dropna()
                if series_b.empty:
                    continue
                result = self._best_lag_relationship(
                    name_a,
                    series_a,
                    name_b,
                    series_b,
                    frequency_seconds=frequency_seconds,
                )
                if result is None:
                    continue
                key = self._relationship_key(name_a, name_b)
                smoothed = self._smooth_relationship(key, result, timestamp)
                relationships.append(smoothed)

        relationships.sort(key=lambda entry: abs(float(entry.get("smoothed_correlation", 0.0))), reverse=True)
        return relationships[: self._max_relationships]

    def _best_lag_relationship(
        self,
        name_a: str,
        series_a: pd.Series,
        name_b: str,
        series_b: pd.Series,
        *,
        frequency_seconds: float | None,
    ) -> dict[str, Any] | None:
        best_payload: dict[str, Any] | None = None
        best_score = -math.inf

        for lag in range(-self._max_lag, self._max_lag + 1):
            evaluation = self._correlation_at_lag(series_a, series_b, lag)
            if evaluation is None:
                continue
            corr, samples = evaluation
            score = abs(corr)
            if math.isnan(corr):
                continue
            if score < best_score:
                continue
            leader, follower = self._infer_lead_follow(name_a, name_b, lag)
            lag_seconds = None
            if frequency_seconds is not None:
                lag_seconds = float(lag * frequency_seconds)

            best_payload = {
                "pair": (name_a, name_b),
                "leader": leader,
                "follower": follower,
                "lag": lag,
                "lag_seconds": lag_seconds,
                "correlation": corr,
                "samples": samples,
                "overlap": samples / max(1, min(series_a.size, series_b.size)),
                "significant": abs(corr) >= self._threshold,
            }
            best_score = score

        return best_payload

    def _smooth_relationship(
        self,
        key: str,
        payload: Mapping[str, Any],
        timestamp: datetime,
    ) -> dict[str, Any]:
        previous = self._history.get(key)
        correlation = float(payload.get("correlation", 0.0))
        lag = float(payload.get("lag", 0.0))
        samples = int(payload.get("samples", 0))
        base_confidence = self._relationship_confidence(payload)

        if previous is None or self._smoothing == 0.0:
            smoothed_correlation = correlation
            smoothed_lag = lag
            smoothed_confidence = base_confidence
        else:
            weight = self._smoothing
            smoothed_correlation = (1.0 - weight) * previous.correlation + weight * correlation
            smoothed_lag = (1.0 - weight) * previous.lag + weight * lag
            smoothed_confidence = max(
                base_confidence,
                (1.0 - weight) * previous.confidence + weight * base_confidence,
            )

        state = _RelationshipState(
            correlation=smoothed_correlation,
            lag=smoothed_lag,
            confidence=smoothed_confidence,
            samples=samples,
            updated_at=timestamp,
        )
        self._history[key] = state

        enriched = dict(payload)
        enriched["smoothed_correlation"] = smoothed_correlation
        enriched["smoothed_lag"] = smoothed_lag
        enriched["confidence"] = min(1.0, max(0.0, smoothed_confidence))
        enriched["history"] = {
            "previous_correlation": previous.correlation if previous else None,
            "previous_lag": previous.lag if previous else None,
            "previous_confidence": previous.confidence if previous else None,
            "updated_at": timestamp.isoformat(),
        }
        return enriched

    def _relationship_confidence(self, payload: Mapping[str, Any]) -> float:
        corr = abs(float(payload.get("correlation", 0.0)))
        samples = int(payload.get("samples", 0))
        overlap = float(payload.get("overlap", 0.0))

        sample_factor = min(1.0, samples / max(self._min_overlap, 1))
        overlap_factor = min(1.0, overlap)
        confidence = 0.25 + 0.5 * corr + 0.25 * min(sample_factor, overlap_factor)
        return float(min(1.0, max(0.05, confidence)))

    def _correlation_at_lag(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        lag: int,
    ) -> tuple[float, int] | None:
        if series_a.empty or series_b.empty:
            return None

        shifted_b = series_b.shift(-lag) if lag != 0 else series_b
        aligned = pd.concat([series_a, shifted_b], axis=1, join="inner").dropna()
        if aligned.shape[0] < self._min_overlap:
            return None

        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if corr is None or math.isnan(corr):
            return None
        return float(corr), int(aligned.shape[0])

    def _infer_lead_follow(self, name_a: str, name_b: str, lag: int) -> tuple[str | None, str | None]:
        if lag > 0:
            return name_a, name_b
        if lag < 0:
            return name_b, name_a
        return None, None

    def _relationship_key(self, name_a: str, name_b: str) -> str:
        first, second = sorted((name_a, name_b))
        return f"{first}__{second}"

    def _build_price_panel(self, df: pd.DataFrame) -> pd.DataFrame | None:
        time_column = self._resolve_time_column(df)
        price_column = self._resolve_price_column(df)
        if time_column is None or price_column is None:
            return None

        timestamps = pd.to_datetime(df[time_column], utc=True, errors="coerce")
        prices = pd.to_numeric(df[price_column], errors="coerce")
        labels = self._resolve_series_labels(df)

        payload = pd.DataFrame({
            "timestamp": timestamps,
            "price": prices,
            "series": labels,
        }).dropna(subset=["timestamp", "price"])

        if payload.empty:
            return None

        pivot = (
            payload.pivot_table(
                index="timestamp",
                columns="series",
                values="price",
                aggfunc="last",
            )
            .sort_index()
        )
        pivot = pivot.loc[:, pivot.notna().any(axis=0)]
        return pivot

    def _resolve_time_column(self, df: pd.DataFrame) -> str | None:
        for candidate in ("timestamp", "time", "datetime", "as_of"):
            if candidate in df.columns:
                return candidate
        return None

    def _resolve_price_column(self, df: pd.DataFrame) -> str | None:
        for candidate in ("mid_price", "mid", "price", "last", "close"):
            if candidate in df.columns:
                return candidate
        numeric_columns = [
            column
            for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column]) and column not in {"volume"}
        ]
        return numeric_columns[0] if numeric_columns else None

    def _resolve_series_labels(self, df: pd.DataFrame) -> list[str]:
        symbols = df["symbol"].tolist() if "symbol" in df.columns else [None] * len(df)
        venues = df["venue"].tolist() if "venue" in df.columns else [None] * len(df)
        markets = df["market"].tolist() if "market" in df.columns else [None] * len(df)

        labels: list[str] = []
        for idx in range(len(df)):
            symbol = self._normalise_label(symbols[idx])
            venue = self._normalise_label(venues[idx])
            market = self._normalise_label(markets[idx])

            label_parts = [symbol or "UNKNOWN"]
            if venue and venue != "UNKNOWN":
                label_parts.append(venue)
            elif market and market != "UNKNOWN":
                label_parts.append(market)
            labels.append("@".join(label_parts))
        return labels

    def _normalise_label(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.lower() in {"nan", "none", "null"}:
                return None
            return text
        stringified = str(value).strip()
        if not stringified or stringified.lower() == "nan":
            return None
        return stringified

    def _infer_frequency_seconds(self, index: pd.Index) -> float | None:
        if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
            diffs = index.to_series().diff().dropna()
            if not diffs.empty:
                median = diffs.dt.total_seconds().median()
                if isinstance(median, float) and median > 0:
                    return float(median)
        return None

    def _resolve_timestamp(self, df: pd.DataFrame | None) -> datetime:
        if df is None or df.empty:
            return datetime.now(timezone.utc)
        if "timestamp" in df:
            ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True, errors="coerce")
            if ts is not None and not pd.isna(ts):
                if ts.tzinfo is None:
                    ts = ts.tz_localize(timezone.utc)
                return ts.to_pydatetime()
        return datetime.now(timezone.utc)

    def _default_signal(self, timestamp: datetime, *, reason: str) -> SensorSignal:
        confidence = 0.1
        lineage = build_lineage_record(
            "CORRELATION",
            "sensory.correlation",
            inputs={"reason": reason},
            outputs={"strength": 0.0, "confidence": confidence},
            telemetry={},
            metadata={
                "timestamp": timestamp.isoformat(),
                "mode": "default",
                "reason": reason,
            },
        )
        quality = {
            "source": "sensory.correlation",
            "timestamp": timestamp.isoformat(),
            "confidence": confidence,
            "strength": 0.0,
            "reason": reason,
        }
        metadata: dict[str, Any] = {
            "source": "sensory.correlation",
            "state": "idle",
            "reason": reason,
            "relationships": [],
            "quality": quality,
            "lineage": lineage.as_dict(),
        }
        value = {
            "strength": 0.0,
            "confidence": confidence,
            "state": "idle",
            "relationships": [],
            "reason": reason,
        }
        return SensorSignal(
            signal_type="CORRELATION",
            value=value,
            confidence=confidence,
            metadata=metadata,
            lineage=lineage,
            timestamp=timestamp,
        )

    def status(self) -> Mapping[str, Any]:  # pragma: no cover - diagnostics only
        snapshot: MutableMapping[str, Any] = {}
        for key, state in self._history.items():
            snapshot[key] = {
                "correlation": state.correlation,
                "lag": state.lag,
                "confidence": state.confidence,
                "samples": state.samples,
                "updated_at": state.updated_at.isoformat(),
            }
        return dict(snapshot)
