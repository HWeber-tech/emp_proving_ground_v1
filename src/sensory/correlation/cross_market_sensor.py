from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from src.sensory.lineage import build_lineage_record
from src.sensory.signals import SensorSignal

__all__ = ["CrossMarketCorrelationSensor", "CrossMarketCorrelationConfig"]


@dataclass(slots=True)
class CrossMarketCorrelationConfig:
    """Configuration for the cross-market correlation sensor."""

    window: int = 240
    max_lag: int = 6
    min_samples: int = 30
    min_correlation: float = 0.4
    top_relationships: int = 5

    def effective_window(self) -> int:
        return max(self.window, self.min_samples)


class CrossMarketCorrelationSensor:
    """Estimate lead/lag structure between correlated assets and venues."""

    def __init__(self, config: CrossMarketCorrelationConfig | None = None) -> None:
        self._config = config or CrossMarketCorrelationConfig()

    @property
    def config(self) -> CrossMarketCorrelationConfig:
        return self._config

    def process(self, data: pd.DataFrame | Mapping[str, Any] | None) -> list[SensorSignal]:
        frame = self._normalise_input(data)
        if frame is None or frame.empty:
            return [self._default_signal(reason="no_market_data")]

        timestamp = self._extract_timestamp(frame)
        value_frame = self._extract_value_frame(frame)
        if value_frame is None or value_frame.empty or value_frame.shape[1] < 2:
            return [self._default_signal(reason="insufficient_series", timestamp=timestamp)]

        relationships = self._compute_relationships(value_frame)
        if not relationships:
            return [
                self._default_signal(
                    reason="no_significant_relationships",
                    timestamp=timestamp,
                    extra_metadata={"series": list(value_frame.columns)},
                )
            ]

        top_n = max(1, self._config.top_relationships)
        top_relationships = [dict(item) for item in relationships[:top_n]]

        top_strengths = [item["strength"] for item in top_relationships]
        mean_strength = float(np.mean(top_strengths)) if top_strengths else 0.0
        max_strength = float(top_strengths[0]) if top_strengths else 0.0
        dominant = dict(top_relationships[0])
        dominant_samples = int(dominant.get("samples", self._config.min_samples))
        confidence = self._resolve_confidence(max_strength, dominant_samples)

        quality = {
            "source": "sensory.cross_market_correlation",
            "timestamp": timestamp.isoformat(),
            "confidence": float(confidence),
            "strength": float(mean_strength),
            "relationships": len(relationships),
        }

        lineage = build_lineage_record(
            "CORRELATION",
            "sensory.cross_market_correlation",
            inputs={
                "series": list(value_frame.columns),
                "window": int(value_frame.shape[0]),
                "max_lag": int(self._config.max_lag),
                "min_samples": int(self._config.min_samples),
            },
            outputs={
                "confidence": float(confidence),
                "top_strength": float(max_strength),
                "relationships": len(relationships),
            },
            telemetry={
                "top_relationships": top_relationships,
                "mean_strength": mean_strength,
            },
            metadata={
                "timestamp": timestamp.isoformat(),
                "window": int(value_frame.shape[0]),
                "max_lag": int(self._config.max_lag),
            },
        )

        metadata: MutableMapping[str, Any] = {
            "source": "sensory.cross_market_correlation",
            "relationships": top_relationships,
            "window": int(value_frame.shape[0]),
            "max_lag": int(self._config.max_lag),
            "min_correlation": float(self._config.min_correlation),
            "quality": quality,
            "lineage": lineage.as_dict(),
        }

        value: dict[str, Any] = {
            "dominant_relationship": dominant,
            "top_relationships": top_relationships,
            "mean_strength": mean_strength,
            "observed_series": list(value_frame.columns),
        }

        signal = SensorSignal(
            signal_type="CROSS_MARKET_CORRELATION",
            value=value,
            confidence=confidence,
            metadata=dict(metadata),
            lineage=lineage,
        )
        return [signal]

    # ------------------------------------------------------------------
    def _compute_relationships(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        columns = list(frame.columns)
        max_lag = max(0, int(self._config.max_lag))
        min_samples = max(2, int(self._config.min_samples))
        min_corr = abs(float(self._config.min_correlation))

        relationships: list[dict[str, Any]] = []
        for idx, left in enumerate(columns):
            series_left = frame[left]
            for right in columns[idx + 1 :]:
                series_right = frame[right]
                best = self._best_pair_alignment(
                    left,
                    series_left,
                    right,
                    series_right,
                    max_lag=max_lag,
                    min_samples=min_samples,
                )
                if not best:
                    continue
                if best["strength"] < min_corr:
                    continue
                relationships.append(dict(best))

        relationships.sort(key=lambda item: item["strength"], reverse=True)
        return relationships

    def _best_pair_alignment(
        self,
        name_a: str,
        series_a: pd.Series,
        name_b: str,
        series_b: pd.Series,
        *,
        max_lag: int,
        min_samples: int,
    ) -> dict[str, Any] | None:
        best_result: dict[str, Any] | None = None
        for lag in range(-max_lag, max_lag + 1):
            shifted = pd.concat(
                [series_a, series_b.shift(lag)],
                axis=1,
                join="inner",
            ).dropna(how="any")
            if shifted.empty or len(shifted) < min_samples:
                continue
            corr = shifted.iloc[:, 0].corr(shifted.iloc[:, 1])
            if corr is None or not np.isfinite(corr):
                continue
            strength = abs(float(corr))
            if best_result is not None and strength <= best_result["strength"]:
                continue
            leader, follower, lead_steps = self._interpret_lag(
                name_a,
                name_b,
                lag,
            )
            best_result = {
                "pair": (name_a, name_b),
                "lag": int(lag),
                "leader": leader,
                "follower": follower,
                "lead_steps": int(lead_steps),
                "correlation": float(corr),
                "strength": strength,
                "samples": int(len(shifted)),
            }
        return best_result

    def _interpret_lag(
        self,
        name_a: str,
        name_b: str,
        lag: int,
    ) -> tuple[str, str | None, int]:
        if lag < 0:
            return (name_a, name_b, abs(lag))
        if lag > 0:
            return (name_b, name_a, lag)
        return ("synchronous", None, 0)

    def _resolve_confidence(self, strength: float, samples: int) -> float:
        strength = float(max(0.0, min(1.0, strength)))
        samples = max(1, samples)
        decay = 1.0 - exp(-samples / 60.0)
        confidence = max(0.15, min(1.0, strength * (0.6 + 0.4 * decay)))
        return confidence

    def _extract_value_frame(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(axis=0, how="all")
        numeric = numeric.dropna(axis=1, how="all")
        window = self._config.effective_window()
        if window > 0:
            numeric = numeric.tail(window)
        valid_columns = [
            col for col in numeric.columns if numeric[col].dropna().shape[0] >= self._config.min_samples
        ]
        return numeric[valid_columns] if valid_columns else None

    def _normalise_input(
        self, data: pd.DataFrame | Mapping[str, Any] | None
    ) -> pd.DataFrame | None:
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif isinstance(data, Mapping):
            frame = self._mapping_to_frame(data)
        else:
            return None

        if frame is None or frame.empty:
            return frame

        if "timestamp" in frame.columns:
            ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.assign(timestamp=ts)
            frame = frame.dropna(subset=["timestamp"])
            frame = frame.sort_values("timestamp")
            frame = frame.set_index("timestamp", drop=True)
        elif isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.sort_index()
        return frame

    def _mapping_to_frame(self, mapping: Mapping[str, Any]) -> pd.DataFrame | None:
        series_map: MutableMapping[str, pd.Series] = {}
        for raw_key, raw_value in mapping.items():
            series = self._coerce_series(raw_value)
            if series is None or series.dropna().empty:
                continue
            name = self._normalise_label(raw_key)
            series_map[name] = series
        if not series_map:
            return None
        frame = pd.DataFrame(series_map)
        return frame

    def _coerce_series(self, value: Any) -> pd.Series | None:
        if isinstance(value, pd.Series):
            return value.astype(float, copy=False)
        if isinstance(value, pd.DataFrame):
            for candidate in ("close", "price", "mid", "value"):
                if candidate in value.columns:
                    return value[candidate].astype(float, copy=False)
            numeric_cols = value.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                return value[numeric_cols[0]].astype(float, copy=False)
            return None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            try:
                series = pd.Series(list(value), dtype=float)
            except (TypeError, ValueError):
                return None
            return series
        return None

    def _normalise_label(self, label: Any) -> str:
        if isinstance(label, tuple):
            return "::".join(str(part) for part in label if part is not None)
        return str(label)

    def _extract_timestamp(self, frame: pd.DataFrame) -> datetime:
        if isinstance(frame.index, pd.DatetimeIndex) and not frame.index.empty:
            ts = frame.index[-1]
            if ts.tzinfo is None:
                ts = ts.tz_localize(timezone.utc)
            else:
                ts = ts.tz_convert(timezone.utc)
            return ts.to_pydatetime()
        for column in ("timestamp", "time", "datetime"):
            if column in frame.columns:
                series = pd.to_datetime(frame[column], utc=True, errors="coerce")
                series = series.dropna()
                if not series.empty:
                    return series.iloc[-1].to_pydatetime()
        return datetime.now(timezone.utc)

    def _default_signal(
        self,
        *,
        reason: str,
        timestamp: datetime | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> SensorSignal:
        ts = timestamp or datetime.now(timezone.utc)
        lineage = build_lineage_record(
            "CORRELATION",
            "sensory.cross_market_correlation",
            inputs={"reason": reason},
            outputs={"confidence": 0.1, "strength": 0.0},
            metadata={"timestamp": ts.isoformat(), "mode": "default"},
        )
        base_metadata: MutableMapping[str, Any] = {
            "source": "sensory.cross_market_correlation",
            "reason": reason,
            "quality": {
                "source": "sensory.cross_market_correlation",
                "timestamp": ts.isoformat(),
                "confidence": 0.1,
                "strength": 0.0,
                "reason": reason,
            },
            "lineage": lineage.as_dict(),
        }
        if extra_metadata:
            base_metadata.update({str(k): v for k, v in extra_metadata.items()})
        observed_series: list[str] = []
        if extra_metadata and "series" in extra_metadata:
            series_payload = extra_metadata["series"]
            if isinstance(series_payload, Sequence) and not isinstance(
                series_payload, (str, bytes, bytearray)
            ):
                observed_series = [str(item) for item in series_payload]
            else:
                observed_series = [str(series_payload)]
        return SensorSignal(
            signal_type="CROSS_MARKET_CORRELATION",
            value={
                "dominant_relationship": None,
                "top_relationships": [],
                "mean_strength": 0.0,
                "observed_series": observed_series,
            },
            confidence=0.1,
            metadata=dict(base_metadata),
            lineage=lineage,
        )
