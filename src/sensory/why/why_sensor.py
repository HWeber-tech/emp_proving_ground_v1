#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

import pandas as pd

from src.sensory.dimensions.why.yield_signal import YieldSlopeTracker
from src.sensory.why.narrative_hooks import NarrativeEvent, NarrativeHookEngine, NarrativeSummary
from src.sensory.signals import SensorSignal
from src.sensory.lineage import build_lineage_record

_DEFAULT_YIELD_COLUMNS: Mapping[str, str] = {
    "yield_2y": "2Y",
    "yield_02y": "2Y",
    "us02y": "2Y",
    "us2y": "2Y",
    "yield_5y": "5Y",
    "yield_10y": "10Y",
    "yield_30y": "30Y",
    "us10y": "10Y",
    "us30y": "30Y",
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _coerce_curve_value(value: object) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode()
        except Exception:  # pragma: no cover - defensive decoding
            return None
    return None


class WhySensor:
    """Macro proxy sensor (WHY dimension) with yield-curve awareness."""

    def __init__(
        self,
        yield_column_map: Mapping[str, str] | None = None,
        *,
        narrative_engine: NarrativeHookEngine | None = None,
    ) -> None:
        mapping = dict(_DEFAULT_YIELD_COLUMNS)
        if yield_column_map:
            for column, tenor in yield_column_map.items():
                if column:
                    mapping[column] = str(tenor)
        self._yield_columns = mapping
        self._narrative_engine = narrative_engine or NarrativeHookEngine()

    def _extract_yields(self, row: Mapping[str, object]) -> Mapping[str, float | str | None]:
        tracker = YieldSlopeTracker()
        for column, tenor in self._yield_columns.items():
            value = _coerce_curve_value(row.get(column))
            tracker.update(tenor, value)

        raw_curve = row.get("yield_curve")
        if isinstance(raw_curve, Mapping):
            entries: list[tuple[str, float | int | str | None]] = []
            for raw_tenor, raw_value in raw_curve.items():
                if not isinstance(raw_tenor, str):
                    continue
                entries.append((raw_tenor, _coerce_curve_value(raw_value)))
            tracker.update_many(entries)

        return tracker.snapshot().as_dict()

    def process(
        self,
        df: pd.DataFrame,
        *,
        narrative_events: list[NarrativeEvent] | None = None,
        macro_regime_flags: Mapping[str, float] | None = None,
        as_of: datetime | pd.Timestamp | None = None,
    ) -> list[SensorSignal]:
        if df is None or df.empty or "close" not in df:
            return [self._default_signal(reason="no_market_data")]

        last_row = df.iloc[-1]
        returns = df["close"].pct_change().dropna()
        vol = (
            float(returns.rolling(window=20, min_periods=5).std().iloc[-1])
            if not returns.empty
            else 0.0
        )
        slope = 0.0
        if len(df) >= 20:
            base = float(df["close"].iloc[-20]) or 1.0
            slope = float((df["close"].iloc[-1] - df["close"].iloc[-20]) / base)

        macro_bias = float(last_row.get("macro_bias", 0.0) or 0.0)

        base_strength = 0.0
        base_confidence = 0.45
        if vol > 0.02:
            base_strength = -0.35
            base_confidence = 0.6
        else:
            base_strength = 0.25 if slope > 0 else 0.05
            base_confidence = 0.55

        yield_snapshot_dict = dict(self._extract_yields(last_row))
        yield_direction = float(yield_snapshot_dict.get("direction", 0.0) or 0.0)
        yield_confidence = float(yield_snapshot_dict.get("confidence", 0.0) or 0.0)
        slope_2s10s = yield_snapshot_dict.get("slope_2s10s")

        yield_strength = 0.0
        if slope_2s10s is not None:
            yield_strength = yield_direction * _clamp(abs(float(slope_2s10s)) * 8.0, 0.0, 0.75)

        combined_strength = _clamp(
            0.55 * base_strength + 0.25 * yield_strength + 0.20 * _clamp(macro_bias, -1.0, 1.0),
            -1.0,
            1.0,
        )

        combined_confidence = _clamp(
            max(base_confidence, yield_confidence) * 0.6 + 0.4 * _clamp(abs(macro_bias), 0.0, 1.0),
            0.2,
            1.0,
        )

        narrative_summary: NarrativeSummary | None = None
        try:
            as_of_raw = as_of if as_of is not None else last_row.get("timestamp")
            if as_of_raw is None:
                index_value = df.index[-1]
                as_of_raw = index_value if isinstance(index_value, (pd.Timestamp, datetime)) else pd.Timestamp.utcnow()
            as_of_ts = pd.Timestamp(as_of_raw)
            if as_of_ts.tzinfo is None:
                as_of_ts = as_of_ts.tz_localize("UTC")
            narrative_summary = self._narrative_engine.summarise(
                as_of=as_of_ts,
                events=narrative_events,
                macro_flags=macro_regime_flags or {},
            )
        except Exception:
            narrative_summary = None

        if narrative_summary is not None:
            combined_strength = 0.8 * combined_strength + 0.2 * narrative_summary.sentiment_score
            combined_confidence = max(
                combined_confidence,
                0.5 + 0.2 * abs(narrative_summary.sentiment_score),
            )

        metadata: dict[str, object] = {
            "volatility": vol,
            "price_slope": slope,
            "macro_bias": macro_bias,
            "macro_strength": base_strength,
            "macro_confidence": base_confidence,
            "yield_curve": yield_snapshot_dict,
        }
        if narrative_summary is not None:
            metadata["narrative"] = narrative_summary.as_dict()

        value: dict[str, object] = {
            "strength": float(combined_strength),
            "confidence": float(combined_confidence),
        }
        if narrative_summary is not None:
            value["narrative_sentiment"] = float(narrative_summary.sentiment_score)

        timestamp = self._resolve_timestamp(df, as_of)
        quality = {
            "source": "sensory.why",
            "timestamp": timestamp.isoformat(),
            "confidence": float(combined_confidence),
            "strength": float(combined_strength),
        }
        data_quality = self._extract_data_quality(df)
        if data_quality is not None:
            quality["data_quality"] = data_quality

        lineage = build_lineage_record(
            "WHY",
            "sensory.why",
            inputs={
                "volatility": float(vol),
                "price_slope": float(slope),
                "macro_bias": float(macro_bias),
                "yield_direction": float(yield_direction),
                "yield_confidence": float(yield_confidence),
            },
            outputs={
                "strength": float(combined_strength),
                "confidence": float(combined_confidence),
            },
            telemetry={
                "macro_strength": float(base_strength),
                "macro_confidence": float(base_confidence),
                "yield_strength": float(yield_strength),
            },
            metadata={
                "timestamp": timestamp.isoformat(),
                "mode": "macro_yield_fusion",
                "narrative_present": narrative_summary is not None,
            },
        )

        metadata["quality"] = quality
        metadata["lineage"] = lineage.as_dict()

        return [
            SensorSignal(
                signal_type="WHY",
                value=value,
                confidence=float(combined_confidence),
                metadata=metadata,
                lineage=lineage,
            )
        ]

    def _default_signal(
        self,
        *,
        reason: str,
        confidence: float = 0.1,
    ) -> SensorSignal:
        timestamp = datetime.now(timezone.utc)
        lineage = build_lineage_record(
            "WHY",
            "sensory.why",
            inputs={},
            outputs={"strength": 0.0, "confidence": confidence},
            telemetry={},
            metadata={
                "timestamp": timestamp.isoformat(),
                "mode": "default",
                "reason": reason,
            },
        )
        quality = {
            "source": "sensory.why",
            "timestamp": timestamp.isoformat(),
            "confidence": confidence,
            "strength": 0.0,
            "reason": reason,
        }
        metadata: dict[str, object] = {
            "source": "sensory.why",
            "reason": reason,
            "quality": quality,
            "lineage": lineage.as_dict(),
        }
        return SensorSignal(
            signal_type="WHY",
            value={"strength": 0.0, "confidence": confidence},
            confidence=confidence,
            metadata=metadata,
            lineage=lineage,
        )

    def _resolve_timestamp(
        self, df: pd.DataFrame, as_of: datetime | pd.Timestamp | None
    ) -> datetime:
        if as_of is not None:
            ts = pd.Timestamp(as_of)
            if ts.tzinfo is None:
                ts = ts.tz_localize(timezone.utc)
            else:
                ts = ts.tz_convert(timezone.utc)
            return ts.to_pydatetime()
        if not df.empty and "timestamp" in df:
            ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True, errors="coerce")
            if ts is not None and not pd.isna(ts):
                if ts.tzinfo is None:
                    ts = ts.tz_localize(timezone.utc)
                return ts.to_pydatetime()
        return datetime.now(timezone.utc)

    def _extract_data_quality(self, df: pd.DataFrame) -> float | None:
        if "data_quality" not in df or df["data_quality"].empty:
            return None
        try:
            return float(df["data_quality"].iloc[-1])
        except (TypeError, ValueError):
            return None
