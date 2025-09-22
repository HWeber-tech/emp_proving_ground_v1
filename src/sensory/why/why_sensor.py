#!/usr/bin/env python3

from __future__ import annotations

from typing import Mapping

import pandas as pd

from src.sensory.dimensions.why.yield_signal import YieldSlopeTracker
from src.sensory.signals import SensorSignal

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


class WhySensor:
    """Macro proxy sensor (WHY dimension) with yield-curve awareness."""

    def __init__(self, yield_column_map: Mapping[str, str] | None = None) -> None:
        mapping = dict(_DEFAULT_YIELD_COLUMNS)
        if yield_column_map:
            for column, tenor in yield_column_map.items():
                if column:
                    mapping[column] = str(tenor)
        self._yield_columns = mapping

    def _extract_yields(self, row: Mapping[str, object]) -> dict[str, float | str | None]:
        tracker = YieldSlopeTracker()
        for column, tenor in self._yield_columns.items():
            value = row.get(column)
            tracker.update(tenor, value)

        raw_curve = row.get("yield_curve")
        if isinstance(raw_curve, Mapping):
            tracker.update_many(raw_curve)  # type: ignore[arg-type]

        return tracker.snapshot().as_dict()

    def process(self, df: pd.DataFrame) -> list[SensorSignal]:
        if df is None or df.empty or "close" not in df:
            return [SensorSignal(signal_type="WHY", value={"strength": 0.0}, confidence=0.1)]

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

        yield_snapshot_dict = self._extract_yields(last_row)
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

        metadata = {
            "volatility": vol,
            "price_slope": slope,
            "macro_bias": macro_bias,
            "macro_strength": base_strength,
            "macro_confidence": base_confidence,
            "yield_curve": yield_snapshot_dict,
        }

        return [
            SensorSignal(
                signal_type="WHY",
                value={"strength": float(combined_strength)},
                confidence=float(combined_confidence),
                metadata=metadata,
            )
        ]
