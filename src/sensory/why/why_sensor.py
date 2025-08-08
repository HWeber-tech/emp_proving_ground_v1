#!/usr/bin/env python3

from __future__ import annotations

from typing import List
import pandas as pd

from src.sensory.signals import SensorSignal


class WhySensor:
    """Macro proxy sensor (WHY dimension).

    For now, derives a simple regime label based on realized volatility:
    higher vol -> bearish bias; lower vol -> neutral/bullish depending on slope.
    """

    def process(self, df: pd.DataFrame) -> List[SensorSignal]:
        if df is None or df.empty or 'close' not in df:
            return [SensorSignal(name='WHY', strength=0.0, confidence=0.1)]

        returns = df['close'].pct_change().dropna()
        vol = returns.rolling(window=20, min_periods=5).std().iloc[-1]
        slope = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] if len(df) >= 20 else 0.0

        strength = 0.0
        confidence = 0.5
        if vol is not None:
            # If volatility is high, be cautious (slightly bearish)
            if vol > 0.02:
                strength = -0.3
                confidence = 0.6
            else:
                # If slope positive, small bullish tilt
                strength = 0.2 if slope > 0 else 0.0
                confidence = 0.5

        return [SensorSignal(name='WHY', strength=float(strength), confidence=float(confidence))]


