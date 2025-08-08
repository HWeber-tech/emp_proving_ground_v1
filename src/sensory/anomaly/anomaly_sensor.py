#!/usr/bin/env python3

from __future__ import annotations

from typing import List
import pandas as pd

from src.sensory.signals import SensorSignal


class AnomalySensor:
    """Anomaly sensor (ANOMALY dimension).

    Placeholder: flag large absolute return or volume spike vs rolling mean.
    """

    def process(self, df: pd.DataFrame) -> List[SensorSignal]:
        if df is None or df.empty or 'close' not in df:
            return [SensorSignal(name='ANOMALY', strength=0.0, confidence=0.1)]

        signals: List[SensorSignal] = []
        returns = df['close'].pct_change().dropna()
        if not returns.empty:
            r = float(abs(returns.iloc[-1]))
            rolling = float(returns.rolling(window=20, min_periods=5).std().iloc[-1] or 0.0)
            if rolling > 0 and r > 3 * rolling:
                signals.append(SensorSignal(name='ANOMALY_return', strength=0.0, confidence=0.8))

        if 'volume' in df:
            vol = float(df['volume'].iloc[-1])
            vol_ma = float(df['volume'].rolling(window=20, min_periods=5).mean().iloc[-1] or 0.0)
            if vol_ma > 0 and vol > 3 * vol_ma:
                signals.append(SensorSignal(name='ANOMALY_volume', strength=0.0, confidence=0.7))

        if not signals:
            signals.append(SensorSignal(name='ANOMALY', strength=0.0, confidence=0.2))
        return signals


