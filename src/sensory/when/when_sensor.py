#!/usr/bin/env python3

from __future__ import annotations

from typing import List

import pandas as pd

from src.sensory.signals import SensorSignal


class WhenSensor:
    """Temporal/context sensor (WHEN dimension).

    Placeholder: active if intraday time window (e.g., overlap of EU/US sessions) and
    realized volatility above a small threshold.
    """

    def process(self, df: pd.DataFrame) -> List[SensorSignal]:
        if df is None or df.empty or 'close' not in df or 'timestamp' not in df:
            return [SensorSignal(name='WHEN', strength=0.0, confidence=0.1)]

        # Use last timestamp hour
        ts = pd.to_datetime(df['timestamp'].iloc[-1])
        hour = ts.hour
        session_active = 12 <= hour <= 20  # rough EU/US overlap UTC

        returns = df['close'].pct_change().dropna()
        vol = returns.rolling(window=20, min_periods=5).std().iloc[-1] if not returns.empty else 0.0

        if session_active and vol and vol > 0.003:
            return [SensorSignal(name='WHEN', strength=0.3, confidence=0.6)]
        return [SensorSignal(name='WHEN', strength=0.0, confidence=0.3)]


