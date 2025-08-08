#!/usr/bin/env python3

from __future__ import annotations

from typing import List
import pandas as pd

from src.sensory.signals import SensorSignal


class HowSensor:
    """Institutional footprint heuristic (HOW dimension).

    Placeholder: detect fair-value gap-like conditions via large candle bodies
    and small overlap with previous candle.
    """

    def process(self, df: pd.DataFrame) -> List[SensorSignal]:
        if df is None or df.empty:
            return [SensorSignal(name='HOW', strength=0.0, confidence=0.1)]

        if not set(['open', 'high', 'low', 'close']).issubset(df.columns):
            return [SensorSignal(name='HOW', strength=0.0, confidence=0.1)]

        recent = df.tail(2)
        if len(recent) < 2:
            return [SensorSignal(name='HOW', strength=0.0, confidence=0.1)]

        prev = recent.iloc[-2]
        curr = recent.iloc[-1]

        body = abs(curr['close'] - curr['open'])
        prev_body = abs(prev['close'] - prev['open'])
        overlap_low = max(min(curr['open'], curr['close']), min(prev['open'], prev['close']))
        overlap_high = min(max(curr['open'], curr['close']), max(prev['open'], prev['close']))
        overlap = max(0.0, float(overlap_high - overlap_low))

        # Heuristic: large body and small overlap -> momentum footprint
        if body > prev_body * 1.5 and overlap < body * 0.2:
            strength = 0.5 if curr['close'] > curr['open'] else -0.5
            confidence = 0.6
        else:
            strength = 0.0
            confidence = 0.3

        return [SensorSignal(name='HOW', strength=float(strength), confidence=float(confidence))]


