#!/usr/bin/env python3

from __future__ import annotations

from typing import List

import pandas as pd

from src.sensory.signals import SensorSignal


class WhatSensor:
    """Pattern sensor (WHAT dimension).

    Placeholder: detect simple breakout above/below rolling range.
    """

    def process(self, df: pd.DataFrame) -> List[SensorSignal]:
        if df is None or df.empty or 'close' not in df:
            return [SensorSignal(name='WHAT', strength=0.0, confidence=0.1)]

        window = 20
        if len(df) < window:
            return [SensorSignal(name='WHAT', strength=0.0, confidence=0.2)]

        recent = df.tail(window)
        high = recent['close'].max()
        low = recent['close'].min()
        last = df['close'].iloc[-1]

        if last >= high:
            return [SensorSignal(name='WHAT', strength=0.6, confidence=0.6)]
        if last <= low:
            return [SensorSignal(name='WHAT', strength=-0.6, confidence=0.6)]
        return [SensorSignal(name='WHAT', strength=0.0, confidence=0.3)]


