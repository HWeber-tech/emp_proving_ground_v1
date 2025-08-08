#!/usr/bin/env python3

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.sensory.why.why_sensor import WhySensor
from src.sensory.how.how_sensor import HowSensor
from src.sensory.what.what_sensor import WhatSensor
from src.sensory.when.when_sensor import WhenSensor
from src.sensory.anomaly.anomaly_sensor import AnomalySensor
from src.sensory.signals import SensorSignal


def _mk_df(n: int = 60, up: bool = True) -> pd.DataFrame:
    now = datetime.utcnow()
    base = 1.10
    step = 0.0005 if up else -0.0005
    closes = np.array([base + i * step for i in range(n)])
    opens = closes - step / 2
    highs = closes + 0.0006
    lows = closes - 0.0006
    vols = np.linspace(1000, 3000, n)
    ts = [now - timedelta(minutes=n - i) for i in range(n)]
    return pd.DataFrame({
        'timestamp': ts,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': vols,
    })


def test_sensors_basic_outputs():
    df = _mk_df()
    sensors = [WhySensor(), HowSensor(), WhatSensor(), WhenSensor(), AnomalySensor()]
    for sensor in sensors:
        signals = sensor.process(df)
        assert isinstance(signals, list)
        assert all(isinstance(s, SensorSignal) for s in signals)
        for s in signals:
            assert -1.0 <= s.strength <= 1.0
            assert 0.0 <= s.confidence <= 1.0


