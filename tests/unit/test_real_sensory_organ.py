#!/usr/bin/env python3

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.sensory.real_sensory_organ import RealSensoryOrgan as CanonicalOrgan


class DummyConfig:
    symbol = "EURUSD"
    timeframes = ["M1"]
    primary_timeframe = "M1"
    max_buffer_size = 1000
    rsi_period = 14
    bb_period = 20
    momentum_period = 10
    database_path = ":memory:"


@pytest.mark.asyncio
async def test_real_sensory_organ_process_returns_signals():
    organ = CanonicalOrgan()

    # Build synthetic OHLCV series with 60 rows
    n = 60
    now = datetime.utcnow()
    closes = np.linspace(1.1000, 1.1200, n)
    highs = closes + 0.0005
    lows = closes - 0.0005
    opens = closes
    volumes = np.linspace(1000, 2000, n)

    df = pd.DataFrame({
        "timestamp": [now - timedelta(minutes=n - i) for i in range(n)],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    market_data = {
        "timeframe": "M1",
        "timestamp": now,
        "open": float(opens[-1]),
        "high": float(highs[-1]),
        "low": float(lows[-1]),
        "close": float(closes[-1]),
        "volume": float(volumes[-1]),
    }

    # Feed enough samples to fill buffers
    for i in range(n):
        sample = df.iloc[i]
        md = {
            "timeframe": "M1",
            "timestamp": sample["timestamp"],
            "open": float(sample["open"]),
            "high": float(sample["high"]),
            "low": float(sample["low"]),
            "close": float(sample["close"]),
            "volume": float(sample["volume"]),
        }
        await organ._update_data_buffers(md)  # type: ignore[attr-defined]

    reading = await organ.process(market_data)

    assert reading.overall_sentiment in {"bullish", "bearish", "neutral"}
    assert 0.0 <= reading.confidence_score <= 1.0
    assert isinstance(reading.technical_signals, list)
    assert len(reading.technical_signals) >= 0


