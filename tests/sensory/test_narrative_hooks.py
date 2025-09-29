from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.sensory.why.narrative_hooks import NarrativeEvent, NarrativeHookEngine
from src.sensory.why.why_sensor import WhySensor


def test_narrative_engine_prioritises_upcoming_events() -> None:
    engine = NarrativeHookEngine()
    now = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)
    events = [
        NarrativeEvent(timestamp=now - timedelta(hours=2), sentiment=-0.4, importance=1.0, description="CPI miss"),
        NarrativeEvent(timestamp=now + timedelta(hours=1), sentiment=0.6, importance=1.5, description="Fed presser"),
    ]

    summary = engine.summarise(as_of=now, events=events, macro_flags={"growth": 0.3})

    assert summary.sentiment_score > 0
    assert summary.dominant_theme == "growth"
    assert summary.upcoming_event is not None
    assert summary.upcoming_event["description"] == "Fed presser"


def test_why_sensor_blends_narrative_sentiment() -> None:
    now = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105],
            "open": [99, 100, 101, 102, 103, 104],
            "macro_bias": [0.15, 0.15, 0.2, 0.2, 0.25, 0.3],
            "yield_2y": [0.021, 0.022, 0.023, 0.024, 0.025, 0.026],
            "yield_5y": [0.024, 0.025, 0.026, 0.027, 0.028, 0.029],
            "yield_10y": [0.028, 0.029, 0.030, 0.031, 0.032, 0.033],
            "yield_30y": [0.031, 0.032, 0.033, 0.034, 0.035, 0.036],
        }
    )

    sensor = WhySensor()
    events = [
        NarrativeEvent(timestamp=now + timedelta(minutes=30), sentiment=0.8, importance=2.0, description="Jobs report"),
        NarrativeEvent(timestamp=now + timedelta(hours=3), sentiment=-0.2, importance=0.5, description="Oil inventory"),
    ]

    signals = sensor.process(
        df,
        narrative_events=events,
        macro_regime_flags={"growth": 0.2, "inflation": -0.1},
        as_of=now,
    )

    assert len(signals) == 1
    signal = signals[0]
    assert signal.value.get("narrative_sentiment") is not None
    narrative_meta = signal.metadata.get("narrative")
    assert isinstance(narrative_meta, dict)
    assert narrative_meta.get("upcoming_event") is not None

