from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.sensory.when.session_analytics import SessionAnalytics, SessionAnalyticsConfig


def test_session_analytics_identifies_overlap() -> None:
    analytics = SessionAnalytics()
    timestamp = pd.Timestamp(datetime(2024, 1, 2, 13, 30, tzinfo=timezone.utc))

    snapshot = analytics.analyse(timestamp)

    assert snapshot.intensity >= 0.85
    assert set(snapshot.active_sessions) == {"London", "NY"}
    assert snapshot.session_token in {"London", "NY"}
    assert snapshot.minutes_to_session_close is not None


def test_session_analytics_tracks_upcoming_session() -> None:
    config = SessionAnalyticsConfig()
    analytics = SessionAnalytics(config)
    timestamp = pd.Timestamp(datetime(2024, 1, 2, 5, 0, tzinfo=timezone.utc))

    snapshot = analytics.analyse(timestamp)

    assert snapshot.active_sessions == ("Asia",)
    assert snapshot.minutes_to_next_session is not None
    assert snapshot.minutes_to_session_close is not None
    assert snapshot.session_token == "Asia"


def test_session_analytics_builds_anticipation_before_open() -> None:
    analytics = SessionAnalytics()
    timestamp = pd.Timestamp(datetime(2024, 1, 2, 6, 30, tzinfo=timezone.utc))

    snapshot = analytics.analyse(timestamp)

    assert snapshot.intensity > 0.1
    assert snapshot.upcoming_session in {"London", "Asia", "NY"}
    assert snapshot.session_token == "auction_close"
