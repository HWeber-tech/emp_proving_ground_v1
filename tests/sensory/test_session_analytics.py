from datetime import datetime, timezone

from src.sensory.when.session_analytics import (
    DEFAULT_CALENDAR,
    SessionWindow,
    analyse_session,
)


def test_analyse_session_overlap_intensity() -> None:
    ts = datetime(2024, 1, 1, 13, tzinfo=timezone.utc)

    summary = analyse_session(ts)

    assert "London" in summary.active_sessions
    assert "NewYork" in summary.active_sessions
    assert summary.intensity == 1.0
    assert summary.minutes_to_close > 0
    assert summary.as_metadata()["active_sessions"]


def test_analyse_session_custom_calendar() -> None:
    calendar = (
        SessionWindow(name="Crypto", start_hour=0, end_hour=24),
    )
    ts = datetime(2024, 1, 1, 5, tzinfo=timezone.utc)

    summary = analyse_session(ts, calendar=calendar)

    assert summary.active_sessions == ("Crypto",)
    assert summary.minutes_to_close > 0
    assert summary.minutes_to_next_open == summary.minutes_to_close


def test_next_session_resolution() -> None:
    ts = datetime(2024, 1, 1, 21, tzinfo=timezone.utc)

    summary = analyse_session(ts, calendar=DEFAULT_CALENDAR)

    assert summary.active_sessions == ()
    assert "Asia" in summary.next_sessions
    assert summary.minutes_to_next_open == 60.0
