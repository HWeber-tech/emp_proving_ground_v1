"""Regression coverage for the FRED macro calendar fetcher."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
import requests

from src.data_foundation.ingest.fred_calendar import fetch_fred_calendar


class _RecordingSession:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, Any] | None, float | None]] = []
        self.closed = False

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        self.calls.append((url, params, timeout))
        return self._response

    def close(self) -> None:
        self.closed = True


class _StubResponse:
    def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _DummySession:
    def __init__(self) -> None:
        self.called = False

    def get(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - guard
        self.called = True
        raise AssertionError("get() should not be called when API key missing")


def test_fetch_fred_calendar_requires_api_key(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    session = _DummySession()

    with caplog.at_level("WARNING"):
        events = fetch_fred_calendar("2024-01-01", "2024-01-31", session=session)

    assert events == []
    assert not session.called
    assert "FRED API key not provided" in caplog.text


def test_fetch_fred_calendar_parses_release_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    payload = {
        "releases": [
            {
                "release": "Gross Domestic Product",
                "event": "GDP Advance",
                "date": "2024-01-26",
                "time": "08:30 AM ET",
                "currency": "USD",
                "actual": "3.3",
                "forecast": "3.0",
                "previous": "4.9",
                "importance": "High",
            },
            {
                "title": "Nonfarm Payrolls",
                "date": "2024-02-02",
                "time": "08:30",
                "region": "USD",
            },
        ]
    }
    response = _StubResponse(payload)
    session = _RecordingSession(response)

    events = fetch_fred_calendar(
        "2024-01-01",
        "2024-02-10",
        api_key="test-key",
        session=session,
    )

    assert len(events) == 2
    assert events[0].calendar == "Gross Domestic Product"
    assert events[0].event == "GDP Advance"
    assert events[0].currency == "USD"
    assert events[0].importance == "High"
    assert events[0].actual == pytest.approx(3.3)
    assert events[0].timestamp == datetime(2024, 1, 26, 8, 30, tzinfo=UTC)

    assert events[1].calendar == "Nonfarm Payrolls"
    assert events[1].event == "Nonfarm Payrolls"
    assert events[1].currency == "USD"
    assert events[1].timestamp == datetime(2024, 2, 2, 8, 30, tzinfo=UTC)

    assert session.calls
    url, params, timeout = session.calls[0]
    assert "calendar/releases" in url
    assert params is not None
    assert params["start"] == "2024-01-01"
    assert params["end"] == "2024-02-10"
    assert params["api_key"] == "test-key"
    assert timeout == 10
    assert not session.closed


def test_fetch_fred_calendar_handles_http_error(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("FRED_API_KEY", "abc123")

    response = _StubResponse({}, status=503)
    session = _RecordingSession(response)

    with caplog.at_level("WARNING"):
        events = fetch_fred_calendar("2024-01-01", "2024-01-10", session=session)

    assert events == []
    assert "Failed to fetch FRED calendar" in caplog.text
