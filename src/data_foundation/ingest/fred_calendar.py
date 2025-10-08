"""FRED API integration helpers for macro event ingestion.

The institutional ingest slice relies on macro events to annotate market data
runs with the economic backdrop.  This module implements a lean client around
FRED's calendar endpoint so Timescale ingest plans can pull real release events
whenever an API key is available.  Calls are dependency-light, avoid global
state, and fall back to an empty result set when credentials or network access
are not present.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from typing import Iterable

import requests

from src.data_foundation.schemas import MacroEvent

logger = logging.getLogger(__name__)

_FRED_CALENDAR_URL = "https://api.stlouisfed.org/fred/calendar/releases"
_TIME_SUFFIX = re.compile(r"\s*(ET|CT|MT|PT|UTC|GMT)$", re.IGNORECASE)
_TIME_FORMATS: tuple[str, ...] = ("%I:%M %p", "%H:%M")


def _normalise_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(date_str: str | None, time_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        base = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        logger.debug("Invalid FRED calendar date received: %s", date_str)
        return None

    if time_str:
        cleaned = _TIME_SUFFIX.sub("", time_str.strip())
        for fmt in _TIME_FORMATS:
            try:
                parsed_time = datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
            base = base.replace(hour=parsed_time.hour, minute=parsed_time.minute)
            break

    return base.replace(tzinfo=UTC)


def _iter_release_payload(payload: object | None) -> Iterable[dict[str, object]]:
    if not isinstance(payload, dict):
        return ()
    releases = payload.get("releases")
    if not isinstance(releases, list):
        return ()
    return (item for item in releases if isinstance(item, dict))


def fetch_fred_calendar(
    start: str,
    end: str,
    *,
    api_key: str | None = None,
    session: requests.Session | None = None,
    limit: int = 1000,
) -> list[MacroEvent]:
    """Fetch macro-economic release events from FRED within a date window."""

    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        logger.warning("FRED API key not provided; skipping macro event fetch")
        return []

    close_session = False
    if session is None:
        session = requests.Session()
        close_session = True

    params = {
        "file_type": "json",
        "api_key": key,
        "start": start,
        "end": end,
        "limit": int(limit),
    }

    try:
        response = session.get(_FRED_CALENDAR_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch FRED calendar %s→%s: %s", start, end, exc)
        return []
    except ValueError as exc:
        logger.warning("Invalid JSON from FRED calendar response: %s", exc)
        return []
    finally:
        if close_session:
            try:
                session.close()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.debug("Failed to close FRED session", exc_info=True)

    events: list[MacroEvent] = []
    for item in _iter_release_payload(payload):
        timestamp = _parse_timestamp(
            str(item.get("date")) if item.get("date") is not None else None,
            str(item.get("time")) if item.get("time") is not None else None,
        )
        if timestamp is None:
            continue
        calendar_name = _normalise_str(item.get("release")) or _normalise_str(
            item.get("title")
        ) or "FRED Release"
        event_name = _normalise_str(item.get("event")) or calendar_name
        event = MacroEvent(
            timestamp=timestamp,
            calendar=calendar_name,
            event=event_name,
            currency=_normalise_str(item.get("currency") or item.get("region")),
            actual=_coerce_float(item.get("actual")),
            forecast=_coerce_float(item.get("forecast")),
            previous=_coerce_float(item.get("previous")),
            importance=_normalise_str(item.get("importance") or item.get("importance_level")),
            source="fred",
        )
        events.append(event)

    events.sort(key=lambda entry: entry.timestamp)
    return events


__all__ = ["fetch_fred_calendar"]
