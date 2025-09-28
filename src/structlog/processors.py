"""Simplified processors mirroring the :mod:`structlog.processors` API."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import MutableMapping


class TimeStamper:
    """Add an ISO8601 timestamp to the event dictionary."""

    def __init__(self, *, fmt: str = "iso", utc: bool = False) -> None:
        self._fmt = fmt
        self._utc = utc

    def __call__(
        self,
        _logger: object,
        _method_name: str,
        event_dict: MutableMapping[str, object],
    ) -> MutableMapping[str, object]:
        timestamp = datetime.now(tz=UTC if self._utc else None)
        if self._fmt == "iso":
            event_dict.setdefault("timestamp", timestamp.isoformat())
        else:  # pragma: no cover - alternative formats are out of scope
            event_dict.setdefault("timestamp", timestamp.strftime(self._fmt))
        return event_dict


class JSONRenderer:
    """Render the event dictionary as a JSON encoded string."""

    def __call__(
        self,
        _logger: object,
        _method_name: str,
        event_dict: MutableMapping[str, object],
    ) -> str:
        return json.dumps(event_dict, default=str, sort_keys=True)


__all__ = [
    "TimeStamper",
    "JSONRenderer",
]
