"""Simplified processors mirroring the :mod:`structlog.processors` API."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Callable, MutableMapping


class TimeStamper:
    """Add an ISO8601 timestamp to the event dictionary."""

    def __init__(
        self,
        *,
        fmt: str = "iso",
        utc: bool = False,
        key: str = "timestamp",
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self._fmt = fmt
        self._is_iso = fmt.lower() == "iso"
        self._utc = utc
        self._key = key
        self._now_factory = now_factory

    def __call__(
        self,
        _logger: object,
        _method_name: str,
        event_dict: MutableMapping[str, object],
    ) -> MutableMapping[str, object]:
        if self._key in event_dict:
            return event_dict

        if self._now_factory is not None:
            timestamp = self._now_factory()
        else:
            timestamp = datetime.now(tz=UTC if self._utc else None)
        if self._utc:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            else:
                timestamp = timestamp.astimezone(UTC)
        if self._is_iso:
            rendered = timestamp.isoformat()
            if self._utc and rendered.endswith("+00:00"):
                rendered = rendered[:-6] + "Z"
            event_dict.setdefault(self._key, rendered)
        else:  # pragma: no cover - alternative formats are out of scope
            event_dict.setdefault(self._key, timestamp.strftime(self._fmt))
        return event_dict


class JSONRenderer:
    """Render the event dictionary as a JSON encoded string."""

    def __init__(self, *, sort_keys: bool = True) -> None:
        self._sort_keys = sort_keys

    def __call__(
        self,
        _logger: object,
        _method_name: str,
        event_dict: MutableMapping[str, object],
    ) -> str:
        return json.dumps(event_dict, default=str, sort_keys=self._sort_keys)


__all__ = [
    "TimeStamper",
    "JSONRenderer",
]
