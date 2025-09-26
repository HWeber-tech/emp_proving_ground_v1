"""Redis-backed caching helpers for Timescale reader queries.

These utilities bridge the institutional Redis cache to the Timescale
read-path so frequently accessed slices (latest daily bars, most recent
intraday trades, macro calendars) can be served without repeatedly hitting
the database.  The roadmap calls for Redis caching to accompany the
Timescale + Kafka ingest vertical; this module centralises the
serialisation and cache-key discipline required to make that possible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Callable, Mapping, Sequence, cast

import pandas as pd

from .redis_cache import ManagedRedisCache
from ..persist.timescale_reader import TimescaleQueryResult, TimescaleReader

logger = logging.getLogger(__name__)


def _normalise_subjects(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return tuple()
    return tuple(sorted({str(value).strip().upper() for value in values if str(value).strip()}))


def _normalise_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        converted = timestamp.tz_convert("UTC").to_pydatetime()
        if isinstance(converted, datetime):
            return converted
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return None


def _isoformat(value: object) -> str | None:
    dt = _normalise_datetime(value)
    return dt.isoformat() if dt is not None else None


def _parse_datetime(value: object) -> datetime | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, datetime):
        return _normalise_datetime(value)
    if isinstance(value, pd.Timestamp):
        return _normalise_datetime(value)
    try:
        text = str(value)
    except Exception:  # pragma: no cover - defensive guard
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _normalise_datetime(parsed)


def _render_signature(payload: Mapping[str, object]) -> tuple[str, str]:
    parts = [f"{key}={payload[key]}" for key in sorted(payload)]
    raw = "|".join(parts)
    digest = sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"timescale:{digest}", raw


def _serialise_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return json.dumps({"columns": list(frame.columns), "index": [], "data": []})
    serialised = frame.to_json(orient="split", date_format="iso", date_unit="ns")
    return serialised if isinstance(serialised, str) else str(serialised)


def _deserialise_frame(payload: Mapping[str, object]) -> pd.DataFrame:
    if "data" in payload and "columns" in payload:
        frame = pd.DataFrame(**payload)
    else:
        text = payload.get("frame")
        if not isinstance(text, str):
            raise ValueError("Expected frame serialisation to be a string")
        frame = pd.read_json(text, orient="split")
    for column in ("ts", "timestamp", "ingested_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def _serialise_result(result: TimescaleQueryResult, signature: str, raw_signature: str) -> str:
    payload = {
        "version": 1,
        "dimension": result.dimension,
        "frame": _serialise_frame(result.frame),
        "symbols": list(result.symbols),
        "start_ts": _isoformat(result.start_ts),
        "end_ts": _isoformat(result.end_ts),
        "max_ingested_at": _isoformat(result.max_ingested_at),
        "signature": signature,
        "raw_signature": raw_signature,
    }
    return json.dumps(payload, separators=(",", ":"))


def _deserialise_result(serialised: str) -> TimescaleQueryResult:
    data = cast("dict[str, Any]", json.loads(serialised))
    frame_payload = data.get("frame")
    if isinstance(frame_payload, str):
        frame = pd.read_json(frame_payload, orient="split")
    elif isinstance(frame_payload, Mapping):  # pragma: no cover - legacy guard
        frame = _deserialise_frame(frame_payload)
    else:
        raise ValueError("Invalid frame payload in cache entry")

    for column in ("ts", "timestamp", "ingested_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    return TimescaleQueryResult(
        dimension=str(data.get("dimension", "unknown")),
        frame=frame,
        symbols=tuple(data.get("symbols", ())),
        start_ts=_parse_datetime(data.get("start_ts")),
        end_ts=_parse_datetime(data.get("end_ts")),
        max_ingested_at=_parse_datetime(data.get("max_ingested_at")),
    )


FetchFn = Callable[..., TimescaleQueryResult]


@dataclass
class TimescaleQueryCache:
    """Cache Timescale reader results using :class:`ManagedRedisCache`.

    The cache consumes query parameters, builds a deterministic fingerprint,
    and stores the full :class:`TimescaleQueryResult` serialised as JSON. On
    cache hits the stored frame is reconstructed into a pandas DataFrame.
    """

    reader: TimescaleReader
    cache: ManagedRedisCache | None

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.TimescaleQueryCache")

    @property
    def enabled(self) -> bool:
        return self.cache is not None

    def _fetch(
        self,
        *,
        dimension: str,
        fetcher: FetchFn,
        subjects: Sequence[str] | None,
        start: object,
        end: object,
        limit: int | None,
        subject_param: str,
        extra: Mapping[str, object] | None = None,
    ) -> TimescaleQueryResult:
        if self.cache is None:
            call_kwargs: dict[str, object | None] = {
                subject_param: list(subjects) if subjects is not None else None,
                "start": start,
                "end": end,
                "limit": limit,
            }
            return fetcher(**call_kwargs)

        canonical_subjects = _normalise_subjects(subjects)
        signature_payload: dict[str, object] = {
            "dimension": dimension,
            "subjects": ",".join(canonical_subjects) or "ALL",
            "start": _isoformat(start) or "-",
            "end": _isoformat(end) or "-",
            "limit": limit if limit is not None else "-",
        }
        if extra:
            signature_payload.update({str(k): extra[k] for k in extra})

        signature, raw_signature = _render_signature(signature_payload)
        cache_key = f"timescale:{dimension}:{signature}"  # namespace applied by ManagedRedisCache

        cached = self.cache.get(cache_key)
        if cached is not None:
            try:
                serialised = (
                    cached.decode("utf-8")
                    if isinstance(cached, (bytes, bytearray))
                    else str(cached)
                )
                return _deserialise_result(serialised)
            except Exception:
                self._logger.debug(
                    "Failed to deserialize Timescale cache entry; purging", exc_info=True
                )
                self.cache.delete(cache_key)

        call_kwargs = {
            subject_param: list(subjects) if subjects is not None else None,
            "start": start,
            "end": end,
            "limit": limit,
        }
        result = fetcher(**call_kwargs)
        try:
            payload = _serialise_result(result, signature, raw_signature)
            self.cache.set(cache_key, payload)
        except Exception:
            self._logger.debug("Failed to serialise Timescale result for cache", exc_info=True)
        return result

    # Public API mirrors ``TimescaleReader``
    def fetch_daily_bars(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: object | None = None,
        end: object | None = None,
        limit: int | None = None,
    ) -> TimescaleQueryResult:
        fetcher: FetchFn = getattr(self.reader, "fetch_daily_bars")
        return self._fetch(
            dimension="daily_bars",
            fetcher=fetcher,
            subjects=symbols,
            start=start,
            end=end,
            limit=limit,
            subject_param="symbols",
        )

    def fetch_intraday_trades(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: object | None = None,
        end: object | None = None,
        limit: int | None = None,
    ) -> TimescaleQueryResult:
        fetcher: FetchFn = getattr(self.reader, "fetch_intraday_trades")
        return self._fetch(
            dimension="intraday_trades",
            fetcher=fetcher,
            subjects=symbols,
            start=start,
            end=end,
            limit=limit,
            subject_param="symbols",
        )

    def fetch_macro_events(
        self,
        *,
        calendars: Sequence[str] | None = None,
        start: object | None = None,
        end: object | None = None,
        limit: int | None = None,
    ) -> TimescaleQueryResult:
        fetcher: FetchFn = getattr(self.reader, "fetch_macro_events")
        return self._fetch(
            dimension="macro_events",
            fetcher=fetcher,
            subjects=calendars,
            start=start,
            end=end,
            limit=limit,
            subject_param="calendars",
            extra={"calendars": "configured" if calendars else "ALL"},
        )


__all__ = ["TimescaleQueryCache"]
