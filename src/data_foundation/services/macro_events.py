"""Timescale-backed macro event helpers for sensory and risk modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Mapping, Sequence

import math
from functools import lru_cache

import pandas as pd

from ..cache.timescale_query_cache import TimescaleQueryCache
from ..persist.timescale_reader import TimescaleQueryResult, TimescaleReader

__all__ = [
    "MacroBiasResult",
    "MacroEventRecord",
    "TimescaleMacroEventService",
]


_IMPORTANCE_WEIGHTS: Mapping[str, float] = {
    "high": 1.0,
    "medium": 0.65,
    "low": 0.4,
}

_DEFAULT_LOOKBACK = timedelta(days=5)
_DEFAULT_LOOKAHEAD = timedelta(days=7)
_MAX_EVENTS = 250
_BIAS_CACHE_TTL = timedelta(minutes=5)

# Mapping from ISO currency code to macro calendars we should prioritise.
_CURRENCY_CALENDARS: Mapping[str, tuple[str, ...]] = {
    "USD": ("FOMC", "US", "FED"),
    "EUR": ("ECB", "EU"),
    "GBP": ("BOE", "UK"),
    "JPY": ("BOJ", "JP"),
    "AUD": ("RBA", "AU"),
    "CAD": ("BOC", "CA"),
    "NZD": ("RBNZ", "NZ"),
    "CHF": ("SNB", "CH"),
    "CNY": ("PBOC", "CN"),
}


@dataclass(frozen=True)
class MacroEventRecord:
    """Lightweight representation of a macro calendar event."""

    timestamp: datetime
    calendar: str
    event_name: str
    currency: str | None = None
    importance: str | None = None
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    source: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "calendar": self.calendar,
            "event_name": self.event_name,
            "currency": self.currency,
            "importance": self.importance,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
            "source": self.source,
        }

    @staticmethod
    def _coerce_timestamp(value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value.astimezone(UTC)
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is None:
                return value.to_pydatetime().replace(tzinfo=UTC)
            return value.to_pydatetime().astimezone(UTC)
        return datetime.now(tz=UTC)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "MacroEventRecord":
        timestamp = cls._coerce_timestamp(row.get("ts"))
        calendar = str(row.get("calendar", "")).strip() or "UNKNOWN"
        event = str(row.get("event_name", "")).strip() or "UNKNOWN"
        currency_raw = row.get("currency")
        currency = str(currency_raw).upper() if currency_raw else None
        importance_raw = row.get("importance")
        importance = str(importance_raw).lower() if importance_raw else None
        actual = _safe_float(row.get("actual"))
        forecast = _safe_float(row.get("forecast"))
        previous = _safe_float(row.get("previous"))
        source_raw = row.get("source")
        source = str(source_raw) if source_raw is not None else None
        return cls(
            timestamp=timestamp,
            calendar=calendar,
            event_name=event,
            currency=currency,
            importance=importance,
            actual=actual,
            forecast=forecast,
            previous=previous,
            source=source,
        )


def _safe_float(value: object) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MacroBiasResult:
    """Summarised macro bias derived from calendar events."""

    bias: float
    confidence: float
    events_analyzed: tuple[MacroEventRecord, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload = {
            "bias": self.bias,
            "confidence": self.confidence,
            "events_analyzed": [event.as_dict() for event in self.events_analyzed],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class TimescaleMacroEventService:
    """Fetch macro events from Timescale and derive simple macro bias metrics."""

    def __init__(
        self,
        reader: TimescaleReader,
        cache: TimescaleQueryCache | None = None,
        *,
        lookback: timedelta = _DEFAULT_LOOKBACK,
        lookahead: timedelta = _DEFAULT_LOOKAHEAD,
        max_events: int = _MAX_EVENTS,
        cache_ttl: timedelta = _BIAS_CACHE_TTL,
    ) -> None:
        self._reader = reader
        self._cache = cache
        self.lookback = lookback
        self.lookahead = lookahead
        self.max_events = max_events
        self._bias_cache: dict[str, tuple[datetime, MacroBiasResult]] = {}
        self._cache_ttl = cache_ttl

    def upcoming_events(
        self,
        *,
        calendars: Sequence[str] | None = None,
        currencies: Sequence[str] | None = None,
        as_of: datetime | None = None,
        lookahead: timedelta | None = None,
        limit: int | None = None,
    ) -> list[MacroEventRecord]:
        """Return upcoming events filtered by calendars or currencies."""

        now = (as_of or datetime.now(tz=UTC)).astimezone(UTC)
        end = now + (lookahead or self.lookahead)
        start = now
        calendar_keys = tuple(calendars) if calendars else None
        currency_keys = tuple(currencies) if currencies else None
        return self._fetch_events(
            calendars=_merge_calendars(calendar_keys, currency_keys),
            start=start,
            end=end,
            limit=limit or self.max_events,
        )

    def calculate_macro_bias(
        self,
        symbol: str,
        *,
        as_of: datetime | None = None,
    ) -> MacroBiasResult:
        """Compute a macro bias score for ``symbol`` using nearby macro events."""

        key = symbol.upper()
        anchor = (as_of or datetime.now(tz=UTC)).astimezone(UTC)
        cached = self._bias_cache.get(key)
        if cached is not None:
            cached_at, cached_result = cached
            if anchor - cached_at <= self._cache_ttl:
                return cached_result

        base_currency, quote_currency = _infer_currencies(symbol)
        currencies = [c for c in (base_currency, quote_currency) if c]
        if not currencies:
            currencies = ["USD"]

        window_start = anchor - self.lookback
        window_end = anchor + self.lookahead
        records = self._fetch_events(
            calendars=_merge_calendars(None, tuple(currencies)),
            start=window_start,
            end=window_end,
            limit=self.max_events,
        )
        relevant = [
            record
            for record in records
            if record.currency and record.currency.upper() in {c.upper() for c in currencies}
        ]

        if not relevant:
            result = MacroBiasResult(bias=0.0, confidence=0.1, events_analyzed=tuple())
            self._bias_cache[key] = (anchor, result)
            return result

        scores_by_currency: dict[str, list[float]] = {c: [] for c in currencies}
        weight_meta: dict[str, dict[str, float]] = {
            c: {"events": 0, "weight_sum": 0.0} for c in currencies
        }

        for record in relevant:
            currency = record.currency.upper() if record.currency else None
            if currency not in scores_by_currency:
                continue
            importance_weight = _IMPORTANCE_WEIGHTS.get(record.importance or "", 0.5)
            delta = _event_delta(record)
            decay = _time_decay(anchor, record.timestamp, self.lookback, self.lookahead)
            score = importance_weight * delta * decay
            scores_by_currency[currency].append(score)
            weight_meta[currency]["events"] += 1
            weight_meta[currency]["weight_sum"] += abs(importance_weight * decay)

        def _aggregate(currency: str | None) -> float:
            if not currency or currency not in scores_by_currency:
                return 0.0
            scores = scores_by_currency[currency]
            if not scores:
                return 0.0
            return _clamp(sum(scores) / max(len(scores), 1))

        base_score = _aggregate(base_currency)
        quote_score = _aggregate(quote_currency)
        bias = _clamp(base_score - quote_score)

        total_events = sum(int(meta["events"]) for meta in weight_meta.values())
        avg_weight_sum = sum(meta["weight_sum"] for meta in weight_meta.values()) or 1.0
        confidence = _clamp(
            0.2 + min(0.6, total_events * 0.1) + min(0.2, avg_weight_sum / (total_events or 1))
        )

        metadata = {
            "currencies": currencies,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "event_counts": {
                currency: int(meta["events"]) for currency, meta in weight_meta.items()
            },
        }

        # Limit the number of events stored for downstream consumers
        tracked_events = tuple(
            sorted(relevant, key=lambda rec: abs(_event_delta(rec)), reverse=True)[:10]
        )

        result = MacroBiasResult(
            bias=bias,
            confidence=confidence,
            events_analyzed=tracked_events,
            metadata=metadata,
        )
        self._bias_cache[key] = (anchor, result)
        return result

    def _fetch_events(
        self,
        *,
        calendars: Sequence[str] | None,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[MacroEventRecord]:
        fetcher = (
            self._cache.fetch_macro_events
            if self._cache is not None
            else self._reader.fetch_macro_events
        )
        result: TimescaleQueryResult = fetcher(
            calendars=list(calendars) if calendars else None,
            start=start,
            end=end,
            limit=limit,
        )
        frame = getattr(result, "frame", None)
        if frame is None or frame.empty:
            return []

        records: list[MacroEventRecord] = []
        for _, row in frame.iterrows():
            mapping = row.to_dict()
            records.append(MacroEventRecord.from_mapping(mapping))
        return records


def _event_delta(record: MacroEventRecord) -> float:
    if record.actual is not None and record.forecast not in (None, 0.0):
        baseline = record.forecast if record.forecast not in (None, 0.0) else 1.0
        return _clamp((record.actual - record.forecast) / baseline)
    if record.actual is not None and record.previous not in (None, 0.0):
        baseline = record.previous if record.previous not in (None, 0.0) else 1.0
        return _clamp((record.actual - record.previous) / baseline)
    return 0.0


def _time_decay(anchor: datetime, ts: datetime, lookback: timedelta, lookahead: timedelta) -> float:
    horizon = max(lookback.total_seconds(), lookahead.total_seconds(), 1.0)
    diff_seconds = abs((ts - anchor).total_seconds())
    decay = math.exp(-diff_seconds / (horizon / 2.0))
    return max(0.3, min(1.0, decay))


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@lru_cache(maxsize=128)
def _merge_calendars(
    calendars: tuple[str, ...] | None,
    currencies: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    merged: set[str] = set()
    if calendars:
        merged.update(str(cal).upper() for cal in calendars if cal)
    if currencies:
        for currency in currencies:
            if not currency:
                continue
            codes = _CURRENCY_CALENDARS.get(currency.upper())
            if codes:
                merged.update(code.upper() for code in codes)
    return tuple(sorted(merged)) if merged else None


def _infer_currencies(symbol: str) -> tuple[str | None, str | None]:
    if not symbol:
        return None, None
    token = "".join(ch for ch in symbol.upper() if ch.isalpha())
    if len(token) >= 6:
        base = token[:3]
        quote = token[3:6]
        return base, quote
    if len(token) == 3:
        return token, None
    return token[:3] or None, None
