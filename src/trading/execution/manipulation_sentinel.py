"""Order flow sentinel that flags spoof-like manipulation patterns."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Mapping, MutableMapping

__all__ = ["ManipulationSentinel"]


def _ensure_utc(timestamp: datetime | None, fallback: datetime) -> datetime:
    if timestamp is None:
        return fallback
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _normalise_side(side: str | None) -> str:
    if not side:
        return "UNKNOWN"
    return str(side).strip().upper() or "UNKNOWN"


def _upper_symbol(symbol: str | None) -> str:
    if not symbol:
        return "UNKNOWN"
    return str(symbol).strip().upper() or "UNKNOWN"


@dataclass(slots=True)
class _OrderRecord:
    event_id: str
    symbol: str
    side: str
    notional: float
    timestamp: datetime
    executed: bool | None = None
    outcome: str | None = None
    fill_notional: float | None = None
    strategy_id: str | None = None
    metadata: Mapping[str, Any] | None = None


class ManipulationSentinel:
    """Track order flow and surface spoof-like manipulation patterns."""

    def __init__(
        self,
        *,
        window_seconds: float = 120.0,
        min_large_notional: float = 75_000.0,
        min_pattern_orders: int = 3,
        small_execution_ratio: float = 0.35,
        cooldown_seconds: float = 60.0,
        max_records: int = 512,
        min_small_notional: float = 1_000.0,
    ) -> None:
        self._now = lambda: datetime.now(tz=timezone.utc)
        self._window = timedelta(seconds=max(window_seconds, 1.0))
        self._min_large_notional = float(max(min_large_notional, 0.0))
        self._min_pattern_orders = max(1, int(min_pattern_orders))
        self._small_execution_ratio = max(0.05, float(small_execution_ratio))
        self._cooldown = timedelta(seconds=max(0.0, float(cooldown_seconds)))
        self._max_records = max(32, int(max_records))
        self._min_small_notional = max(0.0, float(min_small_notional))

        self._records: Deque[_OrderRecord] = deque()
        self._index: dict[str, _OrderRecord] = {}
        self._cooldowns: dict[str, dict[str, Any]] = {}
        self._pattern_markers: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def should_block(
        self,
        *,
        symbol: str,
        side: str,
        notional: float,
        timestamp: datetime | None = None,
        event_id: str | None = None,
        strategy_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        """Return metadata describing a spoof-like pattern if present."""

        now = _ensure_utc(timestamp, self._now())
        symbol_key = _upper_symbol(symbol)
        side_key = _normalise_side(side)
        self._purge(now)
        cooldown_payload = self._cooldown_block(symbol_key, now)
        if cooldown_payload:
            return cooldown_payload

        pattern = self._detect_spoof(symbol_key, now)
        if pattern is None:
            return None

        payload: dict[str, Any] = {
            "reason": "manipulation.spoof_pattern",
            "pattern": "spoof",
            "symbol": symbol_key,
            "side_under_review": pattern["dominant_side"],
            "spoof_order_count": pattern["count"],
            "spoof_notional_total": pattern["total_notional"],
            "spoof_notional_avg": pattern["avg_notional"],
            "spoof_notional_max": pattern["max_notional"],
            "executed_order_side": pattern["executed_side"],
            "executed_order_notional": pattern["executed_notional"],
            "pattern_span_seconds": pattern["span_seconds"],
            "window_seconds": self._window.total_seconds(),
            "pattern_first_order_at": pattern["first_large_at"].isoformat(),
            "pattern_last_order_at": pattern["last_large_at"].isoformat(),
            "executed_at": pattern["executed_at"].isoformat(),
            "cooldown_seconds": self._cooldown.total_seconds(),
            "pattern_marker_ts": pattern["last_large_at"].timestamp(),
            "pattern_start_ts": pattern["first_large_at"].timestamp(),
        }
        if event_id:
            payload["event_id"] = event_id
        if strategy_id:
            payload["strategy_id"] = strategy_id
        payload["requested_side"] = side_key
        payload["requested_notional"] = abs(float(notional))
        return payload

    def observe_submission(
        self,
        *,
        event_id: str | None,
        symbol: str,
        side: str,
        notional: float,
        timestamp: datetime | None = None,
        strategy_id: str | None = None,
    ) -> None:
        """Record a new order submission that was allowed to proceed."""

        now = _ensure_utc(timestamp, self._now())
        symbol_key = _upper_symbol(symbol)
        side_key = _normalise_side(side)
        record = _OrderRecord(
            event_id=event_id or f"submission-{now.timestamp():.6f}",
            symbol=symbol_key,
            side=side_key,
            notional=max(0.0, abs(float(notional))),
            timestamp=now,
            executed=None,
            strategy_id=strategy_id,
        )
        self._append_record(record)
        if event_id:
            self._index[event_id] = record
        self._purge(now)

    def mark_outcome(
        self,
        event_id: str,
        *,
        executed: bool,
        timestamp: datetime | None = None,
        fill_notional: float | None = None,
        reason: str | None = None,
    ) -> None:
        """Update an observed submission with its final outcome."""

        record = self._index.get(event_id)
        if record is None:
            return
        record.executed = bool(executed)
        final_ts = _ensure_utc(timestamp, record.timestamp)
        record.timestamp = final_ts
        if fill_notional is not None:
            record.fill_notional = abs(float(fill_notional))
        record.outcome = reason or ("filled" if executed else "not_filled")

    def record_block(
        self,
        *,
        event_id: str | None,
        symbol: str,
        side: str,
        notional: float,
        timestamp: datetime | None = None,
        reason: str = "manipulation.blocked",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a blocked attempt and start cooldown if applicable."""

        now = _ensure_utc(timestamp, self._now())
        symbol_key = _upper_symbol(symbol)
        side_key = _normalise_side(side)
        record = _OrderRecord(
            event_id=event_id or f"blocked-{now.timestamp():.6f}",
            symbol=symbol_key,
            side=side_key,
            notional=max(0.0, abs(float(notional))),
            timestamp=now,
            executed=False,
            outcome=reason,
            metadata=dict(metadata or {}),
        )
        self._append_record(record)
        if event_id:
            self._index[event_id] = record
        self._purge(now)

        if reason == "manipulation.spoof_pattern":
            marker_ts: float | None = None
            if isinstance(metadata, Mapping):
                marker_candidate = metadata.get("pattern_marker_ts")
                if isinstance(marker_candidate, (int, float)):
                    marker_ts = float(marker_candidate)
            self._pattern_markers[symbol_key] = marker_ts or now.timestamp()
            if self._cooldown.total_seconds() > 0.0:
                until = now + self._cooldown
                payload: dict[str, Any] = {
                    "until": until,
                    "pattern": (metadata or {}).get("pattern", "spoof")
                    if isinstance(metadata, Mapping)
                    else "spoof",
                }
                if metadata:
                    payload["metadata"] = dict(metadata)
                self._cooldowns[symbol_key] = payload

    def snapshot(self, symbol: str | None = None) -> Mapping[str, Any]:
        """Return a serialisable snapshot of sentinel state for diagnostics."""

        symbol_key = _upper_symbol(symbol) if symbol else None
        entries: list[dict[str, Any]] = []
        for record in list(self._records):
            if symbol_key and record.symbol != symbol_key:
                continue
            payload: dict[str, Any] = {
                "event_id": record.event_id,
                "symbol": record.symbol,
                "side": record.side,
                "notional": record.notional,
                "timestamp": record.timestamp.isoformat(),
                "executed": record.executed,
                "outcome": record.outcome,
            }
            if record.fill_notional is not None:
                payload["fill_notional"] = record.fill_notional
            if record.strategy_id:
                payload["strategy_id"] = record.strategy_id
            if record.metadata:
                payload["metadata"] = dict(record.metadata)
            entries.append(payload)
        cooldowns: dict[str, Any] = {}
        for key, payload in self._cooldowns.items():
            until = payload.get("until")
            if isinstance(until, datetime):
                cooldowns[key] = {
                    "until": until.isoformat(),
                    "pattern": payload.get("pattern"),
                }
        return {
            "records": entries,
            "cooldowns": cooldowns,
            "pattern_markers": dict(self._pattern_markers),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_record(self, record: _OrderRecord) -> None:
        self._records.append(record)
        if len(self._records) > self._max_records:
            old = self._records.popleft()
            self._index.pop(old.event_id, None)

    def _purge(self, now: datetime) -> None:
        cutoff = now - self._window
        while self._records and self._records[0].timestamp < cutoff:
            old = self._records.popleft()
            self._index.pop(old.event_id, None)
        for symbol, payload in list(self._cooldowns.items()):
            until = payload.get("until")
            if not isinstance(until, datetime) or now >= until:
                self._cooldowns.pop(symbol, None)

    def _cooldown_block(self, symbol: str, now: datetime) -> Mapping[str, Any] | None:
        payload = self._cooldowns.get(symbol)
        if not payload:
            return None
        until = payload.get("until")
        if not isinstance(until, datetime):
            return None
        if now >= until:
            self._cooldowns.pop(symbol, None)
            return None
        remaining = max(0.0, (until - now).total_seconds())
        reason_payload: dict[str, Any] = {
            "reason": "manipulation.cooldown",
            "pattern": payload.get("pattern", "spoof"),
            "cooldown_remaining_seconds": remaining,
            "cooldown_ends_at": until.isoformat(),
        }
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            reason_payload["trigger"] = dict(metadata)
        return reason_payload

    def _detect_spoof(
        self,
        symbol: str,
        now: datetime,
    ) -> Mapping[str, Any] | None:
        if self._min_large_notional <= 0.0:
            return None

        window_records = [
            record
            for record in self._records
            if record.symbol == symbol and now - record.timestamp <= self._window
        ]
        if len(window_records) < self._min_pattern_orders + 1:
            return None

        cancelled_large = [
            record
            for record in window_records
            if record.executed is False and record.notional >= self._min_large_notional
        ]
        if len(cancelled_large) < self._min_pattern_orders:
            return None

        dominant_side = self._dominant_side(cancelled_large)
        dominant_large = [
            record for record in cancelled_large if record.side == dominant_side
        ]
        if len(dominant_large) < self._min_pattern_orders:
            return None

        marker_ts = self._pattern_markers.get(symbol)
        last_large_at = max(record.timestamp for record in dominant_large)
        if marker_ts is not None and last_large_at.timestamp() <= marker_ts:
            return None

        first_large_at = min(record.timestamp for record in dominant_large)
        total_notional = sum(record.notional for record in dominant_large)
        avg_notional = total_notional / len(dominant_large)
        max_notional = max(record.notional for record in dominant_large)
        small_threshold = max(
            self._min_small_notional,
            avg_notional * self._small_execution_ratio,
        )

        executed_candidates = [
            record
            for record in window_records
            if record.executed is True
            and record.side != dominant_side
            and record.notional <= small_threshold + 1e-9
            and record.timestamp >= last_large_at
        ]
        if not executed_candidates:
            return None

        executed = min(executed_candidates, key=lambda item: item.timestamp)
        span_seconds = max(
            0.0, (executed.timestamp - first_large_at).total_seconds()
        )

        return {
            "dominant_side": dominant_side,
            "count": len(dominant_large),
            "total_notional": total_notional,
            "avg_notional": avg_notional,
            "max_notional": max_notional,
            "executed_side": executed.side,
            "executed_notional": executed.notional,
            "span_seconds": span_seconds,
            "first_large_at": first_large_at,
            "last_large_at": last_large_at,
            "executed_at": executed.timestamp,
        }

    @staticmethod
    def _dominant_side(records: list[_OrderRecord]) -> str:
        counts: MutableMapping[str, int] = {}
        for record in records:
            counts[record.side] = counts.get(record.side, 0) + 1
        dominant_side = "UNKNOWN"
        dominant_count = -1
        for side, count in counts.items():
            if count > dominant_count:
                dominant_side = side
                dominant_count = count
        return dominant_side

