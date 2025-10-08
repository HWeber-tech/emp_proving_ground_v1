"""Trade throttle utilities for limiting execution frequency."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Mapping

from src.operations.observability_diary import ThrottleStateSnapshot

__all__ = ["TradeThrottle", "TradeThrottleDecision"]


@dataclass(frozen=True)
class TradeThrottleDecision:
    """Decision payload returned when evaluating the trade throttle."""

    allowed: bool
    reason: str | None
    wait_seconds: float | None
    snapshot: ThrottleStateSnapshot


class TradeThrottle:
    """Simple fixed-window throttle for trade execution attempts."""

    def __init__(
        self,
        *,
        name: str = "trade_throttle",
        max_trades_per_window: int = 4,
        window_seconds: float = 60.0,
        min_interval_seconds: float | None = None,
    ) -> None:
        if max_trades_per_window <= 0:
            raise ValueError("max_trades_per_window must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if min_interval_seconds is not None and min_interval_seconds <= 0:
            raise ValueError("min_interval_seconds must be positive when provided")

        self._name = name
        self._max_trades = int(max_trades_per_window)
        self._window_seconds = float(window_seconds)
        self._min_interval = float(min_interval_seconds) if min_interval_seconds is not None else None
        self._timestamps: Deque[datetime] = deque()
        self._last_trade_at: datetime | None = None
        self._last_snapshot = self._make_snapshot(
            state="open",
            active=False,
            multiplier=1.0,
            reason=None,
            count=0,
            wait_seconds=None,
        )

    @property
    def max_trades_per_window(self) -> int:
        return self._max_trades

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    @property
    def min_interval_seconds(self) -> float | None:
        return self._min_interval

    @property
    def last_snapshot(self) -> ThrottleStateSnapshot:
        return self._last_snapshot

    def evaluate(self, when: datetime | None = None) -> TradeThrottleDecision:
        """Evaluate whether a trade is allowed at ``when`` (defaults to ``now``)."""

        now = self._normalise_timestamp(when)
        self._prune(now)

        wait_seconds: float | None = None

        if self._min_interval is not None and self._last_trade_at is not None:
            delta = (now - self._last_trade_at).total_seconds()
            if delta < self._min_interval:
                wait_seconds = max(self._min_interval - delta, 0.0)
                snapshot = self._make_snapshot(
                    state="cooldown",
                    active=True,
                    multiplier=0.0,
                    reason="minimum_interval",
                    count=len(self._timestamps),
                    wait_seconds=wait_seconds,
                )
                self._last_snapshot = snapshot
                return TradeThrottleDecision(
                    allowed=False,
                    reason="minimum_interval",
                    wait_seconds=wait_seconds,
                    snapshot=snapshot,
                )

        if len(self._timestamps) >= self._max_trades:
            oldest = self._timestamps[0]
            elapsed = (now - oldest).total_seconds()
            wait_seconds = max(self._window_seconds - elapsed, 0.0)
            snapshot = self._make_snapshot(
                state="rate_limited",
                active=True,
                multiplier=0.0,
                reason="rate_limit",
                count=len(self._timestamps),
                wait_seconds=wait_seconds,
            )
            self._last_snapshot = snapshot
            return TradeThrottleDecision(
                allowed=False,
                reason="rate_limit",
                wait_seconds=wait_seconds,
                snapshot=snapshot,
            )

        snapshot = self._make_snapshot(
            state="open",
            active=False,
            multiplier=1.0,
            reason=None,
            count=len(self._timestamps),
            wait_seconds=None,
        )
        self._last_snapshot = snapshot
        return TradeThrottleDecision(
            allowed=True,
            reason=None,
            wait_seconds=None,
            snapshot=snapshot,
        )

    def register_trade(self, when: datetime | None = None) -> ThrottleStateSnapshot:
        """Record a trade execution attempt at ``when`` (defaults to ``now``)."""

        now = self._normalise_timestamp(when)
        self._timestamps.append(now)
        self._last_trade_at = now
        self._prune(now)

        snapshot = self._make_snapshot(
            state="open",
            active=False,
            multiplier=1.0,
            reason=None,
            count=len(self._timestamps),
            wait_seconds=None,
        )
        self._last_snapshot = snapshot
        return snapshot

    def describe(self) -> Mapping[str, object]:
        """Return a serialisable description of the throttle configuration/state."""

        description: dict[str, object] = {
            "name": self._name,
            "max_trades_per_window": self._max_trades,
            "window_seconds": self._window_seconds,
            "trades_in_window": len(self._timestamps),
        }
        if self._min_interval is not None:
            description["min_interval_seconds"] = self._min_interval
        if self._last_trade_at is not None:
            description["last_trade_at"] = self._last_trade_at.astimezone(timezone.utc).isoformat()
        return description

    def reset(self) -> None:
        """Clear recorded trades and reset the throttle state."""

        self._timestamps.clear()
        self._last_trade_at = None
        self._last_snapshot = self._make_snapshot(
            state="open",
            active=False,
            multiplier=1.0,
            reason=None,
            count=0,
            wait_seconds=None,
        )

    def _prune(self, now: datetime) -> None:
        window = self._window_seconds
        while self._timestamps and (now - self._timestamps[0]).total_seconds() > window:
            self._timestamps.popleft()

    def _normalise_timestamp(self, when: datetime | None) -> datetime:
        if when is None:
            return datetime.now(tz=timezone.utc)
        if when.tzinfo is None:
            return when.replace(tzinfo=timezone.utc)
        return when.astimezone(timezone.utc)

    def _make_snapshot(
        self,
        *,
        state: str,
        active: bool,
        multiplier: float | None,
        reason: str | None,
        count: int,
        wait_seconds: float | None,
    ) -> ThrottleStateSnapshot:
        metadata = self._build_metadata(count=count, wait_seconds=wait_seconds)
        return ThrottleStateSnapshot(
            name=self._name,
            state=state,
            active=active,
            multiplier=multiplier,
            reason=reason,
            metadata=metadata,
        )

    def _build_metadata(self, *, count: int, wait_seconds: float | None) -> dict[str, object]:
        metadata: dict[str, object] = {
            "window_seconds": float(self._window_seconds),
            "max_trades_per_window": int(self._max_trades),
            "trades_in_window": int(count),
        }
        if self._min_interval is not None:
            metadata["min_interval_seconds"] = float(self._min_interval)
        if wait_seconds is not None:
            metadata["cooldown_seconds"] = max(0.0, float(wait_seconds))
        return metadata
