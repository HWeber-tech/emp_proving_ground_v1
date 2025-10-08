from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Deque, Iterable, Mapping, MutableMapping

from pydantic import BaseModel, Field, validator

from src.operations.observability_diary import ThrottleStateSnapshot


@dataclass(frozen=True)
class TradeThrottleDecision:
    """Outcome of a trade throttle evaluation."""

    allowed: bool
    reason: str | None = None
    snapshot: ThrottleStateSnapshot | None = None
    retry_after_seconds: float | None = None

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "allowed": self.allowed,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.retry_after_seconds is not None:
            payload["retry_after_seconds"] = self.retry_after_seconds
        if self.snapshot is not None:
            payload["snapshot"] = self.snapshot.as_dict()
        return payload


class TradeThrottleConfig(BaseModel):
    """Configuration payload controlling trade throttling behaviour."""

    name: str = Field(default="trade_throttle", description="Human readable throttle name")
    max_trades: int = Field(
        default=1,
        ge=1,
        description="Maximum trades permitted inside the throttling window.",
    )
    window_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Duration of the sliding evaluation window in seconds.",
    )
    min_interval_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional hard minimum number of seconds between trades.",
    )

    @validator("window_seconds", "min_interval_seconds", pre=True)
    def _coerce_float(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        raise TypeError("expected numeric value for throttle timing configuration")

    class Config:
        validate_assignment = True


class TradeThrottle:
    """Simple sliding-window throttle to prevent rapid consecutive trades."""

    def __init__(
        self,
        config: TradeThrottleConfig | Mapping[str, object] | None = None,
        *,
        time_provider: Callable[[], datetime] | None = None,
    ) -> None:
        if config is None:
            config = TradeThrottleConfig()
        elif not isinstance(config, TradeThrottleConfig):
            config = TradeThrottleConfig.parse_obj(dict(config))
        self._config = config
        self._timestamps: Deque[datetime] = deque()
        self._time_provider = time_provider or (lambda: datetime.now(tz=timezone.utc))
        self._last_snapshot: ThrottleStateSnapshot | None = None

    @property
    def config(self) -> TradeThrottleConfig:
        """Expose the validated throttle configuration."""

        return self._config

    def _now(self) -> datetime:
        candidate = self._time_provider()
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)

    def _prune(self, window_start: datetime) -> None:
        while self._timestamps and self._timestamps[0] <= window_start:
            self._timestamps.popleft()

    def evaluate(self, *, now: datetime | None = None) -> TradeThrottleDecision:
        """Evaluate the throttle and optionally record an execution."""

        moment = (now or self._now()).astimezone(timezone.utc)
        window = timedelta(seconds=self._config.window_seconds)
        self._prune(moment - window)

        metadata: MutableMapping[str, object] = {
            "window_seconds": self._config.window_seconds,
            "max_trades": self._config.max_trades,
            "trades_in_window": len(self._timestamps),
        }

        min_interval = self._config.min_interval_seconds
        last_trade_at = self._timestamps[-1] if self._timestamps else None

        reason: str | None = None
        retry_after: float | None = None

        if len(self._timestamps) >= self._config.max_trades:
            reason = (
                "too many trades inside throttle window"
            )
            if self._timestamps:
                retry_after = max(
                    0.0,
                    (self._timestamps[0] + window - moment).total_seconds(),
                )
            metadata["trades_in_window"] = len(self._timestamps)
        elif min_interval is not None and last_trade_at is not None:
            elapsed = (moment - last_trade_at).total_seconds()
            metadata["elapsed_since_last"] = elapsed
            metadata["min_interval_seconds"] = min_interval
            if elapsed < min_interval:
                reason = "minimum trade interval not yet satisfied"
                retry_after = max(0.0, min_interval - elapsed)

        if reason is not None:
            snapshot = ThrottleStateSnapshot(
                name=self._config.name,
                state="throttled",
                active=True,
                multiplier=0.0,
                reason=reason,
                metadata=metadata,
            )
            decision = TradeThrottleDecision(
                allowed=False,
                reason=reason,
                snapshot=snapshot,
                retry_after_seconds=retry_after,
            )
            self._last_snapshot = snapshot
            return decision

        self._timestamps.append(moment)
        snapshot = ThrottleStateSnapshot(
            name=self._config.name,
            state="open",
            active=False,
            multiplier=1.0,
            reason=None,
            metadata=metadata,
        )
        decision = TradeThrottleDecision(
            allowed=True,
            reason=None,
            snapshot=snapshot,
            retry_after_seconds=None,
        )
        self._last_snapshot = snapshot
        return decision

    def reset(self) -> None:
        """Clear recorded executions and reset state."""

        self._timestamps.clear()
        self._last_snapshot = None

    def iter_timestamps(self) -> Iterable[datetime]:
        """Expose a snapshot of recorded execution timestamps."""

        return tuple(self._timestamps)

    def last_snapshot(self) -> ThrottleStateSnapshot | None:
        """Return the latest throttle snapshot, if any."""

        return self._last_snapshot


__all__ = [
    "TradeThrottle",
    "TradeThrottleConfig",
    "TradeThrottleDecision",
]
