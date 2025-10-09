"""Utilities for enforcing trade frequency throttles in execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from types import MappingProxyType
from typing import Any, Deque, Mapping, MutableMapping

from pydantic import BaseModel, Field, validator

UTC = timezone.utc


class TradeThrottleConfig(BaseModel):
    """Configuration describing a trade throttle window."""

    name: str = Field(
        default="trade_rate_limit",
        description="Identifier for the throttle control",
    )
    max_trades: int = Field(
        default=1,
        ge=1,
        description="Maximum number of trades permitted within the window",
    )
    window_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Rolling window in seconds to evaluate trade counts",
    )
    cooldown_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional cooldown period after hitting the limit",
    )
    multiplier: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional position-size multiplier applied when active",
    )

    @validator("name")
    def _normalise_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("Throttle name must be a non-empty string")
        return name


@dataclass(frozen=True)
class TradeThrottleDecision:
    """Outcome of evaluating a trade throttle for a prospective order."""

    allowed: bool
    snapshot: Mapping[str, Any]
    reason: str | None = None
    retry_at: datetime | None = None

    def as_dict(self) -> Mapping[str, Any]:
        """Return a serialisable view of the throttle snapshot."""

        payload = dict(self.snapshot)
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            payload["metadata"] = dict(metadata)
        return payload


class TradeThrottle:
    """Maintains rolling trade counters to limit execution frequency."""

    _RATE_LIMIT_REASON = re.compile(r"^max_(?P<count>\d+)_trades_per_(?P<window>\d+)s$")

    def __init__(self, config: TradeThrottleConfig) -> None:
        self._config = config
        self._timestamps: Deque[datetime] = deque()
        self._cooldown_until: datetime | None = None
        self._last_snapshot: Mapping[str, Any] = self._initial_snapshot()

    @property
    def config(self) -> TradeThrottleConfig:
        """Expose the currently configured throttle parameters."""

        return self._config

    def evaluate(
        self,
        *,
        now: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TradeThrottleDecision:
        """Evaluate whether a trade is allowed at ``now``.

        Args:
            now: Timestamp for the evaluation. Defaults to ``datetime.now(UTC)``.
            metadata: Optional contextual details (symbol, strategy, etc.).
        """

        moment = self._coerce_timestamp(now)
        window_duration = timedelta(seconds=float(self._config.window_seconds))
        cooldown_duration = timedelta(seconds=float(self._config.cooldown_seconds))

        self._prune(moment, window_duration)

        state = "open"
        reason: str | None = None
        active = False
        retry_at: datetime | None = None

        if self._cooldown_until is not None and moment < self._cooldown_until:
            allowed = False
            active = True
            state = "cooldown"
            retry_at = self._cooldown_until
            reason = "cooldown_active"
        elif len(self._timestamps) >= self._config.max_trades:
            allowed = False
            active = True
            state = "rate_limited"
            reason = (
                f"max_{self._config.max_trades}_trades_per_{int(window_duration.total_seconds())}s"
            )
            if cooldown_duration.total_seconds() > 0:
                self._cooldown_until = moment + cooldown_duration
                retry_at = self._cooldown_until
            else:
                oldest = self._timestamps[0] if self._timestamps else moment
                retry_at = oldest + window_duration
        else:
            allowed = True
            self._timestamps.append(moment)

        message = self._format_reason(reason, retry_at)

        snapshot = self._build_snapshot(
            state=state,
            active=active,
            reason=reason,
            retry_at=retry_at,
            metadata=metadata,
            message=message,
        )
        self._last_snapshot = snapshot

        return TradeThrottleDecision(
            allowed=allowed,
            snapshot=snapshot,
            reason=reason,
            retry_at=retry_at,
        )

    def snapshot(self) -> Mapping[str, Any]:
        """Return the most recent throttle snapshot."""

        return dict(self._last_snapshot)

    def _initial_snapshot(self) -> Mapping[str, Any]:
        return self._build_snapshot(
            state="open",
            active=False,
            reason=None,
            retry_at=None,
            metadata=None,
            message=None,
        )

    def _build_snapshot(
        self,
        *,
        state: str,
        active: bool,
        reason: str | None,
        retry_at: datetime | None,
        metadata: Mapping[str, Any] | None,
        message: str | None,
    ) -> Mapping[str, Any]:
        meta_payload: MutableMapping[str, Any] = {
            "max_trades": self._config.max_trades,
            "window_seconds": float(self._config.window_seconds),
            "recent_trades": len(self._timestamps),
        }
        cooldown_seconds = float(self._config.cooldown_seconds)
        if cooldown_seconds:
            meta_payload["cooldown_seconds"] = cooldown_seconds
        if self._cooldown_until is not None:
            meta_payload["cooldown_until"] = self._cooldown_until.astimezone(UTC).isoformat()
        if metadata:
            meta_payload["context"] = dict(metadata)
        snapshot: dict[str, Any] = {
            "name": self._config.name,
            "state": state,
            "active": active,
            "metadata": meta_payload,
        }
        if self._config.multiplier is not None:
            snapshot["multiplier"] = float(self._config.multiplier)
        if reason:
            snapshot["reason"] = reason
        if message:
            snapshot["message"] = message
        if retry_at is not None:
            snapshot.setdefault("metadata", meta_payload)
            snapshot["metadata"]["retry_at"] = retry_at.astimezone(UTC).isoformat()
        return MappingProxyType(snapshot)

    def _prune(self, moment: datetime, window: timedelta) -> None:
        while self._timestamps and moment - self._timestamps[0] >= window:
            self._timestamps.popleft()
        if self._cooldown_until is not None and moment >= self._cooldown_until:
            self._cooldown_until = None

    @staticmethod
    def _coerce_timestamp(candidate: datetime | None) -> datetime:
        if candidate is None:
            return datetime.now(tz=UTC)
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=UTC)
        return candidate.astimezone(UTC)

    def _format_reason(self, reason: str | None, retry_at: datetime | None) -> str | None:
        if not reason:
            return None

        if reason == "cooldown_active":
            if retry_at is not None:
                iso_retry = retry_at.astimezone(UTC).isoformat()
                return f"Throttle cooldown active until {iso_retry}"
            return "Throttle cooldown active"

        match = self._RATE_LIMIT_REASON.match(reason)
        if match:
            count = int(match.group("count"))
            window_seconds = int(match.group("window"))
            trades_label = "trade" if count == 1 else "trades"
            window_label = self._format_window_description(window_seconds)
            return (
                "Throttled: too many trades in short time "
                f"(limit {count} {trades_label} per {window_label})"
            )

        return f"Throttle reason: {reason}"

    @staticmethod
    def _format_window_description(window_seconds: int) -> str:
        if window_seconds % 3600 == 0:
            hours = window_seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''}"
        if window_seconds % 60 == 0:
            minutes = window_seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        return f"{window_seconds} second{'s' if window_seconds != 1 else ''}"
