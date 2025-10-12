"""Utilities for enforcing trade frequency throttles in execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
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
    min_spacing_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum elapsed time required between consecutive trades before "
            "the throttle permits a new order"
        ),
    )
    multiplier: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional position-size multiplier applied when active",
    )
    scope_fields: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Optional metadata fields used to scope throttle counters. When provided, "
            "the throttle maintains independent rolling windows per unique "
            "combination of these fields."
        ),
    )

    @validator("name")
    def _normalise_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("Throttle name must be a non-empty string")
        return name

    @validator("scope_fields", pre=True)
    def _coerce_scope_fields(
        cls, value: Any  # type: ignore[override]
    ) -> tuple[str, ...]:
        if value is None or value == ():
            return ()
        if isinstance(value, str):
            candidates = [value]
        else:
            try:
                candidates = list(value)
            except TypeError as exc:  # pragma: no cover - defensive guard
                raise ValueError("scope_fields must be an iterable of strings") from exc

        normalised: list[str] = []
        seen: set[str] = set()
        for entry in candidates:
            field = str(entry).strip()
            if not field:
                raise ValueError("scope_fields entries must be non-empty strings")
            field_lower = field.lower()
            if field_lower in seen:
                continue
            seen.add(field_lower)
            normalised.append(field)
        return tuple(normalised)


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
    _GLOBAL_SCOPE: tuple[str, ...] = ("__global__",)

    @dataclass
    class _ThrottleState:
        timestamps: Deque[datetime]
        cooldown_until: datetime | None = None
        last_trade: datetime | None = None

    def __init__(self, config: TradeThrottleConfig) -> None:
        self._config = config
        self._states: dict[tuple[str, ...], TradeThrottle._ThrottleState] = {}
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

        scope_key, scope_descriptor = self._resolve_scope(metadata)
        state = self._get_state(scope_key)

        self._prune(scope_key, moment, window_duration, state)

        throttle_state = "open"
        reason: str | None = None
        active = False
        retry_at: datetime | None = None
        message: str | None = None

        if state.cooldown_until is not None and moment < state.cooldown_until:
            allowed = False
            active = True
            throttle_state = "cooldown"
            retry_at = state.cooldown_until
            reason = "cooldown_active"
        elif self._min_spacing_violation(state, moment):
            allowed = False
            active = True
            throttle_state = "min_interval"
            reason, retry_at = self._build_min_spacing_reason(state, moment)
            if retry_at is not None:
                message = self._format_min_spacing_message(retry_at, moment)
        elif len(state.timestamps) >= self._config.max_trades:
            allowed = False
            active = True
            throttle_state = "rate_limited"
            reason = (
                f"max_{self._config.max_trades}_trades_per_{int(window_duration.total_seconds())}s"
            )
            if cooldown_duration.total_seconds() > 0:
                state.cooldown_until = moment + cooldown_duration
                retry_at = state.cooldown_until
            else:
                oldest = state.timestamps[0] if state.timestamps else moment
                retry_at = oldest + window_duration
        else:
            allowed = True
            state.timestamps.append(moment)
            state.cooldown_until = None
            state.last_trade = moment

        if message is None:
            message = self._format_reason(reason, retry_at)

        snapshot = self._build_snapshot(
            state=throttle_state,
            active=active,
            reason=reason,
            retry_at=retry_at,
            metadata=metadata,
            message=message,
            scope_key=scope_key,
            scope_descriptor=scope_descriptor,
            recent_trades=len(state.timestamps),
            cooldown_until=state.cooldown_until,
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
            scope_key=self._GLOBAL_SCOPE,
            scope_descriptor=None,
            recent_trades=0,
            cooldown_until=None,
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
        scope_key: tuple[str, ...],
        scope_descriptor: Mapping[str, Any] | None,
        recent_trades: int,
        cooldown_until: datetime | None,
    ) -> Mapping[str, Any]:
        meta_payload: MutableMapping[str, Any] = {
            "max_trades": self._config.max_trades,
            "window_seconds": float(self._config.window_seconds),
            "recent_trades": recent_trades,
        }
        if self._config.min_spacing_seconds:
            meta_payload["min_spacing_seconds"] = float(self._config.min_spacing_seconds)
        cooldown_seconds = float(self._config.cooldown_seconds)
        if cooldown_seconds:
            meta_payload["cooldown_seconds"] = cooldown_seconds
        if scope_descriptor is not None and self._config.scope_fields:
            meta_payload["scope"] = {
                field: scope_descriptor.get(field)
                for field in self._config.scope_fields
            }
        elif self._config.scope_fields:
            meta_payload["scope"] = {
                field: None for field in self._config.scope_fields
            }
        if cooldown_until is not None:
            meta_payload["cooldown_until"] = cooldown_until.astimezone(UTC).isoformat()
        if metadata:
            meta_payload["context"] = dict(metadata)
        snapshot: dict[str, Any] = {
            "name": self._config.name,
            "state": state,
            "active": active,
            "metadata": meta_payload,
        }
        if self._config.scope_fields:
            snapshot["scope_key"] = list(scope_key)
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

    def _get_state(self, scope_key: tuple[str, ...]) -> "TradeThrottle._ThrottleState":
        state = self._states.get(scope_key)
        if state is None:
            state = TradeThrottle._ThrottleState(timestamps=deque())
            self._states[scope_key] = state
        return state

    def _min_spacing_violation(
        self, state: "TradeThrottle._ThrottleState", moment: datetime
    ) -> bool:
        min_spacing_seconds = float(self._config.min_spacing_seconds)
        if min_spacing_seconds <= 0:
            return False
        last_trade = state.last_trade
        if last_trade is None:
            return False
        min_spacing = timedelta(seconds=min_spacing_seconds)
        if moment - last_trade < min_spacing:
            return True
        return False

    def _build_min_spacing_reason(
        self, state: "TradeThrottle._ThrottleState", moment: datetime
    ) -> tuple[str, datetime | None]:
        last_trade = state.last_trade
        min_spacing_seconds = float(self._config.min_spacing_seconds)
        min_spacing = timedelta(seconds=min_spacing_seconds)
        retry_at = None
        if last_trade is not None:
            retry_at = last_trade + min_spacing
        reason = f"min_interval_{min_spacing_seconds:g}s"
        return reason, retry_at

    def _format_min_spacing_message(
        self, retry_at: datetime | None, moment: datetime
    ) -> str | None:
        if retry_at is None:
            return None
        remaining = retry_at - moment
        remaining_seconds = max(remaining.total_seconds(), 0.0)
        label = self._format_seconds_brief(float(self._config.min_spacing_seconds))
        remaining_label = self._format_seconds_brief(remaining_seconds)
        retry_iso = retry_at.astimezone(UTC).isoformat()
        return (
            "Throttled: minimum interval of "
            f"{label} between trades (retry in {remaining_label} at {retry_iso})"
        )

    def _resolve_scope(
        self, metadata: Mapping[str, Any] | None
    ) -> tuple[tuple[str, ...], Mapping[str, Any] | None]:
        if not self._config.scope_fields:
            return self._GLOBAL_SCOPE, None

        scope_context: dict[str, Any] = {}
        context_source: Mapping[str, Any]
        if isinstance(metadata, Mapping):
            context_source = metadata
            if not isinstance(metadata, dict):
                context_source = dict(metadata)
        else:
            context_source = {}
        key_parts: list[str] = []
        for field in self._config.scope_fields:
            value = context_source.get(field) if context_source else None
            scope_context[field] = value
            key_parts.append(self._normalise_scope_value(value))
        return tuple(key_parts), scope_context

    @staticmethod
    def _normalise_scope_value(value: Any) -> str:
        if isinstance(value, datetime):
            return f"datetime:{value.astimezone(UTC).isoformat()}"
        if isinstance(value, (str, int, float, bool)) or value is None:
            return f"{type(value).__name__}:{value!r}"
        return f"{type(value).__name__}:{repr(value)}"

    def _prune(
        self,
        scope_key: tuple[str, ...],
        moment: datetime,
        window: timedelta,
        state: "TradeThrottle._ThrottleState",
    ) -> None:
        timestamps = state.timestamps
        while timestamps and moment - timestamps[0] >= window:
            timestamps.popleft()
        if state.cooldown_until is not None and moment >= state.cooldown_until:
            state.cooldown_until = None

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

    @staticmethod
    def _format_seconds_brief(seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        if seconds >= 3600 and math.isclose(seconds % 3600, 0.0, abs_tol=1e-9):
            hours = seconds / 3600
            return f"{hours:g} hour{'s' if hours != 1 else ''}"
        if seconds >= 60 and math.isclose(seconds % 60, 0.0, abs_tol=1e-9):
            minutes = seconds / 60
            return f"{minutes:g} minute{'s' if minutes != 1 else ''}"
        if math.isclose(seconds, 1.0, abs_tol=1e-9):
            return "1 second"
        if seconds.is_integer():
            return f"{int(seconds)} seconds"
        return f"{seconds:.2f} seconds"
