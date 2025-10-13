"""Utilities for enforcing trade frequency throttles in execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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
    max_notional: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum aggregate notional permitted within the window",
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
    multiplier: float | None = None
    retry_in_seconds: float | None = None
    evaluated_at: datetime | None = None
    scope_key: tuple[str, ...] = field(default_factory=tuple)
    remaining_notional: float | None = None
    notional_utilisation: float | None = None
    applied_notional: float | None = None

    def as_dict(self) -> Mapping[str, Any]:
        """Return a serialisable view of the throttle snapshot."""

        payload = dict(self.snapshot)
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            payload["metadata"] = dict(metadata)
        return payload


class TradeThrottle:
    """Maintains rolling trade counters to limit execution frequency."""

    _RATE_LIMIT_REASON = re.compile(
        r"^max_(?P<count>\d+)_trades_per_(?P<window>\d+(?:\.\d+)?)s$"
    )
    _NOTIONAL_LIMIT_REASON = re.compile(
        r"^max_notional_(?P<notional>\d+(?:\.\d+)?)_per_(?P<window>\d+(?:\.\d+)?)s$"
    )
    _GLOBAL_SCOPE: tuple[str, ...] = ("__global__",)

    @dataclass
    class _ThrottleState:
        timestamps: Deque[datetime]
        cooldown_until: datetime | None = None
        last_trade: datetime | None = None
        scope_descriptor: Mapping[str, Any] | None = None
        notional_records: Deque[tuple[datetime, float]] = field(default_factory=deque)
        notional_total: float = 0.0

    @dataclass(frozen=True)
    class _ExternalCooldown:
        reason: str
        message: str | None = None
        metadata: Mapping[str, Any] | None = None

    def __init__(self, config: TradeThrottleConfig) -> None:
        self._config = config
        self._states: dict[tuple[str, ...], TradeThrottle._ThrottleState] = {}
        self._scope_snapshots: dict[tuple[str, ...], Mapping[str, Any]] = {}
        self._external_cooldowns: dict[
            tuple[str, ...], TradeThrottle._ExternalCooldown
        ] = {}
        initial_snapshot = self._initial_snapshot()
        self._last_snapshot: Mapping[str, Any] = initial_snapshot
        if not self._config.scope_fields:
            self._scope_snapshots[self._GLOBAL_SCOPE] = initial_snapshot

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
        window_seconds = float(self._config.window_seconds)
        window_duration = timedelta(seconds=window_seconds)
        cooldown_duration = timedelta(seconds=float(self._config.cooldown_seconds))

        scope_key, scope_descriptor = self._resolve_scope(metadata)
        self._purge_stale_scopes(moment, window_duration)
        state = self._get_state(scope_key)

        state_removed = self._prune(scope_key, moment, window_duration, state)
        if state_removed:
            state = self._get_state(scope_key)

        if scope_descriptor:
            state.scope_descriptor = MappingProxyType(dict(scope_descriptor))
        elif self._config.scope_fields:
            state.scope_descriptor = MappingProxyType(
                {field: None for field in self._config.scope_fields}
            )
        else:
            state.scope_descriptor = None

        throttle_state = "open"
        reason: str | None = None
        active = False
        retry_at: datetime | None = None
        message: str | None = None
        retry_in_seconds: float | None = None
        cooldown_payload = self._external_cooldowns.get(scope_key)
        applied_notional: float | None = None

        notional_limit = (
            float(self._config.max_notional)
            if self._config.max_notional is not None
            else None
        )
        trade_notional = self._resolve_notional(metadata)

        if state.cooldown_until is not None and moment < state.cooldown_until:
            allowed = False
            active = True
            throttle_state = "cooldown"
            retry_at = state.cooldown_until
            if cooldown_payload is not None:
                reason = cooldown_payload.reason
                message = cooldown_payload.message
            else:
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
            reason = f"max_{self._config.max_trades}_trades_per_{window_seconds:g}s"
            if cooldown_duration.total_seconds() > 0:
                state.cooldown_until = moment + cooldown_duration
                retry_at = state.cooldown_until
            else:
                oldest = state.timestamps[0] if state.timestamps else moment
                retry_at = oldest + window_duration
        elif notional_limit is not None:
            projected_notional = state.notional_total
            if trade_notional is not None:
                projected_notional += trade_notional
            if projected_notional > notional_limit + 1e-9:
                allowed = False
                active = True
                throttle_state = "notional_limit"
                reason = (
                    f"max_notional_{notional_limit:g}_per_{window_seconds:g}s"
                )
                if state.notional_records:
                    oldest_notional = state.notional_records[0][0]
                elif state.timestamps:
                    oldest_notional = state.timestamps[0]
                else:
                    oldest_notional = moment
                retry_at = oldest_notional + window_duration
            else:
                allowed = True
        else:
            allowed = True

        if allowed:
            state.timestamps.append(moment)
            state.cooldown_until = None
            state.last_trade = moment
            if trade_notional is not None:
                state.notional_records.append((moment, trade_notional))
                state.notional_total += trade_notional
                applied_notional = trade_notional
            else:
                applied_notional = None
        if message is None:
            message = self._format_reason(reason, retry_at)

        recent_trades = len(state.timestamps)

        window_reset_at: datetime | None = None
        window_reset_in_seconds: float | None = None
        if state.timestamps:
            oldest_trade = state.timestamps[0]
            reset_target = oldest_trade + window_duration
            window_reset_at = reset_target
            window_reset_in_seconds = max(
                (reset_target - moment).total_seconds(),
                0.0,
            )

        if retry_at is not None:
            retry_in_seconds = max((retry_at - moment).total_seconds(), 0.0)

        attempted_notional: float | None = None
        if not allowed and throttle_state == "notional_limit":
            attempted_notional = trade_notional

        snapshot = self._build_snapshot(
            state=throttle_state,
            active=active,
            reason=reason,
            retry_at=retry_at,
            metadata=metadata,
            message=message,
            scope_key=scope_key,
            scope_descriptor=state.scope_descriptor,
            recent_trades=recent_trades,
            cooldown_until=state.cooldown_until,
            moment=moment,
            window_reset_at=window_reset_at,
            window_reset_in_seconds=window_reset_in_seconds,
            retry_in_seconds=retry_in_seconds,
            cooldown_metadata=(
                cooldown_payload.metadata if cooldown_payload is not None else None
            ),
            notional_total=state.notional_total,
            attempted_notional=attempted_notional,
        )
        self._last_snapshot = snapshot
        self._scope_snapshots[scope_key] = snapshot

        multiplier: float | None
        if self._config.multiplier is None:
            multiplier = None
        else:
            multiplier = float(self._config.multiplier)
            if not math.isfinite(multiplier):
                multiplier = None

        remaining_notional: float | None = None
        notional_utilisation: float | None = None
        metadata_view = snapshot.get("metadata")
        if isinstance(metadata_view, Mapping):
            remaining_raw = metadata_view.get("remaining_notional")
            utilisation_raw = metadata_view.get("notional_utilisation")
            try:
                if isinstance(remaining_raw, (int, float)):
                    remaining_notional = float(remaining_raw)
            except (TypeError, ValueError):
                remaining_notional = None
            try:
                if isinstance(utilisation_raw, (int, float)):
                    notional_utilisation = float(utilisation_raw)
            except (TypeError, ValueError):
                notional_utilisation = None

        return TradeThrottleDecision(
            allowed=allowed,
            snapshot=snapshot,
            reason=reason,
            retry_at=retry_at,
            multiplier=multiplier,
            retry_in_seconds=retry_in_seconds,
            evaluated_at=moment,
            scope_key=scope_key,
            remaining_notional=remaining_notional,
            notional_utilisation=notional_utilisation,
            applied_notional=applied_notional,
        )

    def snapshot(self) -> Mapping[str, Any]:
        """Return the most recent throttle snapshot."""

        return dict(self._last_snapshot)

    def scope_snapshots(self) -> tuple[Mapping[str, Any], ...]:
        """Return snapshots for each tracked scope."""

        entries = sorted(self._scope_snapshots.items(), key=lambda item: item[0])
        return tuple(self._clone_snapshot(snapshot) for _key, snapshot in entries)

    def rollback(self, decision: TradeThrottleDecision) -> Mapping[str, Any] | None:
        """Roll back counters for a trade that was allowed but not executed.

        Args:
            decision: The previously returned decision corresponding to the trade
                that needs to be reverted.

        Returns:
            Updated snapshot for the affected scope or ``None`` if no changes were
            applied (for example, when the decision was already blocked).
        """

        if not isinstance(decision, TradeThrottleDecision):
            raise TypeError("rollback expects a TradeThrottleDecision instance")
        if not decision.allowed:
            return None

        scope_key = decision.scope_key or self._GLOBAL_SCOPE
        state = self._states.get(scope_key)
        if state is None:
            return None

        moment = decision.evaluated_at
        if moment is None:
            return None

        moment = self._coerce_timestamp(moment)
        timestamps = state.timestamps
        removed = False
        for index in range(len(timestamps) - 1, -1, -1):
            candidate = timestamps[index]
            if candidate == moment:
                del timestamps[index]
                removed = True
                break
            if abs((candidate - moment).total_seconds()) <= 1e-6:
                del timestamps[index]
                removed = True
                break

        if not removed:
            return None

        applied_notional = decision.applied_notional
        if applied_notional is not None:
            records = state.notional_records
            target_notional = float(applied_notional)
            for index in range(len(records) - 1, -1, -1):
                record_time, record_value = records[index]
                if abs((record_time - moment).total_seconds()) <= 1e-6 and math.isclose(
                    float(record_value), target_notional, rel_tol=1e-9, abs_tol=1e-6
                ):
                    del records[index]
                    state.notional_total -= float(record_value)
                    break
            state.notional_total = max(state.notional_total, 0.0)

        if state.last_trade == moment:
            state.last_trade = timestamps[-1] if timestamps else None

        window_duration = timedelta(seconds=float(self._config.window_seconds))
        self._cleanup_scope_if_idle(scope_key, state, moment, window_duration)

        if scope_key != self._GLOBAL_SCOPE and scope_key not in self._states:
            self._scope_snapshots.pop(scope_key, None)
            return None

        scope_descriptor = state.scope_descriptor
        if scope_descriptor is None and self._config.scope_fields:
            scope_descriptor = MappingProxyType(
                {field: None for field in self._config.scope_fields}
            )

        metadata_context: Mapping[str, Any] | None = None
        snapshot_metadata = decision.snapshot.get("metadata")
        if isinstance(snapshot_metadata, Mapping):
            context_payload = snapshot_metadata.get("context")
            if isinstance(context_payload, Mapping):
                metadata_context = dict(context_payload)

        recent_trades = len(timestamps)
        cooldown_until = state.cooldown_until

        throttle_state = "open"
        reason: str | None = None
        active = False
        retry_at: datetime | None = None
        retry_in_seconds: float | None = None
        message: str | None = None
        cooldown_payload = self._external_cooldowns.get(scope_key)

        if cooldown_until is not None and moment < cooldown_until:
            throttle_state = "cooldown"
            if cooldown_payload is not None:
                reason = cooldown_payload.reason
                message = cooldown_payload.message
            else:
                reason = "cooldown_active"
            active = True
            retry_at = cooldown_until
            retry_in_seconds = max((retry_at - moment).total_seconds(), 0.0)
        elif self._config.min_spacing_seconds > 0.0 and state.last_trade is not None:
            min_spacing = timedelta(seconds=float(self._config.min_spacing_seconds))
            if moment - state.last_trade < min_spacing:
                throttle_state = "min_interval"
                reason = f"min_interval_{float(self._config.min_spacing_seconds):g}s"
                retry_at = state.last_trade + min_spacing
                retry_in_seconds = max((retry_at - moment).total_seconds(), 0.0)

        window_reset_at: datetime | None = None
        window_reset_in_seconds: float | None = None
        if timestamps:
            oldest = timestamps[0]
            reset_target = oldest + window_duration
            window_reset_at = reset_target
            window_reset_in_seconds = max((reset_target - moment).total_seconds(), 0.0)

        snapshot = self._build_snapshot(
            state=throttle_state,
            active=active,
            reason=reason,
            retry_at=retry_at,
            metadata=metadata_context,
            message=message or self._format_reason(reason, retry_at),
            scope_key=scope_key,
            scope_descriptor=scope_descriptor,
            recent_trades=recent_trades,
            cooldown_until=cooldown_until,
            moment=moment,
            window_reset_at=window_reset_at,
            window_reset_in_seconds=window_reset_in_seconds,
            retry_in_seconds=retry_in_seconds,
            cooldown_metadata=(
                cooldown_payload.metadata if cooldown_payload is not None else None
            ),
            notional_total=state.notional_total,
            attempted_notional=None,
        )

        if scope_key == self._GLOBAL_SCOPE:
            self._last_snapshot = snapshot
        if recent_trades or scope_key == self._GLOBAL_SCOPE:
            self._scope_snapshots[scope_key] = snapshot
        else:
            self._scope_snapshots.pop(scope_key, None)

        if cooldown_until is None and scope_key in self._external_cooldowns:
            self._external_cooldowns.pop(scope_key, None)

        return snapshot

    def apply_external_cooldown(
        self,
        duration_seconds: float,
        *,
        reason: str = "external_cooldown",
        message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        scope_metadata: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> Mapping[str, Any]:
        """Impose a cooldown window without consuming trade credits."""

        if duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive")

        moment = self._coerce_timestamp(now)
        scope_key, scope_descriptor = self._resolve_scope(scope_metadata)
        state = self._get_state(scope_key)
        if scope_descriptor:
            state.scope_descriptor = MappingProxyType(dict(scope_descriptor))
        elif self._config.scope_fields:
            state.scope_descriptor = MappingProxyType(
                {field: None for field in self._config.scope_fields}
            )
        else:
            state.scope_descriptor = None

        cooldown_until = moment + timedelta(seconds=float(duration_seconds))
        state.cooldown_until = cooldown_until

        metadata_payload: Mapping[str, Any] | None
        if metadata is None:
            metadata_payload = None
        elif isinstance(metadata, Mapping):
            metadata_payload = MappingProxyType(dict(metadata))
        else:
            metadata_payload = MappingProxyType(dict(metadata))  # type: ignore[arg-type]

        self._external_cooldowns[scope_key] = TradeThrottle._ExternalCooldown(
            reason=reason,
            message=message,
            metadata=metadata_payload,
        )

        retry_in_seconds = max((cooldown_until - moment).total_seconds(), 0.0)
        snapshot = self._build_snapshot(
            state="cooldown",
            active=True,
            reason=reason,
            retry_at=cooldown_until,
            metadata=scope_metadata,
            message=message or self._format_reason(reason, cooldown_until),
            scope_key=scope_key,
            scope_descriptor=state.scope_descriptor,
            recent_trades=len(state.timestamps),
            cooldown_until=cooldown_until,
            moment=moment,
            window_reset_at=None,
            window_reset_in_seconds=None,
            retry_in_seconds=retry_in_seconds,
            cooldown_metadata=metadata_payload,
            notional_total=state.notional_total,
            attempted_notional=None,
        )

        self._last_snapshot = snapshot
        self._scope_snapshots[scope_key] = snapshot
        return snapshot

    def _initial_snapshot(self) -> Mapping[str, Any]:
        now = datetime.now(tz=UTC)

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
            moment=now,
            window_reset_at=None,
            window_reset_in_seconds=None,
            retry_in_seconds=None,
            notional_total=0.0,
            attempted_notional=None,
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
        moment: datetime,
        window_reset_at: datetime | None,
        window_reset_in_seconds: float | None,
        retry_in_seconds: float | None,
        cooldown_metadata: Mapping[str, Any] | None = None,
        notional_total: float = 0.0,
        attempted_notional: float | None = None,
    ) -> Mapping[str, Any]:
        remaining_trades = max(self._config.max_trades - int(recent_trades), 0)

        meta_payload: MutableMapping[str, Any] = {
            "max_trades": self._config.max_trades,
            "window_seconds": float(self._config.window_seconds),
            "recent_trades": recent_trades,
            "remaining_trades": remaining_trades,
        }
        if self._config.max_notional is not None:
            max_notional_value = float(self._config.max_notional)
            consumed_notional = max(float(notional_total), 0.0)
            remaining_notional = max(max_notional_value - consumed_notional, 0.0)
            utilisation = (
                0.0
                if max_notional_value <= 0.0
                else min(consumed_notional / max_notional_value, 1.0)
            )
            meta_payload["max_notional"] = max_notional_value
            meta_payload["consumed_notional"] = consumed_notional
            meta_payload["remaining_notional"] = remaining_notional
            meta_payload["notional_utilisation"] = utilisation
            if attempted_notional is not None:
                try:
                    meta_payload["attempted_notional"] = float(attempted_notional)
                except (TypeError, ValueError):
                    meta_payload["attempted_notional"] = attempted_notional
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
        if isinstance(metadata, Mapping):
            meta_payload["context"] = dict(metadata)
        if cooldown_metadata is not None:
            meta_payload["cooldown_context"] = dict(cooldown_metadata)
        utilisation = recent_trades / float(self._config.max_trades)
        utilisation = min(max(utilisation, 0.0), 1.0)
        meta_payload["window_utilisation"] = utilisation
        if window_reset_at is not None:
            meta_payload["window_reset_at"] = window_reset_at.astimezone(UTC).isoformat()
        if window_reset_in_seconds is not None:
            meta_payload["window_reset_in_seconds"] = max(window_reset_in_seconds, 0.0)

        retry_iso: str | None = None
        if retry_at is not None:
            retry_iso = retry_at.astimezone(UTC).isoformat()
            meta_payload["retry_at"] = retry_iso
        if retry_in_seconds is not None:
            meta_payload["retry_in_seconds"] = max(retry_in_seconds, 0.0)

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
            if isinstance(reason, str) and "cooldown" in reason:
                meta_payload["cooldown_reason"] = reason
        if message:
            snapshot["message"] = message
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

    def _resolve_notional(self, metadata: Mapping[str, Any] | None) -> float | None:
        if not isinstance(metadata, Mapping):
            return None
        candidate = metadata.get("notional")
        if candidate is None:
            return None
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return abs(value)

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
    ) -> bool:
        timestamps = state.timestamps
        while timestamps and moment - timestamps[0] >= window:
            timestamps.popleft()
        notional_records = state.notional_records
        while notional_records and moment - notional_records[0][0] >= window:
            _record_time, record_notional = notional_records.popleft()
            state.notional_total -= float(record_notional)
        state.notional_total = max(state.notional_total, 0.0)
        if state.cooldown_until is not None and moment >= state.cooldown_until:
            state.cooldown_until = None
            self._external_cooldowns.pop(scope_key, None)
        return self._cleanup_scope_if_idle(scope_key, state, moment, window)

    def _cleanup_scope_if_idle(
        self,
        scope_key: tuple[str, ...],
        state: "TradeThrottle._ThrottleState",
        moment: datetime,
        window: timedelta,
    ) -> bool:
        if scope_key == self._GLOBAL_SCOPE:
            return False
        if not self._should_remove_scope(state, moment, window):
            return False
        state.last_trade = None
        state.notional_records.clear()
        state.notional_total = 0.0
        self._states.pop(scope_key, None)
        self._scope_snapshots.pop(scope_key, None)
        return True

    def _purge_stale_scopes(self, moment: datetime, window: timedelta) -> None:
        for key, state in list(self._states.items()):
            if key == self._GLOBAL_SCOPE:
                continue
            self._prune(key, moment, window, state)

    def _should_remove_scope(
        self,
        state: "TradeThrottle._ThrottleState",
        moment: datetime,
        window: timedelta,
    ) -> bool:
        if state.timestamps:
            return False
        if state.notional_records:
            return False
        cooldown_until = state.cooldown_until
        if cooldown_until is not None and moment < cooldown_until:
            return False

        expiry_threshold = self._resolve_expiry_threshold(window)
        last_trade = state.last_trade
        if last_trade is None:
            return True
        if expiry_threshold <= timedelta(0):
            return True
        return moment - last_trade >= expiry_threshold

    def _resolve_expiry_threshold(self, window: timedelta) -> timedelta:
        min_spacing_seconds = float(self._config.min_spacing_seconds)
        if min_spacing_seconds <= 0.0:
            return window
        min_spacing = timedelta(seconds=min_spacing_seconds)
        return window if window >= min_spacing else min_spacing

    @staticmethod
    def _coerce_timestamp(candidate: datetime | None) -> datetime:
        if candidate is None:
            return datetime.now(tz=UTC)
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=UTC)
        return candidate.astimezone(UTC)

    @staticmethod
    def _clone_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(snapshot)
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            metadata_copy: dict[str, Any] = dict(metadata)
            scope_meta = metadata_copy.get("scope")
            if isinstance(scope_meta, Mapping):
                metadata_copy["scope"] = dict(scope_meta)
            cooldown_meta = metadata_copy.get("cooldown_context")
            if isinstance(cooldown_meta, Mapping):
                metadata_copy["cooldown_context"] = dict(cooldown_meta)
            payload["metadata"] = metadata_copy
        scope_key = payload.get("scope_key")
        if isinstance(scope_key, list):
            payload["scope_key"] = list(scope_key)
        return payload
    def _format_reason(self, reason: str | None, retry_at: datetime | None) -> str | None:
        if not reason:
            return None

        if reason == "cooldown_active":
            if retry_at is not None:
                iso_retry = retry_at.astimezone(UTC).isoformat()
                return f"Throttle cooldown active until {iso_retry}"
            return "Throttle cooldown active"

        if reason == "backlog_cooldown":
            if retry_at is not None:
                iso_retry = retry_at.astimezone(UTC).isoformat()
                return f"Throttled: backlog cooldown active until {iso_retry}"
            return "Throttled: backlog cooldown active"

        match = self._RATE_LIMIT_REASON.match(reason)
        if match:
            count = int(match.group("count"))
            window_seconds = float(match.group("window"))
            trades_label = "trade" if count == 1 else "trades"
            window_label = self._format_window_description(window_seconds)
            return (
                "Throttled: too many trades in short time "
                f"(limit {count} {trades_label} per {window_label})"
            )

        notional_match = self._NOTIONAL_LIMIT_REASON.match(reason)
        if notional_match:
            notional_limit = float(notional_match.group("notional"))
            window_seconds = float(notional_match.group("window"))
            window_label = self._format_window_description(window_seconds)
            notional_label = self._format_notional_value(notional_limit)
            return (
                "Throttled: notional limit exceeded "
                f"(limit {notional_label} per {window_label})"
            )

        return f"Throttle reason: {reason}"

    @staticmethod
    def _format_window_description(window_seconds: float) -> str:
        seconds = float(window_seconds)
        if seconds <= 0.0:
            return "0 seconds"
        if seconds >= 3600.0:
            hours = seconds / 3600.0
            if math.isclose(hours, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                return "1 hour"
            return f"{hours:g} hours"
        if seconds >= 60.0:
            minutes = seconds / 60.0
            if math.isclose(minutes, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                return "1 minute"
            return f"{minutes:g} minutes"
        if math.isclose(seconds, 1.0, abs_tol=1e-9):
            return "1 second"
        return f"{seconds:g} seconds"

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

    @staticmethod
    def _format_notional_value(value: float) -> str:
        if not math.isfinite(value):
            return str(value)
        absolute = abs(value)
        if absolute >= 1_000_000:
            return f"{value:,.0f}"
        if absolute >= 1_000:
            formatted = f"{value:,.2f}"
        else:
            formatted = f"{value:g}"
        return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
