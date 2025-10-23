"""Strategy-level emergency controls for individual trading strategies."""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "StrategyControlError",
    "StrategyRiskLimitBreach",
    "StrategyControlState",
    "StrategyControls",
]


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat()


def _from_iso(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _normalise_strategy_id(strategy_id: str) -> str:
    token = str(strategy_id or "").strip()
    if not token:
        raise ValueError("strategy_id must be a non-empty string")
    return token


class StrategyControlError(RuntimeError):
    """Raised when strategy control operations fail."""


@dataclass(frozen=True, slots=True)
class StrategyRiskLimitBreach:
    """Represents a risk threshold violation for a strategy."""

    strategy_id: str
    limit: str
    threshold: float
    observed: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "limit": self.limit,
            "threshold": self.threshold,
            "observed": self.observed,
        }


@dataclass(slots=True)
class StrategyControlState:
    """Materialised control state for a single strategy."""

    strategy_id: str
    paused: bool = False
    pause_reason: str | None = None
    risk_limits: dict[str, float] = field(default_factory=dict)
    quarantined: bool = False
    quarantine_reason: str | None = None
    quarantine_expires_at: datetime | None = None
    updated_at: datetime = field(default_factory=_utc_now)
    updated_by: str | None = None

    def snapshot(self) -> "StrategyControlState":
        clone = StrategyControlState(
            strategy_id=self.strategy_id,
            paused=self.paused,
            pause_reason=self.pause_reason,
            risk_limits=dict(self.risk_limits),
            quarantined=self.quarantined,
            quarantine_reason=self.quarantine_reason,
            quarantine_expires_at=self.quarantine_expires_at,
            updated_at=self.updated_at,
            updated_by=self.updated_by,
        )
        return clone

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "strategy_id": self.strategy_id,
            "paused": self.paused,
            "risk_limits": dict(self.risk_limits),
            "quarantined": self.quarantined,
        }
        if self.pause_reason:
            payload["pause_reason"] = self.pause_reason
        if self.quarantine_reason:
            payload["quarantine_reason"] = self.quarantine_reason
        if self.quarantine_expires_at is not None:
            payload["quarantine_expires_at"] = _to_iso(self.quarantine_expires_at)
        if self.updated_at is not None:
            payload["updated_at"] = _to_iso(self.updated_at)
        if self.updated_by:
            payload["updated_by"] = self.updated_by
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StrategyControlState":
        strategy_id = _normalise_strategy_id(payload.get("strategy_id", ""))
        paused = bool(payload.get("paused", False))
        pause_reason_value = payload.get("pause_reason")
        pause_reason = str(pause_reason_value) if pause_reason_value else None
        risk_limits_payload = payload.get("risk_limits") or {}
        risk_limits: dict[str, float] = {}
        if isinstance(risk_limits_payload, Mapping):
            for key, value in risk_limits_payload.items():
                try:
                    limit_value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(limit_value) or math.isinf(limit_value) or limit_value < 0:
                    continue
                risk_limits[str(key)] = limit_value
        quarantined = bool(payload.get("quarantined", False))
        quarantine_reason_value = payload.get("quarantine_reason")
        quarantine_reason = (
            str(quarantine_reason_value).strip()
            if quarantine_reason_value
            else None
        )
        quarantine_expires_at = _from_iso(payload.get("quarantine_expires_at"))
        updated_at = _from_iso(payload.get("updated_at")) or _utc_now()
        updated_by_value = payload.get("updated_by")
        updated_by = str(updated_by_value).strip() if updated_by_value else None
        return cls(
            strategy_id=strategy_id,
            paused=paused,
            pause_reason=pause_reason,
            risk_limits=risk_limits,
            quarantined=quarantined,
            quarantine_reason=quarantine_reason,
            quarantine_expires_at=quarantine_expires_at,
            updated_at=updated_at,
            updated_by=updated_by,
        )


class StrategyControls:
    """Manage per-strategy operational controls with persistence."""

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        auto_flush: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        resolved = Path(path or "artifacts/governance/strategy_controls.json")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._path = resolved
        self._auto_flush = bool(auto_flush)
        self._logger = logger or logging.getLogger(f"{__name__}.StrategyControls")
        self._lock = threading.RLock()
        self._states: dict[str, StrategyControlState] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public query helpers
    # ------------------------------------------------------------------
    def list_states(self) -> tuple[StrategyControlState, ...]:
        with self._lock:
            return tuple(state.snapshot() for state in self._states.values())

    def get_state(self, strategy_id: str) -> StrategyControlState | None:
        key = _normalise_strategy_id(strategy_id)
        with self._lock:
            state = self._states.get(key)
            return state.snapshot() if state else None

    def is_paused(self, strategy_id: str) -> bool:
        state = self.get_state(strategy_id)
        return bool(state.paused) if state else False

    def is_quarantined(self, strategy_id: str) -> bool:
        state = self.get_state(strategy_id)
        if state is None:
            return False
        if not state.quarantined:
            return False
        if state.quarantine_expires_at and state.quarantine_expires_at <= _utc_now():
            return False
        return True

    def get_risk_limits(self, strategy_id: str) -> Mapping[str, float]:
        state = self.get_state(strategy_id)
        return dict(state.risk_limits) if state else {}

    # ------------------------------------------------------------------
    # Control operations
    # ------------------------------------------------------------------
    def pause_strategy(
        self,
        strategy_id: str,
        *,
        reason: str | None = None,
        actor: str | None = None,
    ) -> None:
        key = _normalise_strategy_id(strategy_id)
        reason_text = reason.strip() if isinstance(reason, str) else None
        with self._lock:
            state = self._get_or_create_state(key)
            state.paused = True
            state.pause_reason = reason_text
            self._touch(state, actor)
            self._flush()

    def resume_strategy(self, strategy_id: str, *, actor: str | None = None) -> None:
        key = _normalise_strategy_id(strategy_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return
            if state.quarantined and not self._quarantine_expired(state):
                raise StrategyControlError(
                    f"Strategy {key} remains quarantined; release quarantine before resuming."
                )
            state.paused = False
            state.pause_reason = None
            self._touch(state, actor)
            self._flush()

    def set_risk_limits(
        self,
        strategy_id: str,
        limits: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
        *,
        actor: str | None = None,
    ) -> None:
        key = _normalise_strategy_id(strategy_id)
        normalised = self._normalise_limits(limits)
        with self._lock:
            state = self._get_or_create_state(key)
            state.risk_limits = normalised
            self._touch(state, actor)
            self._flush()

    def clear_risk_limits(self, strategy_id: str, *, actor: str | None = None) -> None:
        self.set_risk_limits(strategy_id, {}, actor=actor)

    def quarantine_strategy(
        self,
        strategy_id: str,
        *,
        reason: str,
        actor: str | None = None,
        expires_at: datetime | str | None = None,
        also_pause: bool = True,
    ) -> None:
        key = _normalise_strategy_id(strategy_id)
        if not reason or not str(reason).strip():
            raise ValueError("Quarantine reason must be provided")
        expiry = _from_iso(expires_at) if isinstance(expires_at, str) else expires_at
        if expiry is not None and expiry <= _utc_now():
            raise ValueError("Quarantine expiry must be in the future")
        with self._lock:
            state = self._get_or_create_state(key)
            state.quarantined = True
            state.quarantine_reason = str(reason).strip()
            state.quarantine_expires_at = expiry
            if also_pause:
                state.paused = True
                state.pause_reason = state.pause_reason or state.quarantine_reason
            self._touch(state, actor)
            self._flush()

    def release_quarantine(self, strategy_id: str, *, actor: str | None = None) -> None:
        key = _normalise_strategy_id(strategy_id)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return
            state.quarantined = False
            state.quarantine_reason = None
            state.quarantine_expires_at = None
            self._touch(state, actor)
            self._flush()

    # ------------------------------------------------------------------
    # Risk evaluation
    # ------------------------------------------------------------------
    def check_risk_limits(
        self,
        strategy_id: str,
        telemetry: Mapping[str, Any] | None,
    ) -> tuple[bool, tuple[StrategyRiskLimitBreach, ...]]:
        key = _normalise_strategy_id(strategy_id)
        state = self.get_state(key)
        if state is None or not state.risk_limits:
            return True, ()
        metrics = telemetry or {}
        breaches: list[StrategyRiskLimitBreach] = []
        for limit_key, threshold in state.risk_limits.items():
            value = self._extract_metric(limit_key, metrics)
            if value is None:
                continue
            if self._violates(limit_key, value, threshold):
                breaches.append(
                    StrategyRiskLimitBreach(
                        strategy_id=key,
                        limit=limit_key,
                        threshold=threshold,
                        observed=value,
                    )
                )
        return (len(breaches) == 0, tuple(breaches))

    def enforce_risk_limits(
        self,
        strategy_id: str,
        telemetry: Mapping[str, Any] | None,
    ) -> None:
        ok, breaches = self.check_risk_limits(strategy_id, telemetry)
        if not ok:
            joined = ", ".join(
                f"{breach.limit} (observed={breach.observed}, threshold={breach.threshold})"
                for breach in breaches
            )
            raise StrategyControlError(
                f"Strategy {strategy_id} breached risk limits: {joined}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_or_create_state(self, strategy_id: str) -> StrategyControlState:
        state = self._states.get(strategy_id)
        if state is None:
            state = StrategyControlState(strategy_id=strategy_id)
            self._states[strategy_id] = state
        return state

    def _touch(self, state: StrategyControlState, actor: str | None) -> None:
        state.updated_at = _utc_now()
        state.updated_by = str(actor).strip() if actor else None

    def _quarantine_expired(self, state: StrategyControlState) -> bool:
        expiry = state.quarantine_expires_at
        if expiry is None:
            return False
        return expiry <= _utc_now()

    def _normalise_limits(
        self,
        limits: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
    ) -> dict[str, float]:
        if limits is None:
            return {}
        if isinstance(limits, Mapping):
            iterable: Iterable[tuple[str, Any]] = limits.items()
        else:
            iterable = limits
        normalised: dict[str, float] = {}
        for key, value in iterable:
            name = str(key or "").strip()
            if not name:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise StrategyControlError(
                    f"Risk limit '{key}' must be numeric"
                ) from exc
            if math.isnan(numeric) or math.isinf(numeric):
                raise StrategyControlError(f"Risk limit '{key}' must be finite")
            if numeric < 0:
                raise StrategyControlError(f"Risk limit '{key}' must be greater or equal to zero")
            normalised[name] = numeric
        return normalised

    def _extract_metric(self, limit: str, telemetry: Mapping[str, Any]) -> float | None:
        candidates = _METRIC_CANDIDATES.get(limit)
        metric_keys: Sequence[str]
        if candidates is not None:
            metric_keys = candidates
        else:
            base = limit
            if base.startswith("max_"):
                base = base[len("max_") :]
            elif base.startswith("min_"):
                base = base[len("min_") :]
            metric_keys = (limit, base)
        for key in metric_keys:
            value = telemetry.get(key)
            numeric = self._to_float(value)
            if numeric is not None:
                return numeric
        return None

    @staticmethod
    def _violates(limit: str, value: float, threshold: float) -> bool:
        if limit.startswith("min_"):
            return value < threshold
        return value > threshold

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            numeric = float(value)
        else:
            try:
                numeric = float(str(value))
            except (TypeError, ValueError):
                return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._logger.warning("Failed to parse strategy controls at %s: %s", self._path, exc)
            return
        strategies = raw.get("strategies") if isinstance(raw, Mapping) else None
        if not isinstance(strategies, Mapping):
            return
        for strategy_id, payload in strategies.items():
            if not isinstance(payload, Mapping):
                continue
            try:
                state = StrategyControlState.from_payload({"strategy_id": strategy_id, **payload})
            except Exception:
                continue
            self._states[state.strategy_id] = state

    def _flush(self) -> None:
        if not self._auto_flush:
            return
        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "generated_at": _to_iso(_utc_now()),
            "strategies": {
                strategy_id: state.as_dict()
                for strategy_id, state in sorted(self._states.items())
            },
        }
        temp_path = self._path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self._path)


_METRIC_CANDIDATES: dict[str, Sequence[str]] = {
    "max_position_size": ("position_size", "current_position_size", "open_position_size", "exposure_units"),
    "max_notional": ("notional", "notional_exposure", "exposure_notional"),
    "max_drawdown": ("drawdown", "current_drawdown", "max_drawdown"),
    "max_drawdown_pct": ("drawdown_pct", "current_drawdown_pct", "max_drawdown_pct"),
    "max_leverage": ("leverage", "current_leverage"),
    "max_daily_loss": ("daily_loss", "pnl_daily", "pnl_day"),
    "max_open_positions": ("open_positions", "position_count", "open_position_count"),
    "min_signal_confidence": ("signal_confidence", "confidence"),
}
