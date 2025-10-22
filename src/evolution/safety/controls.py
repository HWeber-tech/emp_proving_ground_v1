"""Guard rails for controlling when evolution cycles may execute.

The safety controller complements :mod:`src.evolution.engine.scheduler` by
adding risk-aware gating on top of the time/telemetry triggers.  It applies a
policy of quantitative thresholds – drawdown limits, gross exposure caps, data
quality floors – and records breach history to provide cooling periods and
lock-down escalation when violations persist.

The implementation is intentionally deterministic and side-effect free so it can
be exercised in unit tests and wired into orchestration flows without extra
infrastructure dependencies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable, Deque, Iterable, Mapping

__all__ = [
    "EvolutionSafetyPolicy",
    "EvolutionSafetyViolation",
    "EvolutionSafetyDecision",
    "EvolutionSafetyState",
    "EvolutionSafetyController",
]


def _ensure_aware(moment: datetime) -> datetime:
    """Return a timezone-aware timestamp normalised to UTC."""

    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _coerce_float(value: object) -> float | None:
    """Best-effort conversion of loose numeric inputs."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _extract_metric(payload: Mapping[str, object] | object, name: str) -> float | None:
    """Read a named metric from mappings or attribute containers."""

    if isinstance(payload, Mapping):
        candidate = payload.get(name)
    else:
        candidate = getattr(payload, name, None)
    return _coerce_float(candidate)


@dataclass(slots=True, frozen=True)
class EvolutionSafetyPolicy:
    """Configuration governing evolution safety enforcement."""

    max_drawdown: float = 0.12
    max_value_at_risk: float = 0.06
    max_gross_exposure: float = 2.5
    max_position_concentration: float = 0.30
    max_latency_ms: float = 250.0
    max_slippage_bps: float = 20.0
    min_data_completeness: float = 0.97
    cooldown: timedelta = timedelta(minutes=15)
    lockout_threshold: int = 3
    lockout_window: timedelta = timedelta(hours=1)

    def __post_init__(self) -> None:
        if self.max_drawdown <= 0:
            raise ValueError("max_drawdown must be positive")
        if self.max_value_at_risk <= 0:
            raise ValueError("max_value_at_risk must be positive")
        if self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be positive")
        if self.max_position_concentration <= 0:
            raise ValueError("max_position_concentration must be positive")
        if self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")
        if self.max_slippage_bps <= 0:
            raise ValueError("max_slippage_bps must be positive")
        if not 0 < self.min_data_completeness <= 1:
            raise ValueError("min_data_completeness must be in (0, 1]")
        if self.cooldown < timedelta(0):
            raise ValueError("cooldown cannot be negative")
        if self.lockout_threshold < 0:
            raise ValueError("lockout_threshold cannot be negative")
        if self.lockout_window <= timedelta(0):
            raise ValueError("lockout_window must be positive")


@dataclass(slots=True, frozen=True)
class EvolutionSafetyViolation:
    """Structured description of a breached safety gate."""

    gate: str
    metric: str
    observed: float
    limit: float
    comparator: str
    severity: str = "critical"

    def as_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "metric": self.metric,
            "observed": self.observed,
            "limit": self.limit,
            "comparator": self.comparator,
            "severity": self.severity,
        }


@dataclass(slots=True, frozen=True)
class EvolutionSafetyDecision:
    """Decision emitted when evaluating current evolution risk posture."""

    allowed: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, float]
    violations: tuple[EvolutionSafetyViolation, ...]
    cooldown_active: bool
    lockdown_active: bool
    evaluated_at: datetime

    def as_dict(self) -> dict[str, object]:
        payload = {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "violations": [violation.as_dict() for violation in self.violations],
            "cooldown_active": self.cooldown_active,
            "lockdown_active": self.lockdown_active,
            "evaluated_at": self.evaluated_at.isoformat(),
        }
        return payload


@dataclass(slots=True, frozen=True)
class EvolutionSafetyState:
    """Snapshot of controller state for observability surfaces."""

    last_breach_at: datetime | None
    cooldown_expires_at: datetime | None
    breach_count: int
    lockdown_active: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "last_breach_at": self.last_breach_at.isoformat() if self.last_breach_at else None,
            "cooldown_expires_at": self.cooldown_expires_at.isoformat()
            if self.cooldown_expires_at
            else None,
            "breach_count": self.breach_count,
            "lockdown_active": self.lockdown_active,
        }


class EvolutionSafetyController:
    """Evaluate safety policy compliance for evolution orchestration."""

    _METRIC_NAMES: tuple[str, ...] = (
        "max_drawdown",
        "value_at_risk",
        "gross_exposure",
        "position_concentration",
        "latency_ms_p95",
        "slippage_bps",
        "data_completeness",
    )

    def __init__(
        self,
        policy: EvolutionSafetyPolicy | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._policy = policy or EvolutionSafetyPolicy()
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._breaches: Deque[datetime] = deque()
        self._last_breach: datetime | None = None
        self._last_decision: EvolutionSafetyDecision | None = None

    @property
    def policy(self) -> EvolutionSafetyPolicy:
        return self._policy

    @property
    def last_decision(self) -> EvolutionSafetyDecision | None:
        return self._last_decision

    def evaluate(
        self,
        metrics: Mapping[str, object] | object,
    ) -> EvolutionSafetyDecision:
        """Evaluate the provided metrics against the configured policy."""

        timestamp = _ensure_aware(self._now())
        self._prune_breaches(timestamp)

        normalised = self._normalise_metrics(metrics)
        violations = tuple(self._check_violations(normalised))
        cooldown_active = self._cooldown_active(timestamp)
        lockdown_active = self._lockdown_active(pending=bool(violations))

        allowed = not violations and not cooldown_active and not lockdown_active

        reasons: list[str] = []
        if violations:
            reasons.append("metric_violations")
        if cooldown_active:
            reasons.append("cooldown_active")
        if lockdown_active:
            reasons.append("lockdown_active")

        decision = EvolutionSafetyDecision(
            allowed=allowed,
            reasons=tuple(reasons),
            metrics=normalised,
            violations=violations,
            cooldown_active=cooldown_active,
            lockdown_active=lockdown_active,
            evaluated_at=timestamp,
        )

        if violations:
            self._register_breach(timestamp)
        self._last_decision = decision
        return decision

    def state(self) -> EvolutionSafetyState:
        """Return a snapshot describing the current controller posture."""

        now = _ensure_aware(self._now())
        self._prune_breaches(now)
        cooldown_expires = self._cooldown_expiry()
        return EvolutionSafetyState(
            last_breach_at=self._last_breach,
            cooldown_expires_at=cooldown_expires,
            breach_count=len(self._breaches),
            lockdown_active=self._lockdown_active(),
        )

    def reset(self) -> None:
        """Clear stored breach history (used in tests or manual overrides)."""

        self._breaches.clear()
        self._last_breach = None
        self._last_decision = None

    def _normalise_metrics(self, metrics: Mapping[str, object] | object) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in self._METRIC_NAMES:
            value = _extract_metric(metrics, name)
            if value is not None:
                result[name] = value
        return result

    def _check_violations(self, metrics: Mapping[str, float]) -> Iterable[EvolutionSafetyViolation]:
        policy = self._policy

        drawdown = metrics.get("max_drawdown")
        if drawdown is not None and drawdown > policy.max_drawdown:
            yield EvolutionSafetyViolation(
                gate="max_drawdown",
                metric="max_drawdown",
                observed=drawdown,
                limit=policy.max_drawdown,
                comparator="<=",
            )

        var_value = metrics.get("value_at_risk")
        if var_value is not None and var_value > policy.max_value_at_risk:
            yield EvolutionSafetyViolation(
                gate="value_at_risk",
                metric="value_at_risk",
                observed=var_value,
                limit=policy.max_value_at_risk,
                comparator="<=",
            )

        exposure = metrics.get("gross_exposure")
        if exposure is not None and exposure > policy.max_gross_exposure:
            yield EvolutionSafetyViolation(
                gate="gross_exposure",
                metric="gross_exposure",
                observed=exposure,
                limit=policy.max_gross_exposure,
                comparator="<=",
            )

        concentration = metrics.get("position_concentration")
        if concentration is not None and concentration > policy.max_position_concentration:
            yield EvolutionSafetyViolation(
                gate="position_concentration",
                metric="position_concentration",
                observed=concentration,
                limit=policy.max_position_concentration,
                comparator="<=",
            )

        latency = metrics.get("latency_ms_p95")
        if latency is not None and latency > policy.max_latency_ms:
            yield EvolutionSafetyViolation(
                gate="latency_ms",
                metric="latency_ms_p95",
                observed=latency,
                limit=policy.max_latency_ms,
                comparator="<=",
            )

        slippage = metrics.get("slippage_bps")
        if slippage is not None and slippage > policy.max_slippage_bps:
            yield EvolutionSafetyViolation(
                gate="slippage_bps",
                metric="slippage_bps",
                observed=slippage,
                limit=policy.max_slippage_bps,
                comparator="<=",
            )

        data_quality = metrics.get("data_completeness")
        if data_quality is not None and data_quality < policy.min_data_completeness:
            yield EvolutionSafetyViolation(
                gate="data_completeness",
                metric="data_completeness",
                observed=data_quality,
                limit=policy.min_data_completeness,
                comparator=">=",
            )

    def _register_breach(self, timestamp: datetime) -> None:
        self._breaches.append(timestamp)
        self._last_breach = timestamp

    def _prune_breaches(self, now: datetime) -> None:
        if not self._breaches:
            return
        window_start = now - self._policy.lockout_window
        while self._breaches and self._breaches[0] < window_start:
            self._breaches.popleft()

    def _cooldown_expiry(self) -> datetime | None:
        if self._last_breach is None:
            return None
        if self._policy.cooldown == timedelta(0):
            return None
        return self._last_breach + self._policy.cooldown

    def _cooldown_active(self, now: datetime) -> bool:
        expiry = self._cooldown_expiry()
        if expiry is None:
            return False
        return now < expiry

    def _lockdown_active(self, *, pending: bool = False) -> bool:
        if self._policy.lockout_threshold == 0:
            return False
        current = len(self._breaches) + (1 if pending else 0)
        return current >= self._policy.lockout_threshold
