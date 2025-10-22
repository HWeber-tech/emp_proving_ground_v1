"""Event-driven scheduler coordinating evolution cycles from live telemetry.

This module fulfils the roadmap item **Implement evolution scheduler** by
providing a deterministic decision engine that inspects recent live trading
telemetry – realised PnL, drawdown, and execution latency – and decides when an
evolution cycle should be triggered.  The scheduler is designed to consume
metrics sourced from Kafka topics (or any streaming feed) via the
``record_sample`` method.  It maintains a rolling time window of telemetry and
applies configurable guard rails so evolution runs only occur when performance
or execution risk breaches agreed thresholds.

Key capabilities
----------------
* Aggregates live telemetry over a fixed-duration sliding window.
* Applies independent triggers for PnL degradation, drawdown spikes, and
  excessive execution latency.
* Enforces a minimum interval between successive evolution runs to prevent
  thrashing when metrics oscillate around the thresholds.
* Exposes structured decision artefacts suitable for observability surfaces or
  downstream orchestration.

The scheduler purposefully avoids tight coupling with the orchestration layer;
callers can inspect :class:`EvolutionSchedulerDecision` and, when
``triggered`` is ``True``, launch an evolution cycle via
``src.orchestration.evolution_cycle.EvolutionCycleOrchestrator``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import mean
from typing import Callable, Deque, Mapping, Sequence

__all__ = [
    "EvolutionTelemetrySample",
    "EvolutionSchedulerConfig",
    "EvolutionSchedulerState",
    "EvolutionSchedulerDecision",
    "EvolutionScheduler",
]


def _ensure_aware(moment: datetime) -> datetime:
    """Return a timezone-aware timestamp coerced to UTC."""

    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return the ``percentile`` (0–1) using linear interpolation."""

    if not values:
        raise ValueError("values cannot be empty when computing a percentile")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")

    sorted_values = sorted(values)
    if percentile <= 0.0:
        return sorted_values[0]
    if percentile >= 1.0:
        return sorted_values[-1]

    index = percentile * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    if upper == lower:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


@dataclass(slots=True, frozen=True)
class EvolutionTelemetrySample:
    """Telemetry tuple capturing live performance metrics."""

    timestamp: datetime
    pnl: float
    drawdown: float
    latency_ms: float
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _ensure_aware(self.timestamp))

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "timestamp": self.timestamp.isoformat(),
            "pnl": float(self.pnl),
            "drawdown": float(self.drawdown),
            "latency_ms": float(self.latency_ms),
        }
        if isinstance(self.metadata, Mapping) and self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class EvolutionSchedulerConfig:
    """Runtime configuration governing scheduler sensitivity."""

    window: timedelta = timedelta(minutes=15)
    min_interval: timedelta = timedelta(minutes=5)
    min_samples: int = 5
    pnl_floor: float = 0.0
    drawdown_ceiling: float = 0.08
    latency_ceiling_ms: float = 250.0
    latency_percentile: float = 0.95
    stagnation_timeout: timedelta = timedelta(minutes=10)

    def __post_init__(self) -> None:
        if self.window <= timedelta(0):
            raise ValueError("window must be positive")
        if self.min_interval < timedelta(0):
            raise ValueError("min_interval cannot be negative")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if self.drawdown_ceiling < 0:
            raise ValueError("drawdown_ceiling must be non-negative")
        if self.latency_ceiling_ms <= 0:
            raise ValueError("latency_ceiling_ms must be positive")
        if not 0.0 <= self.latency_percentile <= 1.0:
            raise ValueError("latency_percentile must be in [0, 1]")
        if self.stagnation_timeout < timedelta(0):
            raise ValueError("stagnation_timeout cannot be negative")


@dataclass(slots=True, frozen=True)
class EvolutionSchedulerState:
    """Lightweight snapshot describing the scheduler state."""

    last_triggered_at: datetime | None = None
    last_decision: Mapping[str, object] | None = None
    sample_count: int = 0
    window_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "last_triggered_at": self.last_triggered_at.isoformat()
            if self.last_triggered_at
            else None,
            "sample_count": int(self.sample_count),
            "window_seconds": float(self.window_seconds),
        }
        if self.last_decision is not None:
            payload["last_decision"] = dict(self.last_decision)
        return payload


@dataclass(slots=True, frozen=True)
class EvolutionSchedulerDecision:
    """Decision emitted after evaluating live telemetry."""

    triggered: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, float]
    sample_count: int
    evaluated_at: datetime
    next_eligible_at: datetime | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "triggered": self.triggered,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "sample_count": self.sample_count,
            "evaluated_at": self.evaluated_at.isoformat(),
        }
        if self.next_eligible_at is not None:
            payload["next_eligible_at"] = self.next_eligible_at.isoformat()
        return payload


class EvolutionScheduler:
    """Consume live telemetry and decide when to launch evolution cycles."""

    def __init__(
        self,
        config: EvolutionSchedulerConfig | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self._config = config or EvolutionSchedulerConfig()
        self._now = now or (lambda: datetime.now(tz=UTC))
        self._samples: Deque[EvolutionTelemetrySample] = deque(maxlen=max_samples or 512)
        self._last_triggered_at: datetime | None = None
        self._last_decision: EvolutionSchedulerDecision | None = None

    @property
    def config(self) -> EvolutionSchedulerConfig:
        return self._config

    def record_sample(
        self,
        sample: EvolutionTelemetrySample | Mapping[str, object],
    ) -> EvolutionTelemetrySample:
        """Record a telemetry sample sourced from streaming infrastructure."""

        if isinstance(sample, EvolutionTelemetrySample):
            record = sample
        else:
            try:
                timestamp = sample["timestamp"]  # type: ignore[index]
                pnl = float(sample["pnl"])  # type: ignore[index]
                drawdown = float(sample["drawdown"])  # type: ignore[index]
                latency = float(sample["latency_ms"])  # type: ignore[index]
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError("sample mapping missing required keys") from exc
            metadata = sample.get("metadata") if isinstance(sample, Mapping) else None
            if not isinstance(timestamp, datetime):
                raise TypeError("timestamp must be a datetime instance")
            record = EvolutionTelemetrySample(
                timestamp=timestamp,
                pnl=pnl,
                drawdown=drawdown,
                latency_ms=latency,
                metadata=metadata if isinstance(metadata, Mapping) else None,
            )

        self._samples.append(record)
        return record

    def evaluate(self, *, now: datetime | None = None) -> EvolutionSchedulerDecision:
        """Evaluate current telemetry and decide whether to trigger evolution."""

        timestamp = _ensure_aware(now or self._now())
        self._prune_samples(timestamp)
        reasons: list[str] = []

        metrics = self._aggregate_metrics()
        sample_count = int(metrics.get("sample_count", 0))

        if sample_count < self._config.min_samples:
            reasons.append("insufficient_samples")
            decision = EvolutionSchedulerDecision(
                triggered=False,
                reasons=tuple(reasons),
                metrics=metrics,
                sample_count=sample_count,
                evaluated_at=timestamp,
                next_eligible_at=self._next_eligible_at(timestamp),
            )
            self._last_decision = decision
            return decision

        triggered, trigger_reasons = self._evaluate_triggers(metrics)
        reasons.extend(trigger_reasons)

        cooldown_active = self._cooldown_active(timestamp)
        if cooldown_active and triggered:
            reasons.append("cooldown_active")
            triggered = False

        stagnated = self._stagnation_detected(timestamp)
        if stagnated:
            reasons.append("stagnation_detected")
            if not cooldown_active:
                triggered = True

        decision = EvolutionSchedulerDecision(
            triggered=triggered,
            reasons=tuple(reasons),
            metrics=metrics,
            sample_count=sample_count,
            evaluated_at=timestamp,
            next_eligible_at=self._next_eligible_at(timestamp) if not triggered else None,
        )

        if decision.triggered:
            self._last_triggered_at = timestamp

        self._last_decision = decision
        return decision

    def state(self) -> EvolutionSchedulerState:
        """Return a serialisable snapshot of scheduler state."""

        decision_payload = self._last_decision.as_dict() if self._last_decision else None
        return EvolutionSchedulerState(
            last_triggered_at=self._last_triggered_at,
            last_decision=decision_payload,
            sample_count=len(self._samples),
            window_seconds=float(self._config.window.total_seconds()),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_samples(self, now: datetime) -> None:
        cutoff = now - self._config.window
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def _aggregate_metrics(self) -> dict[str, float]:
        sample_count = len(self._samples)
        if not sample_count:
            return {"sample_count": 0}

        pnl_values = [sample.pnl for sample in self._samples]
        drawdown_values = [sample.drawdown for sample in self._samples]
        latency_values = [sample.latency_ms for sample in self._samples]

        metrics: dict[str, float] = {
            "sample_count": float(sample_count),
            "pnl_average": float(mean(pnl_values)),
            "pnl_min": float(min(pnl_values)),
            "drawdown_max": float(max(drawdown_values)),
            "latency_average_ms": float(mean(latency_values)),
        }

        try:
            metrics["latency_percentile_ms"] = float(
                _percentile(latency_values, self._config.latency_percentile)
            )
        except ValueError:
            metrics["latency_percentile_ms"] = metrics["latency_average_ms"]

        return metrics

    def _evaluate_triggers(self, metrics: Mapping[str, float]) -> tuple[bool, list[str]]:
        triggered = False
        reasons: list[str] = []

        pnl_average = metrics.get("pnl_average", 0.0)
        pnl_min = metrics.get("pnl_min", 0.0)
        pnl_floor = self._config.pnl_floor
        if pnl_average <= pnl_floor or pnl_min <= pnl_floor:
            triggered = True
            reasons.append("pnl_breach")

        drawdown_max = metrics.get("drawdown_max", 0.0)
        if drawdown_max >= self._config.drawdown_ceiling:
            triggered = True
            reasons.append("drawdown_breach")

        latency_percentile = metrics.get("latency_percentile_ms", 0.0)
        if latency_percentile >= self._config.latency_ceiling_ms:
            triggered = True
            reasons.append("latency_breach")

        return triggered, reasons

    def _cooldown_active(self, now: datetime) -> bool:
        if self._last_triggered_at is None:
            return False
        return now - self._last_triggered_at < self._config.min_interval

    def _stagnation_detected(self, now: datetime) -> bool:
        if not self._samples:
            return False
        latest_sample = self._samples[-1]
        return now - latest_sample.timestamp >= self._config.stagnation_timeout

    def _next_eligible_at(self, now: datetime) -> datetime | None:
        if self._last_triggered_at is None:
            return None
        return self._last_triggered_at + self._config.min_interval

