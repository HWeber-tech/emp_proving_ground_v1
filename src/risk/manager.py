"""Drawdown circuit breaker utilities for the risk roadmap.

The high-impact roadmap calls for explicit drawdown circuit breakers that can
halt trading activity during severe equity shocks and progressively thaw risk
budgets once conditions improve.  This module centralises the behaviour so the
rest of the risk stack – `RiskManagerImpl`, sizing adapters, and telemetry – can
subscribe to a consistent signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

__all__ = [
    "CircuitBreakerState",
    "CircuitBreakerEvent",
    "DrawdownCircuitBreaker",
]


class CircuitBreakerState(str, Enum):
    """Lifecycle state returned after ingesting an equity observation."""

    NORMAL = "normal"
    WARNING = "warning"
    TRIPPED = "tripped"


@dataclass(frozen=True)
class CircuitBreakerEvent:
    """Snapshot emitted whenever the circuit breaker processes equity data."""

    state: CircuitBreakerState
    equity: float
    peak_equity: float
    drawdown: float
    """Fractional drawdown relative to the running peak (0 → 1+)."""

    utilisation: float
    """Drawdown expressed as utilisation of the configured limit."""

    scaling_factor: float
    """Risk budget multiplier suggested by the circuit breaker."""

    triggered: bool
    cooldown_remaining: int


class DrawdownCircuitBreaker:
    """Detect and react to drawdowns exceeding roadmap guardrails.

    Parameters
    ----------
    max_drawdown:
        Maximum fractional drawdown tolerated before the breaker trips.  Values
        should be expressed as decimals (``0.25`` for 25 %).
    warn_threshold:
        Fraction of the drawdown limit that triggers a warning state.  Defaults
        to 80 % of the configured limit to mirror the rest of the risk posture
        grading.
    floor:
        Minimum scaling factor returned when utilisation reaches 100 %.  The
        roadmap mandates throttling instead of a full shutdown so the
        automation can recover gracefully.  Defaults to ``0.25`` which matches
        the historic behaviour in :class:`RiskManagerImpl`.
    cooldown_steps:
        Number of consecutive equity observations required below the recovery
        threshold before clearing a tripped breaker.
    recovery_ratio:
        Fraction of the drawdown limit that must be respected before the
        cooldown counter begins ticking down.  Defaults to 50 % of the limit.
    """

    _MIN_FLOOR: Final[float] = 0.0

    def __init__(
        self,
        *,
        max_drawdown: float,
        warn_threshold: float = 0.8,
        floor: float = 0.25,
        cooldown_steps: int = 5,
        recovery_ratio: float = 0.5,
    ) -> None:
        if max_drawdown < 0:
            raise ValueError("max_drawdown must be non-negative")
        if warn_threshold <= 0:
            raise ValueError("warn_threshold must be positive")
        if recovery_ratio <= 0:
            raise ValueError("recovery_ratio must be positive")

        self.max_drawdown = float(max_drawdown)
        self.warn_threshold = float(warn_threshold)
        self.floor = max(self._MIN_FLOOR, float(floor))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.recovery_ratio = float(recovery_ratio)

        self._peak_equity: float = 0.0
        self._triggered: bool = False
        self._cooldown_remaining: int = 0

    def reset(self) -> None:
        """Reset the breaker to its initial state."""

        self._peak_equity = 0.0
        self._triggered = False
        self._cooldown_remaining = 0

    def record(self, equity: float) -> CircuitBreakerEvent:
        """Process a new equity observation and return the resulting event."""

        equity = max(0.0, float(equity))
        if equity > self._peak_equity:
            self._peak_equity = equity

        peak = self._peak_equity
        drawdown_fraction = 0.0
        if peak > 0:
            drawdown_fraction = max(0.0, (peak - equity) / peak)

        utilisation = self._compute_utilisation(drawdown_fraction)
        scaling_factor = self._compute_scaling(utilisation)

        state = CircuitBreakerState.NORMAL
        if self.max_drawdown > 0:
            if utilisation >= 1.0:
                state = CircuitBreakerState.TRIPPED
                self._triggered = True
                self._cooldown_remaining = self.cooldown_steps
            elif utilisation >= self.warn_threshold:
                state = CircuitBreakerState.WARNING

        if self._triggered and state is not CircuitBreakerState.TRIPPED:
            # Breaker previously tripped – evaluate recovery conditions.
            if utilisation <= self.recovery_ratio:
                if self._cooldown_remaining > 0:
                    self._cooldown_remaining -= 1
                if self._cooldown_remaining <= 0:
                    self._triggered = False
                    state = (
                        CircuitBreakerState.WARNING
                        if utilisation >= self.warn_threshold
                        else CircuitBreakerState.NORMAL
                    )
            else:
                self._cooldown_remaining = self.cooldown_steps
                state = CircuitBreakerState.TRIPPED

        triggered = self._triggered or state is CircuitBreakerState.TRIPPED
        if triggered and self._triggered:
            state = CircuitBreakerState.TRIPPED

        return CircuitBreakerEvent(
            state=state,
            equity=equity,
            peak_equity=peak,
            drawdown=drawdown_fraction,
            utilisation=utilisation,
            scaling_factor=scaling_factor,
            triggered=triggered,
            cooldown_remaining=self._cooldown_remaining,
        )

    @property
    def peak_equity(self) -> float:
        """Return the running peak equity."""

        return self._peak_equity

    @property
    def triggered(self) -> bool:
        """Return ``True`` if the breaker is currently tripped."""

        return self._triggered

    def should_halt_trading(self) -> bool:
        """Flag whether trading should be halted until recovery completes."""

        return self._triggered

    # Internal helpers -------------------------------------------------

    def _compute_utilisation(self, drawdown_fraction: float) -> float:
        if self.max_drawdown <= 0:
            return 0.0
        return drawdown_fraction / self.max_drawdown

    def _compute_scaling(self, utilisation: float) -> float:
        if self.max_drawdown <= 0:
            return 1.0

        scaled = 1.0 - utilisation
        scaled = max(self.floor, scaled)
        return max(self._MIN_FLOOR, min(1.0, scaled))

