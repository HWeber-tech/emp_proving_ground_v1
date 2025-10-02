"""Recorded sensory replay evaluator for adaptive evolution experiments.

The roadmap calls for proving adaptive strategies against recorded sensory
snapshots so evolution runs can be validated without live data feeds.  This
module provides lightweight helpers that transform sensory snapshots produced by
:class:`src.sensory.real_sensory_organ.RealSensoryOrgan` (or their serialized
counterparts) into deterministic evaluation metrics that can be consumed by the
evolution orchestrator.

Usage
-----
```
from src.evolution.evaluation.recorded_replay import (
    RecordedSensorySnapshot,
    RecordedSensoryEvaluator,
)

snapshots = [RecordedSensorySnapshot.from_snapshot(payload) for payload in data]
metrics = RecordedSensoryEvaluator(snapshots).evaluate(genome)
fitness_payload = metrics.to_fitness_payload()
```
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Iterable, Mapping, MutableSequence, Sequence

from src.sensory.signals import IntegratedSignal

__all__ = [
    "RecordedSensorySnapshot",
    "RecordedEvaluationResult",
    "RecordedSensoryEvaluator",
]


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.endswith("Z"):
            trimmed = trimmed[:-1] + "+00:00"
        for parser in (datetime.fromisoformat,):
            try:
                parsed = parser(trimmed)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
    return datetime.now(tz=timezone.utc)


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _try_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_integrated(payload: object) -> tuple[float, float]:
    if isinstance(payload, IntegratedSignal):
        return float(payload.strength), float(payload.confidence)
    if isinstance(payload, Mapping):
        strength = _coerce_float(payload.get("strength"), default=0.0)
        confidence = _coerce_float(payload.get("confidence"), default=0.0)
        return strength, confidence
    return 0.0, 0.0


def _extract_price_from_dimension(entry: Mapping[str, object]) -> float | None:
    value = entry.get("value") if isinstance(entry, Mapping) else None
    candidates: MutableSequence[object] = []
    if isinstance(value, Mapping):
        candidates.extend(
            value.get(name)
            for name in ("last_close", "price", "close", "mid", "mid_price")
        )
    metadata = entry.get("metadata") if isinstance(entry, Mapping) else None
    if isinstance(metadata, Mapping):
        candidates.extend(
            metadata.get(name)
            for name in ("last_close", "price", "close", "mid", "mid_price")
        )
    for candidate in candidates:
        price = _try_float(candidate)
        if price is not None:
            return price
    signal_value = entry.get("signal") if isinstance(entry, Mapping) else None
    if isinstance(signal_value, (int, float)):
        return float(signal_value)
    return None


def _extract_price(payload: Mapping[str, object] | None, default: float) -> float:
    if not isinstance(payload, Mapping):
        return default
    dimensions = payload.get("dimensions")
    if isinstance(dimensions, Mapping):
        for key in ("WHAT", "PRICE", "MARKET"):
            entry = dimensions.get(key)
            if isinstance(entry, Mapping):
                price = _extract_price_from_dimension(entry)
                if price is not None:
                    return price
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("last_price", "price", "close", "mid", "mid_price"):
            if key in metadata:
                price = _try_float(metadata.get(key))
                if price is not None:
                    return price
    return default


def _sorted_snapshots(snapshots: Iterable["RecordedSensorySnapshot"]) -> list["RecordedSensorySnapshot"]:
    return sorted(snapshots, key=lambda snap: snap.timestamp)


@dataclass(frozen=True, slots=True)
class RecordedSensorySnapshot:
    """Normalised view of a sensory snapshot for replay evaluation."""

    timestamp: datetime
    price: float
    strength: float
    confidence: float

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, object],
        *,
        default_price: float = 1.0,
    ) -> "RecordedSensorySnapshot":
        timestamp = _coerce_datetime(snapshot.get("generated_at"))
        strength, confidence = _extract_integrated(snapshot.get("integrated_signal"))
        price = _extract_price(snapshot, default=default_price)
        return cls(timestamp=timestamp, price=price, strength=strength, confidence=confidence)


@dataclass(frozen=True, slots=True)
class RecordedEvaluationResult:
    """Deterministic metrics summarising a replay backtest."""

    equity_curve: tuple[float, ...]
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    win_rate: float
    trades: int
    wins: int
    losses: int

    def to_fitness_payload(self) -> dict[str, float]:
        return {
            "fitness_score": float(self.total_return - self.max_drawdown),
            "total_return": float(self.total_return),
            "max_drawdown": float(self.max_drawdown),
            "sharpe_ratio": float(self.sharpe_ratio),
            "volatility": float(self.volatility),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "equity_curve": list(self.equity_curve),
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "volatility": self.volatility,
            "win_rate": self.win_rate,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
        }


class RecordedSensoryEvaluator:
    """Simulate a simple threshold-based strategy over recorded sensory data."""

    def __init__(
        self,
        snapshots: Sequence[RecordedSensorySnapshot] | Iterable[RecordedSensorySnapshot],
    ) -> None:
        self._snapshots = _sorted_snapshots(snapshots)

    def evaluate(
        self,
        genome: object | Mapping[str, object],
        *,
        min_confidence: float | None = None,
    ) -> RecordedEvaluationResult:
        params = self._extract_parameters(genome)
        entry_threshold = abs(params.get("entry_threshold", 0.35))
        exit_threshold = abs(params.get("exit_threshold", entry_threshold / 2 or 0.15))
        risk_fraction = max(0.0, min(1.0, params.get("risk_fraction", 0.2)))
        confidence_floor = (
            max(0.0, min(1.0, params.get("min_confidence", 0.5)))
            if min_confidence is None
            else max(0.0, min(1.0, min_confidence))
        )
        cooldown_steps = max(0, int(params.get("cooldown_steps", 0)))

        snapshots = self._snapshots
        if len(snapshots) < 2:
            return RecordedEvaluationResult(
                equity_curve=(1.0,),
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                volatility=0.0,
                win_rate=0.0,
                trades=0,
                wins=0,
                losses=0,
            )

        equity = 1.0
        equity_curve: list[float] = [equity]
        returns: list[float] = []
        position = 0  # -1 short, 0 flat, 1 long
        entry_price: float | None = None
        trades = wins = losses = 0
        cooldown = 0
        peak = equity
        max_drawdown = 0.0

        for previous, current in zip(snapshots, snapshots[1:]):
            if previous.price == 0:
                price_return = 0.0
            else:
                price_return = (current.price - previous.price) / previous.price

            step_return = position * risk_fraction * price_return
            equity *= 1.0 + step_return
            returns.append(step_return)
            equity_curve.append(equity)

            if equity > peak:
                peak = equity
            if peak > 0:
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            if position != 0:
                should_exit = (
                    current.confidence < confidence_floor
                    or abs(current.strength) <= exit_threshold
                )
                if should_exit:
                    if entry_price not in (None, 0.0):
                        trade_return = position * (current.price - entry_price) / entry_price
                        if trade_return > 0:
                            wins += 1
                        elif trade_return < 0:
                            losses += 1
                    position = 0
                    entry_price = None
                    cooldown = cooldown_steps
                continue

            if cooldown > 0:
                cooldown -= 1
                continue

            if current.confidence < confidence_floor:
                continue
            if current.strength >= entry_threshold:
                position = 1
                entry_price = current.price
                trades += 1
            elif current.strength <= -entry_threshold:
                position = -1
                entry_price = current.price
                trades += 1

        if position != 0 and entry_price not in (None, 0.0):
            final_price = snapshots[-1].price
            trade_return = position * (final_price - entry_price) / entry_price
            if trade_return > 0:
                wins += 1
            elif trade_return < 0:
                losses += 1

        win_rate = wins / trades if trades else 0.0
        volatility = stdev(returns) if len(returns) >= 2 else 0.0
        avg_return = mean(returns) if returns else 0.0
        sharpe = avg_return / volatility if volatility > 0 else 0.0
        total_return = equity - 1.0

        return RecordedEvaluationResult(
            equity_curve=tuple(equity_curve),
            total_return=float(total_return),
            max_drawdown=float(max_drawdown),
            sharpe_ratio=float(sharpe),
            volatility=float(volatility),
            win_rate=float(win_rate),
            trades=trades,
            wins=wins,
            losses=losses,
        )

    def _extract_parameters(self, genome: object | Mapping[str, object]) -> dict[str, float]:
        if isinstance(genome, Mapping):
            return {
                str(key): _coerce_float(value, default=0.0)
                for key, value in genome.items()
            }
        params = getattr(genome, "parameters", None)
        if isinstance(params, Mapping):
            return {
                str(key): _coerce_float(value, default=0.0)
                for key, value in params.items()
            }
        # Fallback to attribute inspection
        result: dict[str, float] = {}
        for key in ("entry_threshold", "exit_threshold", "risk_fraction", "min_confidence", "cooldown_steps"):
            if hasattr(genome, key):
                result[key] = _coerce_float(getattr(genome, key), default=0.0)
        return result
