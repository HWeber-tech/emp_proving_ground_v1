"""Mean reversion strategy fulfilling roadmap Phase 2 commitments."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Mapping

import numpy as np

from src.core.strategy.engine import BaseStrategy
from src.risk.analytics.volatility_target import (
    calculate_realised_volatility,
    determine_target_allocation,
)
from src.trading.strategies.signals.ict_microstructure import (
    ICTMicrostructureAnalyzer,
)
from src.trading.strategies.capacity import (
    DEFAULT_L1_CAPACITY_RATIO,
    resolve_l1_depth_cap,
)

from .models import StrategySignal

__all__ = ["MeanReversionStrategyConfig", "MeanReversionStrategy"]

_EPSILON = 1e-12


def _extract_closes(market_data: Mapping[str, Any], symbol: str) -> np.ndarray:
    payload = market_data.get(symbol)
    if payload is None or "close" not in payload:
        raise ValueError(f"Missing close prices for symbol {symbol}")
    closes = np.asarray(payload["close"], dtype=float)
    if closes.size < 2:
        raise ValueError("At least two price points are required")
    return closes


def _returns(closes: np.ndarray) -> np.ndarray:
    return np.diff(closes) / closes[:-1]


@dataclass(slots=True)
class InventoryState:
    """Track inventory exposure and recent turnover for a symbol."""

    net_position: float = 0.0
    last_timestamp: datetime | None = None
    minute_turnover: Deque[tuple[datetime, float]] = field(default_factory=deque)
    hour_turnover: Deque[tuple[datetime, float]] = field(default_factory=deque)
    minute_total: float = 0.0
    hour_total: float = 0.0

    def pressure_fraction(self, now: datetime, half_life_minutes: float) -> float:
        """Return a [0, 1] weight expressing how hard inventory should mean revert."""

        if half_life_minutes <= 0.0:
            self.last_timestamp = now
            return 1.0
        if self.last_timestamp is None:
            self.last_timestamp = now
            return 0.0
        delta_seconds = max(0.0, (now - self.last_timestamp).total_seconds())
        self.last_timestamp = now
        if delta_seconds == 0.0:
            return 0.0
        half_life_seconds = max(1e-6, half_life_minutes * 60.0)
        return float(1.0 - math.exp(-delta_seconds / half_life_seconds))

    def _purge_queue(
        self,
        queue: Deque[tuple[datetime, float]],
        now: datetime,
        horizon_seconds: float,
        total_attr: str,
    ) -> None:
        total = getattr(self, total_attr)
        while queue and (now - queue[0][0]).total_seconds() >= horizon_seconds:
            _, amount = queue.popleft()
            total = max(0.0, total - amount)
        setattr(self, total_attr, total)

    def purge_turnover_windows(self, now: datetime) -> None:
        """Drop turnover observations that are outside their window."""

        self._purge_queue(self.minute_turnover, now, 60.0, "minute_total")
        self._purge_queue(self.hour_turnover, now, 3600.0, "hour_total")

    def record_turnover(self, amount: float, now: datetime) -> None:
        """Add executed turnover to the rolling windows."""

        if amount <= 0.0:
            return
        self.minute_turnover.append((now, amount))
        self.hour_turnover.append((now, amount))
        self.minute_total += amount
        self.hour_total += amount

    def apply_turnover_caps(
        self,
        desired_delta: float,
        *,
        now: datetime,
        minute_cap: float | None,
        hour_cap: float | None,
    ) -> tuple[float, dict[str, Any]]:
        """Clamp ``desired_delta`` so minute/hour turnover caps cannot be breached."""

        self.purge_turnover_windows(now)

        amount = abs(desired_delta)
        if amount <= 0.0:
            return 0.0, {
                "limited": False,
                "limited_by": [],
                "available_minute": (
                    max(0.0, minute_cap - self.minute_total)
                    if minute_cap is not None
                    else None
                ),
                "available_hour": (
                    max(0.0, hour_cap - self.hour_total)
                    if hour_cap is not None
                    else None
                ),
                "flatten_component": 0.0,
            }
        limited_by: list[str] = []
        prior_position = self.net_position
        requested_flatten = 0.0
        if prior_position != 0.0 and desired_delta != 0.0:
            if prior_position * desired_delta < 0.0:
                requested_flatten = min(abs(prior_position), amount)

        available_minute: float | None = None
        available_hour: float | None = None
        allowed_total = amount

        if minute_cap is not None:
            available_minute = max(0.0, minute_cap - self.minute_total)
            allowed_total = min(allowed_total, available_minute)
            if amount > available_minute + 1e-9:
                limited_by.append("minute")
        if hour_cap is not None:
            available_hour = max(0.0, hour_cap - self.hour_total)
            allowed_total = min(allowed_total, available_hour)
            if amount > available_hour + 1e-9:
                limited_by.append("hour")

        allowed_total = max(0.0, allowed_total)
        executed_abs = min(amount, allowed_total)

        flatten_executed = min(requested_flatten, executed_abs)

        executed = math.copysign(executed_abs, desired_delta) if executed_abs > 0.0 else 0.0
        if executed_abs > 0.0:
            self.record_turnover(executed_abs, now)

        turnover_meta = {
            "limited": bool(limited_by),
            "limited_by": limited_by,
            "available_minute": available_minute,
            "available_hour": available_hour,
            "flatten_component": flatten_executed if flatten_executed > 0.0 else 0.0,
        }
        return executed, turnover_meta


@dataclass(slots=True)
class MeanReversionStrategyConfig:
    """Configuration for the mean reversion strategy."""

    lookback: int = 30
    zscore_entry: float = 1.0
    target_volatility: float = 0.08
    max_leverage: float = 1.5
    annualisation_factor: float = math.sqrt(252.0)
    inventory_half_life_minutes: float = 5.0
    inventory_flat_threshold: float = 1e-6
    turnover_cap_per_minute: float | None = 0.05
    turnover_cap_per_hour: float | None = 0.25


class MeanReversionStrategy(BaseStrategy):
    """Bollinger-band inspired mean reversion implementation."""

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        *,
        capital: float,
        config: MeanReversionStrategyConfig | None = None,
        microstructure_analyzer: ICTMicrostructureAnalyzer | None = None,
    ) -> None:
        super().__init__(strategy_id=strategy_id, symbols=symbols)
        self._capital = float(capital)
        self._config = config or MeanReversionStrategyConfig()
        self._microstructure_analyzer = (
            microstructure_analyzer or ICTMicrostructureAnalyzer()
        )
        self._inventory: dict[str, InventoryState] = {}
        self._minute_turnover_cap = self._resolve_cap(self._config.turnover_cap_per_minute)
        self._hour_turnover_cap = self._resolve_cap(self._config.turnover_cap_per_hour)
        self._inventory_flat_threshold = max(0.0, float(self._config.inventory_flat_threshold))

    def _resolve_cap(self, cap_value: float | None) -> float | None:
        if cap_value is None:
            return None
        value = float(cap_value)
        if value <= 0.0:
            return 0.0
        if value <= 1.0:
            return float(self._capital * value)
        return value

    async def generate_signal(
        self, market_data: Mapping[str, Any], symbol: str
    ) -> StrategySignal:
        try:
            closes = _extract_closes(market_data, symbol)
            prices = closes[-self._config.lookback :]
            returns = _returns(closes)
            return_window = returns[-self._config.lookback :]
        except Exception as exc:
            return StrategySignal(
                symbol=symbol,
                action="FLAT",
                confidence=0.0,
                notional=0.0,
                metadata={"reason": "insufficient_data", "error": str(exc)},
            )

        mean_price = float(np.mean(prices))
        std_price = float(np.std(prices, ddof=1 if prices.size > 1 else 0))
        last_price = float(prices[-1])
        zscore = (last_price - mean_price) / max(std_price, _EPSILON)

        if zscore >= self._config.zscore_entry:
            action = "SELL"
        elif zscore <= -self._config.zscore_entry:
            action = "BUY"
        else:
            action = "FLAT"
        base_action = action

        confidence = 0.0
        target_notional = 0.0
        capacity_meta: dict[str, object] | None = None
        cap_limit, cap_details = resolve_l1_depth_cap(
            market_data, symbol, ratio=DEFAULT_L1_CAPACITY_RATIO
        )
        realised_vol = calculate_realised_volatility(
            return_window,
            annualisation_factor=self._config.annualisation_factor,
        )

        if action != "FLAT":
            ratio = abs(zscore) / max(self._config.zscore_entry, _EPSILON)
            confidence = float(min(ratio, 2.0) / 2.0)
            allocation = determine_target_allocation(
                capital=self._capital,
                target_volatility=self._config.target_volatility,
                realised_volatility=realised_vol,
                max_leverage=self._config.max_leverage,
                max_notional=cap_limit,
            )
            target_notional = allocation.target_notional
            if action == "SELL":
                target_notional *= -1.0
            if cap_details:
                raw_target = (
                    allocation.raw_target_notional
                    if allocation.raw_target_notional is not None
                    else allocation.target_notional
                )
                capacity_meta = {
                    **cap_details,
                    "raw_target_notional": raw_target,
                    "notional_after_cap": allocation.target_notional,
                    "cap_applied": bool(raw_target > allocation.target_notional + 1e-9),
                }

        metadata = {
            "mean_price": mean_price,
            "price_std": std_price,
            "last_price": last_price,
            "zscore": zscore,
            "target_volatility": self._config.target_volatility,
            "max_leverage": self._config.max_leverage,
            "realised_volatility": realised_vol,
        }

        if self._microstructure_analyzer is not None:
            features = await self._microstructure_analyzer.summarise(market_data, symbol)
            if features is not None:
                alignment_score, breakdown = features.alignment_assessment(action)
                microstructure_metadata = features.to_metadata()
                microstructure_metadata["alignment"] = {
                    "score": alignment_score,
                    "breakdown": breakdown,
                }
                metadata["microstructure"] = microstructure_metadata
                if action in {"BUY", "SELL"} and confidence > 0.0 and alignment_score != 0.0:
                    adjustment = 1.0 + alignment_score * features.confidence * 0.3
                    confidence = float(
                        max(0.0, min(1.0, confidence * adjustment))
                    )

        if capacity_meta is not None:
            metadata["liquidity_capacity"] = capacity_meta

        state = self._inventory.setdefault(symbol, InventoryState())
        now = datetime.now(timezone.utc)
        pressure_fraction = state.pressure_fraction(
            now, self._config.inventory_half_life_minutes
        )
        prior_position = state.net_position
        inventory_flat = abs(target_notional) <= self._inventory_flat_threshold

        if inventory_flat:
            desired_delta = -pressure_fraction * prior_position
            pressure_target = prior_position + desired_delta
        else:
            desired_delta = target_notional - prior_position
            pressure_target = target_notional

        executed_delta, turnover_meta = state.apply_turnover_caps(
            desired_delta,
            now=now,
            minute_cap=self._minute_turnover_cap,
            hour_cap=self._hour_turnover_cap,
        )

        if abs(desired_delta) > 1e-9:
            execution_ratio = min(
                1.0,
                abs(executed_delta) / max(abs(desired_delta), 1e-9),
            )
            if confidence <= 0.0:
                confidence = float(min(1.0, max(execution_ratio, pressure_fraction)))
            else:
                confidence = float(max(0.0, min(1.0, confidence * execution_ratio)))
        else:
            confidence = 0.0 if abs(executed_delta) <= 1e-9 else float(min(1.0, pressure_fraction))

        state.net_position = prior_position + executed_delta

        inventory_meta = {
            "prior_position": prior_position,
            "base_action": base_action,
            "base_target_position": target_notional,
            "pressure_target_position": pressure_target,
            "desired_delta": desired_delta,
            "executed_delta": executed_delta,
            "net_position": state.net_position,
            "pressure_fraction": pressure_fraction,
            "turnover": {
                "minute_cap": self._minute_turnover_cap,
                "minute_utilisation": state.minute_total,
                "hour_cap": self._hour_turnover_cap,
                "hour_utilisation": state.hour_total,
            },
            "turnover_limited": turnover_meta["limited"],
            "turnover_limited_by": turnover_meta["limited_by"],
            "available_turnover_minute": turnover_meta["available_minute"],
            "available_turnover_hour": turnover_meta["available_hour"],
        }

        if abs(executed_delta) <= 1e-9:
            action = "FLAT"
            executed_delta = 0.0
            confidence = 0.0
        else:
            action = "BUY" if executed_delta > 0 else "SELL"

        metadata["target_notional"] = target_notional
        metadata["base_action"] = base_action
        metadata["inventory_state"] = inventory_meta

        return StrategySignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            notional=executed_delta,
            metadata=metadata,
        )
