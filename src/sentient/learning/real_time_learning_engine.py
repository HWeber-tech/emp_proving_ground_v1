#!/usr/bin/env python3
"""
RealTimeLearningEngine - Epic 1: The Predator's Instinct
Generates learning signals from every closed trade for real-time adaptation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class LearningSignalType(Enum):
    """Types of learning signals."""

    PROFITABLE = "profitable"
    LOSS = "loss"
    BREAK_EVEN = "break_even"
    LIQUIDITY_GRAB = "liquidity_grab"
    STOP_CASCADE = "stop_cascade"


@dataclass
class LearningSignal:
    """A learning signal generated from a closed trade."""

    trade_id: str
    timestamp: datetime
    signal_type: LearningSignalType
    context: dict[str, Any]  # Market conditions at trade time
    outcome: dict[str, float]  # P&L, duration, etc.
    features: dict[str, float]  # Extracted features
    metadata: dict[str, Any]  # Additional context

    @property
    def pnl(self) -> float:
        """Get the P&L from this trade."""
        return self.outcome.get("pnl", 0.0)

    @property
    def duration(self) -> float:
        """Get the trade duration in seconds."""
        return self.outcome.get("duration", 0.0)

    @property
    def is_profitable(self) -> bool:
        """Check if this was a profitable trade."""
        return self.pnl > 0


class RealTimeLearningEngine:
    """Generates learning signals from closed trades in real-time."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.signals_buffer: list[LearningSignal] = []
        self.max_buffer_size = config.get("max_buffer_size", 1000)
        self.min_pnl_threshold = config.get("min_pnl_threshold", 0.0001)

    async def process_closed_trade(self, trade_data: dict[str, Any]) -> LearningSignal:
        """Process a closed trade and generate a learning signal."""
        logger.info(f"Processing closed trade: {trade_data.get('trade_id')}")

        # Extract trade information
        trade_id = trade_data["trade_id"]
        timestamp = datetime.fromisoformat(trade_data["close_time"])

        # Determine signal type based on outcome
        pnl = trade_data["pnl"]
        if pnl > self.min_pnl_threshold:
            signal_type = LearningSignalType.PROFITABLE
        elif pnl < -self.min_pnl_threshold:
            signal_type = LearningSignalType.LOSS
        else:
            signal_type = LearningSignalType.BREAK_EVEN

        # Detect special patterns
        if self._detect_liquidity_grab(trade_data):
            signal_type = LearningSignalType.LIQUIDITY_GRAB
        elif self._detect_stop_cascade(trade_data):
            signal_type = LearningSignalType.STOP_CASCADE

        # Extract context and features
        context = self._extract_context(trade_data)
        features = self._extract_features(trade_data)

        # Create learning signal
        signal = LearningSignal(
            trade_id=trade_id,
            timestamp=timestamp,
            signal_type=signal_type,
            context=context,
            outcome={
                "pnl": pnl,
                "duration": trade_data.get("duration", 0),
                "max_drawdown": trade_data.get("max_drawdown", 0),
                "max_profit": trade_data.get("max_profit", 0),
            },
            features=features,
            metadata={
                "strategy": trade_data.get("strategy", "unknown"),
                "market_condition": trade_data.get("market_condition", "neutral"),
                "volatility": trade_data.get("volatility", 0),
            },
        )

        # Add to buffer
        self.signals_buffer.append(signal)
        if len(self.signals_buffer) > self.max_buffer_size:
            self.signals_buffer.pop(0)

        logger.info(f"Generated learning signal: {signal_type.value} with P&L: {pnl}")
        return signal

    def _detect_liquidity_grab(self, trade_data: dict[str, Any]) -> bool:
        """Detect if this trade was part of a liquidity grab."""
        # Check for rapid price movement with high volume
        price_change = abs(trade_data.get("price_change", 0))
        volume = trade_data.get("volume", 0)
        avg_volume = trade_data.get("avg_volume", 1)

        return price_change > 0.001 and volume > 3 * avg_volume

    def _detect_stop_cascade(self, trade_data: dict[str, Any]) -> bool:
        """Detect if this trade was part of a stop cascade."""
        # Check for rapid consecutive losses in similar price levels
        recent_trades = trade_data.get("recent_trades", [])
        if len(recent_trades) < 3:
            return False

        losses = [t for t in recent_trades if t.get("pnl", 0) < -0.0005]
        return len(losses) >= 3

    def _extract_context(self, trade_data: dict[str, Any]) -> dict[str, Any]:
        """Extract market context at trade time."""
        return {
            "price": trade_data.get("entry_price", 0),
            "volume": trade_data.get("volume", 0),
            "spread": trade_data.get("spread", 0),
            "volatility": trade_data.get("volatility", 0),
            "order_imbalance": trade_data.get("order_imbalance", 0),
            "liquidity_depth": trade_data.get("liquidity_depth", 0),
        }

    def _extract_features(self, trade_data: dict[str, Any]) -> dict[str, float]:
        """Extract numerical features for pattern matching."""
        return {
            "price_momentum": trade_data.get("price_momentum", 0),
            "volume_ratio": trade_data.get("volume_ratio", 1),
            "volatility_ratio": trade_data.get("volatility_ratio", 1),
            "liquidity_ratio": trade_data.get("liquidity_ratio", 1),
            "order_flow_imbalance": trade_data.get("order_flow_imbalance", 0),
            "microstructure_score": trade_data.get("microstructure_score", 0),
        }

    def get_recent_signals(self, count: int = 100) -> list[LearningSignal]:
        """Get the most recent learning signals."""
        return self.signals_buffer[-count:]

    def get_signals_by_type(self, signal_type: LearningSignalType) -> list[LearningSignal]:
        """Get all signals of a specific type."""
        return [s for s in self.signals_buffer if s.signal_type == signal_type]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a summary of learning performance."""
        if not self.signals_buffer:
            return {"total_signals": 0}

        signals = self.signals_buffer
        profitable = [s for s in signals if s.is_profitable]

        return {
            "total_signals": len(signals),
            "profitable_signals": len(profitable),
            "win_rate": len(profitable) / len(signals),
            "avg_pnl": np.mean([s.pnl for s in signals]),
            "total_pnl": sum([s.pnl for s in signals]),
            "signal_types": {t.value: len(self.get_signals_by_type(t)) for t in LearningSignalType},
        }
