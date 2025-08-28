#!/usr/bin/env python3
"""
Coordination Engine
===================

Manages coordination between multiple predator strategies to prevent conflicts
and optimize portfolio-level performance.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import DefaultDict, Dict, List, TypedDict, cast

import numpy as np

from src.core.interfaces import CoordinationResult as CoordinationResultProto
from src.core.interfaces import (
    HasSpeciesType,
    ICoordinationEngine,
    MarketContext,
    TradeIntent,
)
from src.core.types import JSONObject

logger = logging.getLogger(__name__)


@dataclass
class CoordinationResultData:
    """Result of coordination process (implementation of CoordinationResult Protocol)."""

    approved_intents: List[TradeIntent]
    rejected_intents: List[TradeIntent]
    coordination_score: float
    portfolio_risk: float
    correlation_impact: float


class PositionEntry(TypedDict):
    intent: TradeIntent
    timestamp: datetime
    status: str


class CoordinationHistoryEntry(TypedDict):
    timestamp: datetime
    result: CoordinationResultData
    context: MarketContext


class CoordinationEngine(ICoordinationEngine):
    """Advanced coordination engine for predator strategies."""

    def __init__(self) -> None:
        self.active_positions: DefaultDict[str, List[PositionEntry]] = defaultdict(list)
        self.coordination_history: List[CoordinationHistoryEntry] = []
        self.risk_limits: Dict[str, float] = {
            "max_portfolio_risk": 0.15,
            "max_correlation": 0.7,
            "max_single_species_risk": 0.05,
            "max_daily_trades": 50.0,
        }

    async def resolve_intents(
        self, intents: List[TradeIntent], market_context: MarketContext
    ) -> CoordinationResultProto:
        """Resolve conflicting intents from multiple strategies."""
        if not intents:
            return CoordinationResultData([], [], 0.0, 0.0, 0.0)

        # Group intents by symbol
        symbol_groups: DefaultDict[str, List[TradeIntent]] = defaultdict(list)
        for intent in intents:
            symbol_groups[intent.symbol].append(intent)

        # Resolve conflicts within each symbol
        approved_intents = []
        rejected_intents = []

        for symbol, symbol_intents in symbol_groups.items():
            resolved = await self._resolve_symbol_conflicts(symbol_intents, market_context)
            approved_intents.extend(resolved["approved"])
            rejected_intents.extend(resolved["rejected"])

        # Calculate coordination metrics
        coordination_score = self._calculate_coordination_score(approved_intents)
        portfolio_risk = self._calculate_portfolio_risk(approved_intents)
        correlation_impact = self._calculate_correlation_impact(approved_intents)

        # Update active positions
        self._update_active_positions(approved_intents)

        result = CoordinationResultData(
            approved_intents=approved_intents,
            rejected_intents=rejected_intents,
            coordination_score=coordination_score,
            portfolio_risk=portfolio_risk,
            correlation_impact=correlation_impact,
        )

        self.coordination_history.append(
            {"timestamp": datetime.now(), "result": result, "context": market_context}
        )

        return result

    async def _resolve_symbol_conflicts(
        self, intents: List[TradeIntent], market_context: MarketContext
    ) -> Dict[str, List[TradeIntent]]:
        """Resolve conflicts for a specific symbol."""
        if len(intents) <= 1:
            return {"approved": intents, "rejected": []}

        # Score each intent
        scored_intents = []
        for intent in intents:
            score = await self._score_intent(intent, market_context)
            scored_intents.append((score, intent))

        # Sort by score (highest first)
        scored_intents.sort(key=lambda x: x[0], reverse=True)

        # Apply coordination rules
        approved = []
        rejected = []
        total_risk: float = 0.0

        for score, intent in scored_intents:
            intent_risk = intent.size * intent.confidence

            # Check risk limits
            if total_risk + intent_risk <= self.risk_limits["max_single_species_risk"]:
                approved.append(intent)
                total_risk += intent_risk
            else:
                rejected.append(intent)

        return {"approved": approved, "rejected": rejected}

    async def _score_intent(self, intent: TradeIntent, market_context: MarketContext) -> float:
        """Score a trading intent based on multiple factors."""
        score = 0.0

        # Confidence score
        score += intent.confidence * 0.3

        # Species suitability score
        species_score = await self._get_species_suitability(intent.species_type, market_context)
        score += species_score * 0.25

        # Risk-adjusted return potential
        risk_adjusted = intent.confidence / (intent.size * 0.01)  # Simple risk adjustment
        score += min(risk_adjusted, 1.0) * 0.2

        # Priority bonus
        score += (intent.priority / 10.0) * 0.15

        # Time decay (recent intents get slight bonus)
        time_bonus = max(0.0, 1 - (datetime.now() - intent.timestamp).seconds / 3600)
        score += time_bonus * 0.1

        return float(score)

    async def _get_species_suitability(
        self, species_type: str, market_context: MarketContext
    ) -> float:
        """Get suitability score for species in current market context."""
        suitability_map = {
            "stalker": {
                "trending_bull": 0.9,
                "trending_bear": 0.8,
                "ranging": 0.3,
                "volatile": 0.5,
                "quiet": 0.4,
            },
            "ambusher": {
                "trending_bull": 0.4,
                "trending_bear": 0.4,
                "ranging": 0.9,
                "volatile": 0.7,
                "quiet": 0.8,
            },
            "pack_hunter": {
                "trending_bull": 0.8,
                "trending_bear": 0.8,
                "ranging": 0.7,
                "volatile": 0.8,
                "quiet": 0.6,
            },
            "scavenger": {
                "trending_bull": 0.5,
                "trending_bear": 0.5,
                "ranging": 0.8,
                "volatile": 0.9,
                "quiet": 0.7,
            },
            "alpha": {
                "trending_bull": 0.9,
                "trending_bear": 0.7,
                "ranging": 0.4,
                "volatile": 0.8,
                "quiet": 0.5,
            },
        }

        return suitability_map.get(species_type, {}).get(market_context.regime, 0.5)

    async def prioritize_strategies(
        self, strategies: List[HasSpeciesType], regime: str
    ) -> List[HasSpeciesType]:
        """Prioritize strategies based on current market regime."""
        if not strategies:
            return []

        # Get species priorities for this regime
        priority_map = {
            "trending_bull": ["alpha", "stalker", "pack_hunter", "scavenger", "ambusher"],
            "trending_bear": ["stalker", "pack_hunter", "scavenger", "alpha", "ambusher"],
            "ranging": ["ambusher", "scavenger", "pack_hunter", "alpha", "stalker"],
            "volatile": ["scavenger", "alpha", "pack_hunter", "ambusher", "stalker"],
            "quiet": ["ambusher", "scavenger", "pack_hunter", "stalker", "alpha"],
        }

        species_order = priority_map.get(regime, ["pack_hunter"] * 5)

        # Sort strategies by species priority
        def get_priority(strategy: HasSpeciesType) -> int:
            species = getattr(strategy, "species_type", "generic")
            try:
                return species_order.index(species)
            except ValueError:
                return 999

        return sorted(strategies, key=get_priority)

    def _calculate_coordination_score(self, approved_intents: List[TradeIntent]) -> float:
        """Calculate how well the approved intents are coordinated."""
        if not approved_intents:
            return 0.0

        # Check for conflicting directions
        directions: DefaultDict[str, int] = defaultdict(int)
        for intent in approved_intents:
            directions[intent.symbol + intent.direction] += 1

        # Calculate coordination score
        max_conflicts = max(directions.values()) if directions else 1
        coordination_score = 1.0 - (max_conflicts - 1) / max(max_conflicts, 1)

        return coordination_score

    def _calculate_portfolio_risk(self, approved_intents: List[TradeIntent]) -> float:
        """Calculate total portfolio risk from approved intents."""
        total_risk = 0.0

        for intent in approved_intents:
            # Simple risk calculation: size * confidence
            risk = intent.size * intent.confidence
            total_risk += risk

        return min(total_risk, 1.0)

    def _calculate_correlation_impact(self, approved_intents: List[TradeIntent]) -> float:
        """Calculate correlation impact of approved intents."""
        if len(approved_intents) < 2:
            return 0.0

        # Group by species
        species_groups: DefaultDict[str, List[TradeIntent]] = defaultdict(list)
        for intent in approved_intents:
            species_groups[intent.species_type].append(intent)

        # Calculate correlation risk
        correlation_risk = 0.0
        for species, intents in species_groups.items():
            if len(intents) > 1:
                # Risk increases with multiple intents from same species
                correlation_risk += len(intents) * 0.1

        return min(correlation_risk, 1.0)

    def _update_active_positions(self, approved_intents: List[TradeIntent]) -> None:
        """Update active positions with new intents."""
        for intent in approved_intents:
            self.active_positions[intent.symbol].append(
                {"intent": intent, "timestamp": datetime.now(), "status": "active"}
            )

        # Clean up old positions
        cutoff_time = datetime.now() - timedelta(hours=24)
        for symbol, positions in self.active_positions.items():
            self.active_positions[symbol] = [
                pos for pos in positions if pos["timestamp"] > cutoff_time
            ]

    async def get_portfolio_summary(self) -> dict[str, object]:
        """Get summary of current portfolio state."""
        total_positions = sum(len(positions) for positions in self.active_positions.values())

        species_distribution: DefaultDict[str, int] = defaultdict(int)
        for positions in self.active_positions.values():
            for pos in positions:
                species_distribution[pos["intent"].species_type] += 1

        return cast(
            dict[str, object],
            {
                "total_active_positions": total_positions,
                "species_distribution": dict(species_distribution),
                "symbols_traded": list(self.active_positions.keys()),
                "coordination_history_length": len(self.coordination_history),
            },
        )

    async def get_coordination_metrics(self) -> Dict[str, float]:
        """Get coordination performance metrics."""
        if not self.coordination_history:
            return {
                "avg_coordination_score": 0.0,
                "avg_portfolio_risk": 0.0,
                "avg_correlation_impact": 0.0,
                "total_resolutions": 0.0,
            }

        recent_history = self.coordination_history[-100:]
        scores = [r["result"].coordination_score for r in recent_history]
        risks = [r["result"].portfolio_risk for r in recent_history]
        correlations = [r["result"].correlation_impact for r in recent_history]

        return {
            "avg_coordination_score": float(np.mean(scores)),
            "avg_portfolio_risk": float(np.mean(risks)),
            "avg_correlation_impact": float(np.mean(correlations)),
            "total_resolutions": float(len(self.coordination_history)),
        }


# Example usage
async def test_coordination_engine() -> None:
    """Test the coordination engine."""
    engine = CoordinationEngine()

    # Create test intents
    test_intents = [
        TradeIntent(
            strategy_id="test_1",
            species_type="stalker",
            symbol="EURUSD",
            direction="BUY",
            confidence=0.8,
            size=0.05,
            priority=8,
            timestamp=datetime.now(),
        ),
        TradeIntent(
            strategy_id="test_2",
            species_type="ambusher",
            symbol="EURUSD",
            direction="SELL",
            confidence=0.7,
            size=0.03,
            priority=7,
            timestamp=datetime.now(),
        ),
        TradeIntent(
            strategy_id="test_3",
            species_type="pack_hunter",
            symbol="GBPUSD",
            direction="BUY",
            confidence=0.9,
            size=0.04,
            priority=9,
            timestamp=datetime.now(),
        ),
    ]

    market_context = MarketContext(
        symbol="EURUSD",
        regime="trending_bull",
        volatility=0.02,
        trend_strength=0.8,
        volume_anomaly=1.2,
    )

    result = await engine.resolve_intents(test_intents, market_context)

    print(f"Coordination Result:")
    print(f"  Approved: {len(result.approved_intents)}")
    print(f"  Rejected: {len(result.rejected_intents)}")
    print(f"  Score: {result.coordination_score:.2f}")
    print(f"  Risk: {result.portfolio_risk:.2f}")
    print(f"  Correlation: {result.correlation_impact:.2f}")


if __name__ == "__main__":
    asyncio.run(test_coordination_engine())
