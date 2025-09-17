#!/usr/bin/env python3
"""
ECOSYSTEM-31: Portfolio-Level Evolution
======================================

Evolve entire strategy portfolios, not just individual strategies.
Implements correlation optimization, risk budgeting evolution, synergy detection,
and diversification maximization for robust portfolio management.

This module creates a sophisticated portfolio evolution system that
optimizes strategy combinations rather than individual strategies.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Protocol, Sequence, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class _ClusteringLike(Protocol):
    def fit_predict(self, X: "NDArray[np.float64]") -> "NDArray[np.int_]": ...


@dataclass
class PortfolioStrategy:
    """Represents a strategy within a portfolio."""

    strategy_id: str
    strategy_type: str
    weight: float
    expected_return: float
    expected_volatility: float
    correlation_vector: List[float]
    risk_contribution: float
    performance_metrics: Dict[str, float]


@dataclass
class PortfolioMetrics:
    """Represents portfolio-level metrics."""

    total_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: List[List[float]]
    diversification_ratio: float
    risk_budget_allocation: Dict[str, float]


@dataclass
class EvolutionResult:
    """Represents portfolio evolution results."""

    portfolio: List[PortfolioStrategy]
    metrics: PortfolioMetrics
    evolution_history: List[Dict[str, object]]
    recommendations: List[str]


class CorrelationOptimizer:
    """Optimizes correlation between strategies in portfolio."""

    def __init__(self) -> None:
        # Localize heavy sklearn imports to avoid import-time failures
        self.correlation_estimator: Optional[object] = None
        self.clustering: Optional[_ClusteringLike] = None
        try:
            from sklearn.covariance import LedoitWolf as _LedoitWolf

            self.correlation_estimator = _LedoitWolf()
        except Exception:  # pragma: no cover
            self.correlation_estimator = None

        try:
            from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering

            self.clustering = _AgglomerativeClustering(n_clusters=3)
        except Exception:  # pragma: no cover
            self.clustering = None  # Fallback clustering handled in method

    async def optimize_correlations(
        self, strategies: List[PortfolioStrategy], market_data: Mapping[str, float]
    ) -> List[PortfolioStrategy]:
        """Optimize strategy correlations for better diversification."""

        # Calculate correlation matrix
        correlation_matrix = await self._calculate_correlation_matrix(strategies, market_data)

        # Identify correlation clusters
        clusters = self._identify_correlation_clusters(correlation_matrix)

        # Optimize strategy selection based on clusters
        optimized_strategies: List[PortfolioStrategy] = self._optimize_strategy_selection(
            strategies, clusters
        )

        return optimized_strategies

    async def _calculate_correlation_matrix(
        self, strategies: List[PortfolioStrategy], market_data: Mapping[str, float]
    ) -> "NDArray[np.float64]":
        """Calculate correlation matrix between strategies."""

        n = len(strategies)
        correlation_matrix = np.eye(n)

        # Simulate correlation calculation
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate correlation based on strategy types and market conditions
                correlation = self._calculate_strategy_correlation(
                    strategies[i], strategies[j], market_data
                )
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation

        return correlation_matrix

    def _calculate_strategy_correlation(
        self,
        strategy1: PortfolioStrategy,
        strategy2: PortfolioStrategy,
        market_data: Mapping[str, float],
    ) -> float:
        """Calculate correlation between two strategies."""

        # Base correlation on strategy types
        type_correlations = {
            ("momentum", "momentum"): 0.8,
            ("momentum", "mean_reversion"): -0.3,
            ("momentum", "arbitrage"): 0.1,
            ("mean_reversion", "mean_reversion"): 0.7,
            ("mean_reversion", "arbitrage"): 0.2,
            ("arbitrage", "arbitrage"): 0.4,
        }

        base_correlation = type_correlations.get(
            (strategy1.strategy_type, strategy2.strategy_type), 0.0
        )

        # Adjust based on market conditions
        volatility = float(market_data.get("volatility", 0.02))
        adjustment = volatility * 0.1

        return float(max(-1.0, min(1.0, base_correlation + adjustment)))

    def _identify_correlation_clusters(
        self, correlation_matrix: "NDArray[np.float64]"
    ) -> List[int]:
        """Identify clusters of highly correlated strategies."""

        # Use hierarchical clustering when available; fall back to simple bucketing
        distance_matrix = 1 - np.abs(correlation_matrix)
        clustering = self.clustering
        if clustering is None:
            n = (
                int(getattr(distance_matrix, "shape", (0,))[0])
                if hasattr(distance_matrix, "shape")
                else 0
            )
            return [i % 3 for i in range(n)]
        clusters_arr = cast("NDArray[np.int_]", clustering.fit_predict(distance_matrix))
        to_list_fn = getattr(clusters_arr, "tolist", None)
        if callable(to_list_fn):
            raw = cast(Sequence[int], to_list_fn())
            return [int(x) for x in raw]
        return [int(x) for x in list(clusters_arr)]

    def _optimize_strategy_selection(
        self, strategies: List[PortfolioStrategy], clusters: List[int]
    ) -> List[PortfolioStrategy]:
        """Optimize strategy selection based on correlation clusters."""

        # Select strategies from different clusters for diversification
        cluster_counts: Dict[int, int] = {}
        for cluster in clusters:
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        # Prefer strategies from underrepresented clusters
        optimized_strategies: List[PortfolioStrategy] = []
        for i, strategy in enumerate(strategies):
            cluster_size = cluster_counts[clusters[i]]
            if cluster_size <= 2:  # Keep strategies from small clusters
                optimized_strategies.append(strategy)

        return optimized_strategies


class RiskBudgetingEvolution:
    """Evolves risk allocation across strategies."""

    def __init__(self) -> None:
        self.risk_models: Dict[str, float] = {}
        self.evolution_history: List[Dict[str, object]] = []

    async def evolve_risk_budgeting(
        self, portfolio: List[PortfolioStrategy], market_data: Mapping[str, float]
    ) -> Dict[str, float]:
        """Evolve risk allocation across strategies."""

        # Calculate current risk contributions
        current_contributions = self._calculate_risk_contributions(portfolio)

        # Optimize risk allocation
        optimized_allocation = await self._optimize_risk_allocation(
            portfolio, current_contributions, market_data
        )

        # Apply evolutionary pressure
        evolved_allocation = self._apply_evolutionary_pressure(optimized_allocation, market_data)

        # Update portfolio
        updated_portfolio = self._update_portfolio_risk(portfolio, evolved_allocation)

        return evolved_allocation

    def _calculate_risk_contributions(self, portfolio: List[PortfolioStrategy]) -> Dict[str, float]:
        """Calculate risk contributions of each strategy."""

        contributions: Dict[str, float] = {}
        total_risk = sum(s.expected_volatility * s.weight for s in portfolio)

        for strategy in portfolio:
            contribution = (strategy.expected_volatility * strategy.weight) / max(total_risk, 0.001)
            contributions[strategy.strategy_id] = contribution

        return contributions

    async def _optimize_risk_allocation(
        self,
        portfolio: List[PortfolioStrategy],
        current_contributions: Dict[str, float],
        market_data: Mapping[str, float],
    ) -> Dict[str, float]:
        """Optimize risk allocation using risk parity approach."""

        # Risk parity: equal risk contribution
        n_strategies = len(portfolio)
        target_contribution = 1.0 / n_strategies

        optimized_allocation: Dict[str, float] = {}

        for strategy in portfolio:
            current = current_contributions[strategy.strategy_id]
            adjustment = (target_contribution - current) * 0.5  # Gradual adjustment

            new_weight = strategy.weight * (1 + adjustment)
            optimized_allocation[strategy.strategy_id] = max(0.01, min(0.5, new_weight))

        # Normalize weights
        total_weight = sum(optimized_allocation.values())
        for strategy_id in optimized_allocation:
            optimized_allocation[strategy_id] /= total_weight

        return optimized_allocation

    def _apply_evolutionary_pressure(
        self, allocation: Dict[str, float], market_data: Mapping[str, float]
    ) -> Dict[str, float]:
        """Apply evolutionary pressure to risk allocation."""

        # Adjust based on market conditions
        volatility = float(market_data.get("volatility", 0.02))

        # Reduce risk in high volatility periods
        volatility_factor = 1.0 - (volatility - 0.02) * 2

        evolved_allocation: Dict[str, float] = {}
        for strategy_id, weight in allocation.items():
            evolved_allocation[strategy_id] = weight * volatility_factor

        # Normalize
        total_weight = sum(evolved_allocation.values())
        for strategy_id in evolved_allocation:
            evolved_allocation[strategy_id] /= total_weight

        return evolved_allocation

    def _update_portfolio_risk(
        self, portfolio: List[PortfolioStrategy], allocation: Dict[str, float]
    ) -> List[PortfolioStrategy]:
        """Update portfolio with new risk allocation."""

        updated_portfolio: List[PortfolioStrategy] = []

        for strategy in portfolio:
            updated_strategy = PortfolioStrategy(
                strategy_id=strategy.strategy_id,
                strategy_type=strategy.strategy_type,
                weight=allocation[strategy.strategy_id],
                expected_return=strategy.expected_return,
                expected_volatility=strategy.expected_volatility,
                correlation_vector=strategy.correlation_vector,
                risk_contribution=allocation[strategy.strategy_id] * strategy.expected_volatility,
                performance_metrics=strategy.performance_metrics,
            )
            updated_portfolio.append(updated_strategy)

        return updated_portfolio


class SynergyDetector:
    """Detects and amplifies positive interactions between strategies."""

    def __init__(self) -> None:
        self.synergy_model = self._build_synergy_model()
        self.interaction_history: List[Dict[str, object]] = []

    def _build_synergy_model(self) -> object:
        """Build neural network for synergy detection."""
        try:
            import torch.nn as nn

            return nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        except Exception:
            # Fallback lightweight stub when torch is unavailable at runtime
            class _StubModel:
                def __call__(self, *args: object, **kwargs: object) -> float:
                    return 0.5

            return _StubModel()

    async def detect_synergies(
        self, portfolio: List[PortfolioStrategy], market_data: Mapping[str, float]
    ) -> List[Dict[str, object]]:
        """Detect positive synergies between strategies."""

        synergies: List[Dict[str, object]] = []

        # Analyze all strategy pairs
        for i in range(len(portfolio)):
            for j in range(i + 1, len(portfolio)):
                synergy = await self._analyze_synergy(portfolio[i], portfolio[j], market_data)
                score_val = float(cast(float, synergy.get("synergy_score", 0.0)))
                if score_val > 0.7:
                    synergies.append(synergy)

        return synergies

    async def _analyze_synergy(
        self,
        strategy1: PortfolioStrategy,
        strategy2: PortfolioStrategy,
        market_data: Mapping[str, float],
    ) -> Dict[str, object]:
        """Analyze synergy between two strategies."""

        # Calculate synergy score
        synergy_score = self._calculate_synergy_score(strategy1, strategy2, market_data)

        # Identify synergy type
        synergy_type = self._identify_synergy_type(strategy1, strategy2)

        # Calculate amplification potential
        amplification = self._calculate_amplification_potential(synergy_score, synergy_type)

        return {
            "strategy_pair": (strategy1.strategy_id, strategy2.strategy_id),
            "synergy_score": synergy_score,
            "synergy_type": synergy_type,
            "amplification_potential": amplification,
            "recommendations": self._generate_synergy_recommendations(synergy_type),
        }

    def _calculate_synergy_score(
        self,
        strategy1: PortfolioStrategy,
        strategy2: PortfolioStrategy,
        market_data: Mapping[str, float],
    ) -> float:
        """Calculate synergy score between two strategies."""

        # Base score on complementary characteristics
        score = 0.0

        # Complementary returns
        if strategy1.expected_return * strategy2.expected_return < 0:
            score += 0.3

        # Low correlation
        correlation = self._get_correlation(strategy1, strategy2)
        if abs(correlation) < 0.3:
            score += 0.4

        # Complementary risk profiles
        risk_diff = abs(strategy1.expected_volatility - strategy2.expected_volatility)
        if risk_diff > 0.01:
            score += 0.3

        return min(1.0, score)

    def _identify_synergy_type(
        self, strategy1: PortfolioStrategy, strategy2: PortfolioStrategy
    ) -> str:
        """Identify the type of synergy."""

        # Determine synergy type based on strategy characteristics
        if strategy1.strategy_type == "momentum" and strategy2.strategy_type == "mean_reversion":
            return "regime_complementary"
        elif strategy1.strategy_type == strategy2.strategy_type:
            return "specialization_synergy"
        else:
            return "diversification_synergy"

    def _calculate_amplification_potential(self, synergy_score: float, synergy_type: str) -> float:
        """Calculate potential for synergy amplification."""

        type_multipliers = {
            "regime_complementary": 1.5,
            "specialization_synergy": 1.2,
            "diversification_synergy": 1.3,
        }

        multiplier = type_multipliers.get(synergy_type, 1.0)
        return min(2.0, synergy_score * multiplier)

    def _generate_synergy_recommendations(self, synergy_type: str) -> List[str]:
        """Generate recommendations for synergy amplification."""

        recommendations = {
            "regime_complementary": [
                "Increase allocation during regime transitions",
                "Use as hedging pair",
            ],
            "specialization_synergy": [
                "Combine signals for higher confidence",
                "Use for position sizing",
            ],
            "diversification_synergy": [
                "Maintain balanced allocation",
                "Monitor correlation stability",
            ],
        }

        return recommendations.get(synergy_type, ["Monitor performance"])

    def _get_correlation(self, strategy1: PortfolioStrategy, strategy2: PortfolioStrategy) -> float:
        """Get correlation between two strategies."""

        # Use stored correlation vector
        if len(strategy1.correlation_vector) > 0 and len(strategy2.correlation_vector) > 0:
            try:
                return float(
                    np.corrcoef(strategy1.correlation_vector, strategy2.correlation_vector)[0, 1]
                )
            except Exception:
                return 0.0

        return 0.0


class DiversificationMaximizer:
    """Maximizes diversification benefits across the portfolio."""

    def __init__(self) -> None:
        self.diversification_metrics: Dict[str, float] = {}
        self.optimization_history: List[Dict[str, object]] = []

    async def maximize_diversification(
        self, portfolio: List[PortfolioStrategy], market_data: Mapping[str, float]
    ) -> List[PortfolioStrategy]:
        """Maximize diversification benefits."""

        # Calculate current diversification
        current_diversification = await self._calculate_diversification_ratio(portfolio)

        # Optimize for maximum diversification
        optimized_portfolio = self._optimize_for_diversification(
            portfolio, current_diversification, market_data
        )

        # Validate diversification benefits
        validated_portfolio = self._validate_diversification(optimized_portfolio)

        return validated_portfolio

    def _optimize_for_diversification(
        self,
        portfolio: List[PortfolioStrategy],
        current_diversification: float,
        market_data: Mapping[str, float],
    ) -> List[PortfolioStrategy]:
        """Heuristic diversification optimizer: inverse-vol weighting with clamping and renormalization."""
        if not portfolio:
            return []
        weights = np.array([s.weight for s in portfolio], dtype=float)
        volatilities = np.array([s.expected_volatility for s in portfolio], dtype=float)
        # If any vol is non-positive, fall back to equal weights
        if np.any(volatilities <= 0):
            eq_w = 1.0 / float(len(portfolio))
            return [
                PortfolioStrategy(
                    strategy_id=s.strategy_id,
                    strategy_type=s.strategy_type,
                    weight=eq_w,
                    expected_return=s.expected_return,
                    expected_volatility=s.expected_volatility,
                    correlation_vector=s.correlation_vector,
                    risk_contribution=eq_w * s.expected_volatility,
                    performance_metrics=s.performance_metrics,
                )
                for s in portfolio
            ]
        inv_risk = 1.0 / volatilities
        new_w = inv_risk / float(np.sum(inv_risk))
        # Clamp to [0.01, 0.5] then renormalize
        new_w = np.clip(new_w, 0.01, 0.5)
        new_w = new_w / float(np.sum(new_w))
        updated: List[PortfolioStrategy] = []
        for idx, s in enumerate(portfolio):
            w = float(new_w[idx])
            updated.append(
                PortfolioStrategy(
                    strategy_id=s.strategy_id,
                    strategy_type=s.strategy_type,
                    weight=w,
                    expected_return=s.expected_return,
                    expected_volatility=s.expected_volatility,
                    correlation_vector=s.correlation_vector,
                    risk_contribution=w * s.expected_volatility,
                    performance_metrics=s.performance_metrics,
                )
            )
        return updated

    def _validate_diversification(
        self, portfolio: List[PortfolioStrategy]
    ) -> List[PortfolioStrategy]:
        """Validate diversification; if pathological, fall back to equal-weight portfolio."""
        n = len(portfolio)
        if n == 0:
            return []
        if n == 1:
            s = portfolio[0]
            w = 1.0
            return [
                PortfolioStrategy(
                    strategy_id=s.strategy_id,
                    strategy_type=s.strategy_type,
                    weight=w,
                    expected_return=s.expected_return,
                    expected_volatility=s.expected_volatility,
                    correlation_vector=s.correlation_vector,
                    risk_contribution=w * s.expected_volatility,
                    performance_metrics=s.performance_metrics,
                )
            ]
        weights = np.array([s.weight for s in portfolio], dtype=float)
        volatilities = np.array([s.expected_volatility for s in portfolio], dtype=float)
        correlation_matrix = self._build_correlation_matrix(portfolio)
        weighted_volatility = float(np.sum(weights * volatilities))
        portfolio_variance = float(
            weights @ (np.outer(volatilities, volatilities) * correlation_matrix) @ weights
        )
        portfolio_volatility = float(np.sqrt(portfolio_variance))
        ratio = weighted_volatility / max(portfolio_volatility, 1e-6)
        if not np.isfinite(ratio) or ratio < 1e-3:
            eq_w = 1.0 / float(n)
            return [
                PortfolioStrategy(
                    strategy_id=s.strategy_id,
                    strategy_type=s.strategy_type,
                    weight=eq_w,
                    expected_return=s.expected_return,
                    expected_volatility=s.expected_volatility,
                    correlation_vector=s.correlation_vector,
                    risk_contribution=eq_w * s.expected_volatility,
                    performance_metrics=s.performance_metrics,
                )
                for s in portfolio
            ]
        return portfolio

    async def _calculate_diversification_ratio(self, portfolio: List[PortfolioStrategy]) -> float:
        """Calculate diversification ratio for the portfolio."""

        if len(portfolio) <= 1:
            return 1.0

        # Calculate weighted average volatility
        weights = np.array([s.weight for s in portfolio])
        volatilities = np.array([s.expected_volatility for s in portfolio])

        weighted_volatility: float = float(np.sum(weights * volatilities))

        # Calculate portfolio volatility (simplified)
        correlation_matrix = self._build_correlation_matrix(portfolio)
        portfolio_variance = float(
            weights @ (np.outer(volatilities, volatilities) * correlation_matrix) @ weights
        )
        portfolio_volatility = float(np.sqrt(portfolio_variance))

        # Diversification ratio
        diversification_ratio = weighted_volatility / max(float(portfolio_volatility), 0.001)

        return float(diversification_ratio)

    def _build_correlation_matrix(
        self, portfolio: List[PortfolioStrategy]
    ) -> "NDArray[np.float64]":
        """Build correlation matrix from strategy data."""

        n = len(portfolio)
        # Initialize correlation matrix as identity matrix. We start with no correlation
        # between strategies (correlation of 1.0 on the diagonal). We'll attempt to fill
        # the off‑diagonal elements using stored correlation vectors if available.
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                corr = 0.0
                # If both strategies have correlation vectors of the same length, compute Pearson correlation
                vec_i = portfolio[i].correlation_vector
                vec_j = portfolio[j].correlation_vector
                if vec_i and vec_j and len(vec_i) == len(vec_j):
                    try:
                        corr = float(np.corrcoef(vec_i, vec_j)[0, 1])
                    except Exception:
                        corr = 0.0
                # Clamp the correlation to the valid range [-1, 1]
                corr = max(-1.0, min(1.0, corr))
                matrix[i, j] = corr
                matrix[j, i] = corr
        return matrix
