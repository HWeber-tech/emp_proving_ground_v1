"""
Evolution Engine Module - v2.0 Implementation

This module implements the innovative anti-fragility fitness system as specified in v2.0,
providing the evolutionary pressure that drives organisms toward resilience and adaptability.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, getcontext
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data import TickDataStorage

# Configure decimal precision for financial calculations
getcontext().prec = 12
getcontext().rounding = ROUND_HALF_UP

logger = logging.getLogger(__name__)


@dataclass
class DecisionGenome:
    """
    Evolvable decision tree representing a trading strategy.

    This class implements the genetic programming approach from the original unified file,
    allowing for the emergence of complex, interpretable trading logic.
    """

    genome_id: str
    decision_tree: Dict[str, Any]
    fitness_score: float = 0.0
    robustness_score: float = 0.0
    adaptability_score: float = 0.0
    efficiency_score: float = 0.0
    antifragility_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_count: int = 0
    crossover_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def evaluate(self, market_data: pd.DataFrame, sensory_cortex) -> Dict[str, Any]:
        """
        Evaluate the genome against market data.

        Args:
            market_data: Historical market data
            sensory_cortex: Sensory cortex for market perception

        Returns:
            Evaluation results
        """
        results = {
            "trades": [],
            "equity_curve": [],
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_consecutive_losses": 0,
            "volatility": 0.0,
            "var_95": 0.0,
            "underwater_periods": 0,
            "recovery_time": 0.0,
        }

        if market_data.empty:
            return results

        # Initialize simulation state
        equity = 100000.0  # Starting capital
        position = 0
        entry_price = 0.0
        entry_time = None
        equity_curve = [equity]

        # Track performance metrics
        trades = []
        returns = []
        underwater_periods = []
        peak_equity = equity

        for i, (timestamp, row) in enumerate(market_data.iterrows()):
            # Get sensory reading
            sensory_reading = self._get_sensory_reading(sensory_cortex, row)

            # Evaluate decision tree
            decision = self._evaluate_decision_tree(sensory_reading)

            # Execute decision
            if decision == "BUY" and position <= 0:
                if position < 0:
                    # Close short position
                    pnl = (entry_price - row["close"]) * abs(position)
                    equity += pnl
                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": timestamp,
                            "entry_price": entry_price,
                            "exit_price": row["close"],
                            "quantity": position,
                            "pnl": pnl,
                            "type": "SHORT_CLOSE",
                        }
                    )

                # Open long position
                position = 10000  # Fixed position size for simplicity
                entry_price = row["close"]
                entry_time = timestamp

            elif decision == "SELL" and position >= 0:
                if position > 0:
                    # Close long position
                    pnl = (row["close"] - entry_price) * position
                    equity += pnl
                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": timestamp,
                            "entry_price": entry_price,
                            "exit_price": row["close"],
                            "quantity": position,
                            "pnl": pnl,
                            "type": "LONG_CLOSE",
                        }
                    )

                # Open short position
                position = -10000  # Fixed position size for simplicity
                entry_price = row["close"]
                entry_time = timestamp

            # Update equity curve
            current_equity = float(equity)
            if position != 0:
                if position > 0:
                    current_equity += float((row["close"] - entry_price) * position)
                else:
                    current_equity += float(
                        (entry_price - row["close"]) * abs(position)
                    )

            equity_curve.append(current_equity)

            # Track underwater periods
            if current_equity < peak_equity:
                underwater_periods.append(1)
            else:
                underwater_periods.append(0)
                peak_equity = current_equity

            # Calculate returns
            if i > 0:
                returns.append((current_equity - equity_curve[-2]) / equity_curve[-2])

        # Close any open position at the end
        if position != 0:
            if position > 0:
                pnl = (row["close"] - entry_price) * position
                equity += pnl
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": row["close"],
                        "quantity": position,
                        "pnl": pnl,
                        "type": "LONG_CLOSE",
                    }
                )
            else:
                pnl = (entry_price - row["close"]) * abs(position)
                equity += pnl
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": row["close"],
                        "quantity": position,
                        "pnl": pnl,
                        "type": "SHORT_CLOSE",
                    }
                )

        # Calculate performance metrics
        results["trades"] = trades
        results["equity_curve"] = equity_curve

        if len(returns) > 0:
            returns_array = np.array(returns)

            # Basic metrics
            results["total_return"] = (
                equity_curve[-1] - equity_curve[0]
            ) / equity_curve[0]
            results["volatility"] = returns_array.std() * np.sqrt(252)  # Annualized
            results["sharpe_ratio"] = (
                (returns_array.mean() * 252) / (returns_array.std() * np.sqrt(252))
                if returns_array.std() > 0
                else 0
            )

            # Sortino ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                results["sortino_ratio"] = (
                    (returns_array.mean() * 252) / downside_deviation
                    if downside_deviation > 0
                    else 0
                )

            # Calmar ratio
            max_dd = self._calculate_max_drawdown(equity_curve)
            results["max_drawdown"] = max_dd
            results["calmar_ratio"] = (
                results["total_return"] / max_dd if max_dd > 0 else 0
            )

            # Profit factor
            winning_trades = [t for t in trades if t["pnl"] > 0]
            losing_trades = [t for t in trades if t["pnl"] < 0]

            total_wins = sum(t["pnl"] for t in winning_trades)
            total_losses = abs(sum(t["pnl"] for t in losing_trades))
            # Cap profit factor to avoid infinite values in evolution
            if total_losses > 0:
                results["profit_factor"] = min(total_wins / total_losses, 10.0)
            else:
                results["profit_factor"] = total_wins / 1e-6 if total_wins > 0 else 0.0

            # Win rate and average trade
            results["win_rate"] = len(winning_trades) / len(trades) if trades else 0
            results["avg_win"] = (
                total_wins / len(winning_trades) if winning_trades else 0
            )
            results["avg_loss"] = (
                total_losses / len(losing_trades) if losing_trades else 0
            )

            # Value at Risk (negative value for loss)
            results["var_95"] = -np.percentile(returns_array, 5)

            # Recovery time
            results["underwater_periods"] = sum(underwater_periods)
            results["recovery_time"] = self._calculate_recovery_time(equity_curve)

        return results

    def _evaluate_decision_tree(self, sensory_reading) -> str:
        """
        Evaluate the decision tree against a sensory reading.

        Args:
            sensory_reading: Current market perception

        Returns:
            Decision: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            # Use genome parameters for evolvable decision making
            params = self.decision_tree["parameters"]
            buy_threshold = params.get("buy_threshold", 0.5)
            sell_threshold = params.get("sell_threshold", 0.5)
            momentum_weight = params.get("momentum_weight", 0.5)
            trend_weight = params.get("trend_weight", 0.5)
            institutional_weight = params.get("institutional_weight", 0.5)

            # Calculate weighted scores for buy/sell decisions
            buy_score = 0.0
            sell_score = 0.0

            # Macro trend contribution
            if hasattr(sensory_reading, "macro_trend") and hasattr(
                sensory_reading, "macro_strength"
            ):
                if sensory_reading.macro_trend == "BULLISH":
                    buy_score += trend_weight * sensory_reading.macro_strength
                elif sensory_reading.macro_trend == "BEARISH":
                    sell_score += trend_weight * sensory_reading.macro_strength

            # Institutional flow contribution
            if hasattr(sensory_reading, "institutional_flow") and hasattr(
                sensory_reading, "institutional_confidence"
            ):
                if sensory_reading.institutional_flow > 0:
                    buy_score += (
                        institutional_weight
                        * sensory_reading.institutional_flow
                        * sensory_reading.institutional_confidence
                    )
                else:
                    sell_score += (
                        institutional_weight
                        * abs(sensory_reading.institutional_flow)
                        * sensory_reading.institutional_confidence
                    )

            # Momentum contribution
            if hasattr(sensory_reading, "momentum_score") and hasattr(
                sensory_reading, "overall_sentiment"
            ):
                if sensory_reading.momentum_score > 0:
                    buy_score += (
                        momentum_weight
                        * sensory_reading.momentum_score
                        * sensory_reading.overall_sentiment
                    )
                else:
                    sell_score += (
                        momentum_weight
                        * abs(sensory_reading.momentum_score)
                        * (1 - sensory_reading.overall_sentiment)
                    )

            # Technical signal validation
            technical_buy = False
            technical_sell = False
            if hasattr(sensory_reading, "technical_signal"):
                if sensory_reading.technical_signal == "BUY":
                    technical_buy = True
                elif sensory_reading.technical_signal == "SELL":
                    technical_sell = True

            # Manipulation check
            manipulation_safe = True
            if hasattr(sensory_reading, "manipulation_probability"):
                manipulation_safe = sensory_reading.manipulation_probability < 0.3

            # Make decision based on thresholds
            if buy_score > buy_threshold and technical_buy and manipulation_safe:
                return "BUY"
            elif sell_score > sell_threshold and technical_sell and manipulation_safe:
                return "SELL"

            return "HOLD"

        except Exception as e:
            logger.error(f"Error evaluating decision tree: {e}")
            return "HOLD"

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_recovery_time(self, equity_curve: List[float]) -> float:
        """Calculate average recovery time from drawdowns."""
        if len(equity_curve) < 2:
            return 0.0

        recovery_times = []
        peak = equity_curve[0]
        underwater_start = None

        for i, equity in enumerate(equity_curve):
            if equity > peak:
                if underwater_start is not None:
                    recovery_times.append(i - underwater_start)
                    underwater_start = None
                peak = equity
            elif underwater_start is None and equity < peak:
                underwater_start = i

        return float(np.mean(recovery_times)) if recovery_times else 0.0

    def _get_sensory_reading(self, sensory_cortex, row):
        """
        Get sensory reading from the sensory cortex.

        Args:
            sensory_cortex: Sensory cortex instance
            row: Market data row

        Returns:
            Sensory reading object with expected attributes
        """
        try:
            # Create a simple sensory reading object with expected attributes
            # This is a compatibility layer for the evolution system

            class SimpleSensoryReading:
                def __init__(self):
                    # Default values
                    self.macro_trend = "NEUTRAL"
                    self.macro_strength = 0.5
                    self.technical_signal = "HOLD"
                    self.technical_strength = 0.5
                    self.institutional_flow = 0.0
                    self.institutional_confidence = 0.5
                    self.momentum_score = 0.0
                    self.overall_sentiment = 0.5
                    self.manipulation_probability = 0.1

            # Create basic reading
            reading = SimpleSensoryReading()

            # Try to get more sophisticated reading if available
            if hasattr(sensory_cortex, "update"):
                try:
                    # Convert row to MarketData format if needed
                    from src.sensory.core.base import MarketData

                    market_data = MarketData(
                        symbol="EURUSD",
                        timestamp=row.name if hasattr(row, "name") else datetime.now(),
                        open=row.get("open", row.get("close", 1.1000)),
                        high=row.get("high", row.get("close", 1.1000)),
                        low=row.get("low", row.get("close", 1.1000)),
                        close=row.get("close", 1.1000),
                        volume=row.get("volume", 1000),
                        bid=row.get("bid", row.get("low", row.get("close", 1.1000))),
                        ask=row.get("ask", row.get("high", row.get("close", 1.1002))),
                        source="synthetic",
                        latency_ms=0.0,
                    )

                    # Try to get synthesis result (this would be async in real usage)
                    # For now, create a simple reading based on price movement
                    if "close" in row:
                        # Simple trend detection
                        if hasattr(self, "_prev_close"):
                            change = row["close"] - self._prev_close
                            if change > 0.0001:
                                reading.macro_trend = "BULLISH"
                                reading.macro_strength = min(0.8, abs(change) * 1000)
                                reading.technical_signal = "BUY"
                            elif change < -0.0001:
                                reading.macro_trend = "BEARISH"
                                reading.macro_strength = min(0.8, abs(change) * 1000)
                                reading.technical_signal = "SELL"
                            else:
                                reading.macro_trend = "NEUTRAL"
                                reading.macro_strength = 0.3
                                reading.technical_signal = "HOLD"

                            # Simple momentum
                            reading.momentum_score = (
                                change * 1000
                            )  # Scale for reasonable range

                            # Simple sentiment (based on volume and price movement)
                            if "volume" in row:
                                volume_factor = min(1.0, row["volume"] / 5000)
                                reading.overall_sentiment = (
                                    0.5 + (reading.momentum_score * 0.3) * volume_factor
                                )
                            else:
                                reading.overall_sentiment = 0.5 + (
                                    reading.momentum_score * 0.3
                                )

                        self._prev_close = row["close"]

                except Exception as e:
                    logger.debug(f"Could not get sophisticated sensory reading: {e}")
                    # Fall back to simple reading

            return reading

        except Exception as e:
            logger.error(f"Error getting sensory reading: {e}")
            # Return a default reading

            class DefaultReading:
                def __init__(self):
                    self.macro_trend = "NEUTRAL"
                    self.macro_strength = 0.5
                    self.technical_signal = "HOLD"
                    self.technical_strength = 0.5
                    self.institutional_flow = 0.0
                    self.institutional_confidence = 0.5
                    self.momentum_score = 0.0
                    self.overall_sentiment = 0.5
                    self.manipulation_probability = 0.1

            return DefaultReading()


@dataclass
class EvolutionConfig:
    """Configuration for the evolution engine."""

    population_size: int = 100
    elite_ratio: float = 0.1
    crossover_ratio: float = 0.6
    mutation_ratio: float = 0.3
    tournament_size: int = 3
    max_generations: int = 50
    convergence_threshold: float = 0.001
    stagnation_limit: int = 10


@dataclass
class EvolutionStats:
    """Statistics for evolution progress."""

    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    diversity: float
    convergence_rate: float
    best_genome_id: str
    population_size: int


class FitnessEvaluator:
    """
    Anti-fragility fitness evaluator with multi-objective optimization.

    This class implements the innovative fitness function from the original unified file,
    evaluating organisms across returns, robustness, adaptability, efficiency, and anti-fragility.
    """

    def __init__(self, data_storage: TickDataStorage, instrument: str = "EURUSD"):
        """
        Initialize the fitness evaluator.

        Args:
            data_storage: Data storage for market data access
            instrument: Trading instrument symbol
        """
        self.data_storage = data_storage
        self.instrument = instrument

        # Market regime datasets
        self.regime_datasets: Dict[str, Optional[pd.DataFrame]] = {
            "TRENDING": None,
            "RANGING": None,
            "VOLATILE": None,
        }

        # Fitness weights
        self.weights = {
            "returns": 0.3,
            "robustness": 0.25,
            "adaptability": 0.2,
            "efficiency": 0.15,
            "antifragility": 0.1,
        }

        logger.info(
            f"FitnessEvaluator initialized with anti-fragility metrics for {instrument}"
        )

    def evaluate_genome(
        self, genome: DecisionGenome, sensory_cortex
    ) -> Dict[str, float]:
        """
        Evaluate a genome across multiple dimensions.

        Args:
            genome: Genome to evaluate
            sensory_cortex: Sensory cortex for market perception

        Returns:
            Fitness scores for each dimension
        """
        scores = {
            "returns": 0.0,
            "robustness": 0.0,
            "adaptability": 0.0,
            "efficiency": 0.0,
            "antifragility": 0.0,
        }

        try:
            # Load market regime datasets
            self._load_regime_datasets()

            # Evaluate across different market regimes
            regime_results = {}
            for regime_name, dataset in self.regime_datasets.items():
                if dataset is not None and not dataset.empty:
                    try:
                        results = genome.evaluate(dataset, sensory_cortex)
                        regime_results[regime_name] = results
                    except Exception as e:
                        logger.warning(f"Error evaluating {regime_name} regime: {e}")
                        continue

            if not regime_results:
                logger.warning(
                    "No regime data available for evaluation - using synthetic data"
                )
                # Create synthetic data for evaluation to allow evolution to proceed
                synthetic_data = self._create_synthetic_market_data()
                if synthetic_data is not None:
                    results = genome.evaluate(synthetic_data, sensory_cortex)
                    regime_results["SYNTHETIC"] = results
                else:
                    return scores

            # Calculate dimension scores
            scores["returns"] = self._calculate_returns_score(regime_results)
            scores["robustness"] = self._calculate_robustness_score(regime_results)
            scores["adaptability"] = self._calculate_adaptability_score(regime_results)
            scores["efficiency"] = self._calculate_efficiency_score(regime_results)
            scores["antifragility"] = self._calculate_antifragility_score(
                regime_results
            )

            # Update genome scores
            genome.fitness_score = self._calculate_overall_fitness(scores)
            genome.robustness_score = scores["robustness"]
            genome.adaptability_score = scores["adaptability"]
            genome.efficiency_score = scores["efficiency"]
            genome.antifragility_score = scores["antifragility"]

            logger.debug(
                f"Genome {genome.genome_id} evaluated - Overall: {genome.fitness_score:.4f}"
            )

        except Exception as e:
            logger.error(f"Error evaluating genome {genome.genome_id}: {e}")

        return scores

    def _load_regime_datasets(self):
        """Load market regime datasets for evaluation."""
        if all(dataset is not None for dataset in self.regime_datasets.values()):
            return  # Already loaded

        try:
            # Load different market regimes
            # In a real implementation, these would be pre-identified regime periods

            # Trending market (strong directional movement)
            trending_start = datetime(2023, 1, 1)
            trending_end = datetime(2023, 3, 31)
            trending_data = self.data_storage.get_data_range(
                self.instrument, trending_start, trending_end
            )
            if trending_data is not None and not trending_data.empty:
                # Ensure required columns exist
                required_columns = ["close", "open", "high", "low", "volume"]
                if all(col in trending_data.columns for col in required_columns):
                    self.regime_datasets["TRENDING"] = trending_data
                else:
                    logger.warning(
                        f"TRENDING data missing required columns. Available: {list(trending_data.columns)}"
                    )
                    self.regime_datasets["TRENDING"] = None
            else:
                self.regime_datasets["TRENDING"] = None

            # Ranging market (sideways movement)
            ranging_start = datetime(2023, 4, 1)
            ranging_end = datetime(2023, 6, 30)
            ranging_data = self.data_storage.get_data_range(
                self.instrument, ranging_start, ranging_end
            )
            if ranging_data is not None and not ranging_data.empty:
                if all(col in ranging_data.columns for col in required_columns):
                    self.regime_datasets["RANGING"] = ranging_data
                else:
                    logger.warning(
                        f"RANGING data missing required columns. Available: {list(ranging_data.columns)}"
                    )
                    self.regime_datasets["RANGING"] = None
            else:
                self.regime_datasets["RANGING"] = None

            # Volatile market (high volatility, crisis-like)
            volatile_start = datetime(2023, 7, 1)
            volatile_end = datetime(2023, 9, 30)
            volatile_data = self.data_storage.get_data_range(
                self.instrument, volatile_start, volatile_end
            )
            if volatile_data is not None and not volatile_data.empty:
                if all(col in volatile_data.columns for col in required_columns):
                    self.regime_datasets["VOLATILE"] = volatile_data
                else:
                    logger.warning(
                        f"VOLATILE data missing required columns. Available: {list(volatile_data.columns)}"
                    )
                    self.regime_datasets["VOLATILE"] = None
            else:
                self.regime_datasets["VOLATILE"] = None

            loaded_count = sum(
                1 for dataset in self.regime_datasets.values() if dataset is not None
            )
            logger.info(f"Market regime datasets loaded: {loaded_count}/3 successful")

        except Exception as e:
            logger.error(f"Error loading regime datasets: {e}")
            # Set all to None on error
            for regime in self.regime_datasets:
                self.regime_datasets[regime] = None

    def _create_synthetic_market_data(self) -> Optional[pd.DataFrame]:
        """Create synthetic market data for evaluation when real data is unavailable."""
        try:
            # Generate 1000 data points with realistic price movements
            np.random.seed(42)  # For reproducibility

            # Base price
            base_price = 1.1000  # EURUSD typical level
            prices = [base_price]

            # Generate price movements with some trend and volatility
            for i in range(999):
                # Random walk with drift
                # Small positive drift, realistic volatility
                change = np.random.normal(0.0001, 0.001)
                new_price = prices[-1] + change
                prices.append(max(0.5, new_price))  # Prevent negative prices

            # Create OHLCV data
            data = []
            for i in range(0, len(prices), 10):  # Group into bars
                if i + 10 <= len(prices):
                    bar_prices = prices[i : i + 10]
                    open_price = bar_prices[0]
                    close_price = bar_prices[-1]
                    high_price = max(bar_prices)
                    low_price = min(bar_prices)
                    volume = np.random.randint(1000, 10000)

                    data.append(
                        {
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": volume,
                            "bid_volume": volume * 0.4,  # Add bid volume
                            "ask_volume": volume * 0.6,  # Add ask volume
                        }
                    )

            if data:
                df = pd.DataFrame(data)
                df.index = pd.date_range(start="2023-01-01", periods=len(df), freq="H")
                logger.info(f"Created synthetic market data with {len(df)} bars")
                return df

            return None

        except Exception as e:
            logger.error(f"Error creating synthetic data: {e}")
            return None

    def _calculate_returns_score(self, regime_results: Dict[str, Dict]) -> float:
        """Calculate returns score across all regimes."""
        returns = []

        for regime_name, results in regime_results.items():
            if "total_return" in results:
                returns.append(results["total_return"])

        if not returns:
            return 0.0

        # Average return across regimes
        avg_return = np.mean(returns)

        # Penalize for negative returns
        if avg_return < 0:
            return float(max(float(avg_return), -1.0))

        # Cap at 100% return
        return float(min(float(avg_return), 1.0))

    def _calculate_robustness_score(self, regime_results: Dict[str, Dict]) -> float:
        """Calculate robustness score (consistency across regimes)."""
        if len(regime_results) < 2:
            return 0.0

        returns = []
        drawdowns = []

        for results in regime_results.values():
            if "total_return" in results:
                returns.append(results["total_return"])
            if "max_drawdown" in results:
                drawdowns.append(results["max_drawdown"])

        if not returns:
            return 0.0

        # Consistency of returns (lower variance = higher robustness)
        return_std = float(np.std(returns))
        return_consistency = max(0.0, 1.0 - return_std)

        # Drawdown control
        avg_drawdown = float(np.mean(drawdowns)) if drawdowns else 0.0
        drawdown_score = max(0.0, 1.0 - avg_drawdown)

        # Combined robustness score
        robustness = (return_consistency * 0.6) + (drawdown_score * 0.4)

        return max(0.0, min(1.0, robustness))

    def _calculate_adaptability_score(self, regime_results: Dict[str, Dict]) -> float:
        """Calculate adaptability score (performance variation across regimes)."""
        if len(regime_results) < 2:
            return 0.0

        regime_performance = {}

        for regime_name, results in regime_results.items():
            if "total_return" in results:
                regime_performance[regime_name] = results["total_return"]

        if len(regime_performance) < 2:
            return 0.0

        # Calculate performance variation
        performances = list(regime_performance.values())
        performance_range = max(performances) - min(performances)

        # Higher adaptability means better performance in different regimes
        # but not necessarily consistent performance (that's robustness)
        adaptability = performance_range * 0.5  # Normalize

        return max(0.0, min(1.0, adaptability))

    def _calculate_efficiency_score(self, regime_results: Dict[str, Dict]) -> float:
        """Calculate efficiency score (risk-adjusted returns)."""
        sharpe_ratios = []
        sortino_ratios = []
        profit_factors = []

        for results in regime_results.values():
            if "sharpe_ratio" in results:
                sharpe_ratios.append(results["sharpe_ratio"])
            if "sortino_ratio" in results:
                sortino_ratios.append(results["sortino_ratio"])
            if "profit_factor" in results:
                profit_factors.append(results["profit_factor"])

        # Average efficiency metrics
        avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0.0
        avg_sortino = np.mean(sortino_ratios) if sortino_ratios else 0.0
        avg_profit_factor = np.mean(profit_factors) if profit_factors else 1.0

        # Normalize and combine
        # Cap at 2.0 Sharpe
        sharpe_score = max(0.0, min(1.0, float(avg_sharpe) / 2.0))
        # Cap at 2.0 Sortino
        sortino_score = max(0.0, min(1.0, float(avg_sortino) / 2.0))
        profit_factor_score = max(
            # Cap at 3.0 PF
            0.0,
            min(1.0, float(avg_profit_factor - 1.0) / 2.0),
        )

        efficiency = (
            (sharpe_score * 0.4) + (sortino_score * 0.4) + (profit_factor_score * 0.2)
        )

        return max(0.0, min(1.0, efficiency))

    def _calculate_antifragility_score(self, regime_results: Dict[str, Dict]) -> float:
        """Calculate anti-fragility score (benefits from volatility)."""
        if len(regime_results) < 2:
            return 0.0

        # Anti-fragility: better performance in volatile conditions
        volatile_performance = 0.0
        stable_performance = 0.0

        if "VOLATILE" in regime_results:
            volatile_results = regime_results["VOLATILE"]
            if "total_return" in volatile_results:
                volatile_performance = volatile_results["total_return"]

        # Average of other regimes
        other_performances = []
        for regime_name, results in regime_results.items():
            if regime_name != "VOLATILE" and "total_return" in results:
                other_performances.append(results["total_return"])

        if other_performances:
            stable_performance = np.mean(other_performances)

        # Anti-fragility: better performance in volatile vs stable conditions
        if stable_performance > 0:
            antifragility_ratio = volatile_performance / stable_performance
        else:
            antifragility_ratio = (
                volatile_performance if volatile_performance > 0 else 0
            )

        # Normalize to [0, 1] range
        antifragility = max(0.0, min(1.0, float(antifragility_ratio)))

        return antifragility

    def _calculate_overall_fitness(self, scores: Dict[str, float]) -> float:
        """Calculate overall fitness using weighted combination."""
        overall_fitness = 0.0

        for dimension, score in scores.items():
            weight = self.weights.get(dimension, 0.0)
            overall_fitness += score * weight

        return max(0.0, min(1.0, overall_fitness))


class EvolutionEngine:
    """
    v2.0 Evolution Engine with anti-fragility selection pressure.

    This class implements the genetic algorithm from the original unified file,
    driving populations toward resilience and adaptability through natural selection.
    """

    def __init__(self, config: EvolutionConfig, fitness_evaluator: FitnessEvaluator):
        """
        Initialize the evolution engine.

        Args:
            config: Evolution configuration
            fitness_evaluator: Fitness evaluator
        """
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.population: List[DecisionGenome] = []
        self.generation = 0
        self.best_genome = None
        self.evolution_history: List[EvolutionStats] = []
        self.genome_counter = 0  # Counter for unique genome IDs

        logger.info("EvolutionEngine initialized with anti-fragility selection")

    def initialize_population(self, seed: Optional[int] = None) -> bool:
        """
        Initialize the population with random genomes.

        Args:
            seed: Random seed for reproducibility

        Returns:
            True if initialization successful
        """
        try:
            if seed is not None:
                np.random.seed(seed)

            self.population = []

            for i in range(self.config.population_size):
                genome = self._create_random_genome(f"genome_{i:04d}")
                self.population.append(genome)

            self.generation = 0
            logger.info(f"Population initialized with {len(self.population)} genomes")
            return True

        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            return False

    def evolve_generation(self) -> EvolutionStats:
        """
        Evolve the population for one generation.

        Returns:
            Evolution statistics
        """
        try:
            # Evaluate current population
            self._evaluate_population()

            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness_score, reverse=True)

            # Update best genome
            if (
                not self.best_genome
                or self.population[0].fitness_score > self.best_genome.fitness_score
            ):
                self.best_genome = self.population[0]

            # Calculate statistics
            stats = self._calculate_generation_stats()
            self.evolution_history.append(stats)

            # Check for convergence
            if stats.convergence_rate < self.config.convergence_threshold:
                logger.info(f"Convergence reached at generation {self.generation}")
                return stats

            # Create next generation
            next_generation = []

            # Elitism: keep best genomes
            elite_count = int(self.config.population_size * self.config.elite_ratio)
            next_generation.extend(self.population[:elite_count])

            # Crossover
            crossover_count = int(
                self.config.population_size * self.config.crossover_ratio
            )
            for _ in range(crossover_count):
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
                next_generation.append(child)

            # Mutation
            mutation_count = self.config.population_size - len(next_generation)
            for _ in range(mutation_count):
                parent = self._tournament_selection()
                child = self._mutate(parent)
                next_generation.append(child)

            # Update population
            self.population = next_generation
            self.generation += 1

            logger.info(
                f"Generation {self.generation} evolved - Best fitness: {stats.best_fitness:.4f}"
            )

            return stats

        except Exception as e:
            logger.error(f"Error evolving generation: {e}")
            return EvolutionStats(
                generation=self.generation,
                best_fitness=0.0,
                avg_fitness=0.0,
                worst_fitness=0.0,
                diversity=0.0,
                convergence_rate=0.0,
                best_genome_id="",
                population_size=len(self.population),
            )

    def _create_random_genome(self, genome_id: str) -> DecisionGenome:
        """Create a random genome."""
        # Simple random decision tree
        decision_tree = {
            "type": "random",
            "parameters": {
                "buy_threshold": np.random.uniform(0.3, 0.7),
                "sell_threshold": np.random.uniform(0.3, 0.7),
                "momentum_weight": np.random.uniform(0.1, 0.9),
                "trend_weight": np.random.uniform(0.1, 0.9),
                "institutional_weight": np.random.uniform(0.1, 0.9),
            },
        }

        return DecisionGenome(
            genome_id=genome_id, decision_tree=decision_tree, generation=0
        )

    def _evaluate_population(self):
        """Evaluate all genomes in the population."""
        # Create sensory cortex once for efficiency (deferred import to avoid circular dependency)
        from sensory.core.base import InstrumentMeta
        from sensory.orchestration.master_orchestrator import \
            MasterOrchestrator

        instrument_meta = InstrumentMeta(
            symbol=self.fitness_evaluator.instrument,
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            typical_spread=0.00015,
            avg_daily_range=0.01,
        )
        sensory_cortex = MasterOrchestrator(instrument_meta)

        for genome in self.population:
            self.fitness_evaluator.evaluate_genome(genome, sensory_cortex)

    def _tournament_selection(self) -> DecisionGenome:
        """Select a genome using tournament selection."""
        tournament_size = min(self.config.tournament_size, len(self.population))
        tournament_indices = np.random.choice(
            len(self.population), size=tournament_size, replace=False
        )
        tournament = [self.population[i] for i in tournament_indices]
        return max(tournament, key=lambda g: g.fitness_score)

    def _crossover(
        self, parent1: DecisionGenome, parent2: DecisionGenome
    ) -> DecisionGenome:
        """Perform crossover between two parents."""
        # Simple parameter crossover
        child_params = {}
        for param in parent1.decision_tree["parameters"]:
            if np.random.random() < 0.5:
                child_params[param] = parent1.decision_tree["parameters"][param]
            else:
                child_params[param] = parent2.decision_tree["parameters"][param]

        child_tree = {"type": "crossover", "parameters": child_params}

        child_id = f"genome_{self.genome_counter:04d}"
        self.genome_counter += 1

        return DecisionGenome(
            genome_id=child_id,
            decision_tree=child_tree,
            generation=self.generation + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id],
            crossover_count=1,
        )

    def _mutate(self, parent: DecisionGenome) -> DecisionGenome:
        """Mutate a parent genome."""
        # Add random noise to parameters
        mutated_params = {}
        for param, value in parent.decision_tree["parameters"].items():
            noise = np.random.normal(0, 0.1)
            mutated_params[param] = max(0.0, min(1.0, value + noise))

        mutated_tree = {"type": "mutation", "parameters": mutated_params}

        child_id = f"genome_{self.genome_counter:04d}"
        self.genome_counter += 1

        return DecisionGenome(
            genome_id=child_id,
            decision_tree=mutated_tree,
            generation=self.generation + 1,
            parent_ids=[parent.genome_id],
            mutation_count=parent.mutation_count + 1,
        )

    def _calculate_generation_stats(self) -> EvolutionStats:
        """Calculate statistics for the current generation."""
        if not self.population:
            return EvolutionStats(
                generation=self.generation,
                best_fitness=0.0,
                avg_fitness=0.0,
                worst_fitness=0.0,
                diversity=0.0,
                convergence_rate=0.0,
                best_genome_id="",
                population_size=0,
            )

        fitness_scores = [g.fitness_score for g in self.population]

        # Calculate diversity (standard deviation of fitness)
        diversity = np.std(fitness_scores) if len(fitness_scores) > 1 else 0.0

        # Calculate convergence rate
        convergence_rate = 0.0
        if len(self.evolution_history) > 1:
            prev_best = self.evolution_history[-1].best_fitness
            current_best = max(fitness_scores)
            convergence_rate = abs(current_best - prev_best)

        return EvolutionStats(
            generation=self.generation,
            best_fitness=float(max(fitness_scores)),
            avg_fitness=float(np.mean(fitness_scores)),
            worst_fitness=float(min(fitness_scores)),
            diversity=float(diversity),
            convergence_rate=convergence_rate,
            best_genome_id=self.population[0].genome_id,
            population_size=len(self.population),
        )

    def get_best_genomes(self, count: int = 5) -> List[DecisionGenome]:
        """Get the best genomes from the current population."""
        sorted_population = sorted(
            self.population, key=lambda g: g.fitness_score, reverse=True
        )
        return sorted_population[:count]

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution progress."""
        if not self.evolution_history:
            return {}

        return {
            "total_generations": self.generation,
            "best_fitness_history": [
                stats.best_fitness for stats in self.evolution_history
            ],
            "avg_fitness_history": [
                stats.avg_fitness for stats in self.evolution_history
            ],
            "diversity_history": [stats.diversity for stats in self.evolution_history],
            "best_genome": self.best_genome.genome_id if self.best_genome else None,
            "best_fitness": self.best_genome.fitness_score if self.best_genome else 0.0,
            "population_size": len(self.population),
        }
