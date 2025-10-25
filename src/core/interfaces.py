"""
Core interfaces for EMP system components.

These protocols define contracts that enable A/B testing and hot-swapping
of implementations without refactoring. Each interface is minimal and focused.

Usage:
    from src.core.interfaces import ForecastingSensor, ExecutionPolicy
    
    class MyForecaster(ForecastingSensor):
        def predict(self, ts_window, meta):
            # Implementation
            return {'mu': ..., 'sigma': ..., 'conf': ...}
"""

from typing import Protocol, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np


# ============================================================================
# 1. ForecastingSensor Interface
# ============================================================================

class ForecastingSensor(Protocol):
    """
    Contract for all forecasting sensors (WHAT, WHY, HOW, WHEN, ANOMALY).
    
    Enables A/B testing of forecasting methods (LSTM vs. Transformer vs. iTransformer)
    without changing downstream code.
    
    Example:
        sensor = iTransformerSensor()  # or LSTMSensor() or RidgeRegressionSensor()
        forecast = sensor.predict(price_window, {'symbol': 'EUR/USD', 'regime': 'bull'})
        
        if forecast['conf'] > 0.7:
            # High confidence forecast
            execute_trade(forecast['mu'], size=1.0)
        else:
            # Low confidence, reduce size
            execute_trade(forecast['mu'], size=0.5)
    """
    
    def predict(
        self, 
        ts_window: np.ndarray,  # Shape: (T, features), time series window
        meta: Dict[str, Any]     # Metadata: symbol, regime, timestamp, etc.
    ) -> Dict[str, Any]:
        """
        Generate forecast with uncertainty quantification.
        
        Args:
            ts_window: Time series data, shape (T, features)
                      - T: Number of time steps (e.g., 100 for 100 minutes)
                      - features: OHLCV, volume, spread, etc.
            meta: Metadata dictionary with keys:
                  - 'symbol': str (e.g., 'EUR/USD')
                  - 'regime': str (e.g., 'bull', 'bear', 'transitional')
                  - 'timestamp': datetime (forecast generation time)
                  - ... (any other context)
        
        Returns:
            Dictionary with keys:
                - 'mu': float - Point forecast (e.g., predicted return)
                - 'sigma': float - Uncertainty (standard deviation)
                - 'conf': float - Confidence score [0, 1]
                - 'horizon': int - Forecast horizon in time steps
                - 'timestamp': datetime - When forecast was generated
                - 'model_version': str - Model identifier for audit trail
        
        Example:
            >>> sensor = iTransformerSensor()
            >>> window = np.random.randn(100, 5)  # 100 time steps, 5 features
            >>> meta = {'symbol': 'EUR/USD', 'regime': 'bull', 'timestamp': datetime.now()}
            >>> forecast = sensor.predict(window, meta)
            >>> print(forecast)
            {
                'mu': 0.0012,           # Predicted 0.12% return
                'sigma': 0.0008,        # ±0.08% uncertainty
                'conf': 0.85,           # 85% confidence
                'horizon': 60,          # 60-minute ahead forecast
                'timestamp': datetime(...),
                'model_version': 'iTransformer-v1.2'
            }
        """
        ...


# ============================================================================
# 2. ExecutionPolicy Interface
# ============================================================================

@dataclass
class Order:
    """Parent order to be scheduled into child orders."""
    symbol: str              # e.g., 'EUR/USD'
    side: str                # 'buy' or 'sell'
    quantity: float          # Total quantity to execute
    urgency: float           # [0, 1], 0=patient, 1=urgent
    alpha_half_life: float   # Alpha decay half-life in seconds
    limit_price: float | None = None  # Optional limit price


@dataclass
class ChildOrder:
    """Scheduled child order with timing and sizing."""
    symbol: str
    side: str                # 'buy' or 'sell'
    quantity: float          # Quantity for this child order
    schedule_time: datetime  # When to submit this order
    limit_price: float | None  # Limit price (None for market order)
    order_type: str          # 'market', 'limit', 'twap', 'vwap', 'is'
    parent_id: str           # Link back to parent order


class ExecutionPolicy(Protocol):
    """
    Contract for execution scheduling (Almgren-Chriss, RL, TWAP, VWAP, IS).
    
    Enables A/B testing of execution algorithms without changing order management.
    
    Example:
        policy = RLExecutionPolicy()  # or AlmgrenChrissPolicy() or TWAPPolicy()
        child_orders = policy.schedule(
            order=Order('EUR/USD', 'buy', 10000, urgency=0.5, alpha_half_life=300),
            liquidity={'bid_depth': 50000, 'ask_depth': 60000, ...},
            impact_params={'permanent': 0.1, 'temporary': 0.05, ...}
        )
        
        for child in child_orders:
            submit_at(child.schedule_time, child)
    """
    
    def schedule(
        self,
        order: Order,
        liquidity: Dict[str, Any],      # Current limit order book state
        impact_params: Dict[str, Any]   # Calibrated market impact parameters
    ) -> List[ChildOrder]:
        """
        Schedule parent order into child orders with timing and sizing.
        
        Args:
            order: Parent order to be executed
            liquidity: Current market liquidity state
                      - 'bid_depth': float (total quantity at best bid)
                      - 'ask_depth': float (total quantity at best ask)
                      - 'bid_ask_spread': float (bps)
                      - 'recent_volume': float (last 5-minute volume)
                      - 'volatility': float (recent realized volatility)
            impact_params: Calibrated impact model parameters
                          - 'permanent': float (permanent impact coefficient)
                          - 'temporary': float (temporary impact coefficient)
                          - 'eta': float (square-root law exponent, typically 0.5)
        
        Returns:
            List of ChildOrder objects with schedule times and sizes.
            
            Constraints:
                - sum(child.quantity) == order.quantity (conservation)
                - All schedule_times >= current_time (no past scheduling)
                - Respects urgency (high urgency → fewer, larger children)
                - Respects alpha_half_life (faster execution if alpha decays quickly)
        
        Example:
            >>> policy = AlmgrenChrissPolicy()
            >>> order = Order('EUR/USD', 'buy', 10000, urgency=0.5, alpha_half_life=300)
            >>> liquidity = {'bid_depth': 50000, 'ask_depth': 60000, 'bid_ask_spread': 1.5}
            >>> impact_params = {'permanent': 0.1, 'temporary': 0.05, 'eta': 0.5}
            >>> children = policy.schedule(order, liquidity, impact_params)
            >>> for child in children:
            ...     print(f"{child.schedule_time}: {child.side} {child.quantity} @ {child.order_type}")
            2025-10-25 10:00:00: buy 2000 @ limit
            2025-10-25 10:05:00: buy 2000 @ limit
            2025-10-25 10:10:00: buy 2000 @ limit
            2025-10-25 10:15:00: buy 2000 @ limit
            2025-10-25 10:20:00: buy 2000 @ limit
        """
        ...


# ============================================================================
# 3. RegimeRouter Interface
# ============================================================================

class RegimeRouter(Protocol):
    """
    Contract for regime-aware strategy routing.
    
    Enables A/B testing of routing methods (MoE, logistic regression, GBM, manual rules)
    without changing capital allocation logic.
    
    Example:
        router = MoERouter()  # or LogisticRouter() or ManualRulesRouter()
        snapshot = {'volatility': 0.015, 'trend': 0.8, 'liquidity': 0.6, ...}
        weights = router.route(snapshot)
        
        for strategy_id, weight in weights.items():
            allocate_capital(strategy_id, weight * total_capital)
    """
    
    def route(
        self,
        snapshot: Dict[str, Any]  # Market snapshot (BeliefState from sensory cortex)
    ) -> Dict[str, float]:
        """
        Route capital to strategies based on current market state.
        
        Args:
            snapshot: Market state snapshot with keys:
                     - 'volatility': float (realized volatility)
                     - 'trend': float (trend strength, [-1, 1])
                     - 'liquidity': float (market liquidity score, [0, 1])
                     - 'regime': str (detected regime: 'bull', 'bear', etc.)
                     - 'anomaly_score': float (anomaly detection score, [0, 1])
                     - ... (any other sensory cortex outputs)
        
        Returns:
            Dictionary mapping strategy_id to target weight [0, 1].
            
            Constraints:
                - sum(weights.values()) == 1.0 (full allocation)
                - All weights >= 0 (no short strategies)
                - Includes hysteresis (doesn't flip regimes rapidly)
                - Respects strategy capacity constraints
        
        Example:
            >>> router = MoERouter()
            >>> snapshot = {
            ...     'volatility': 0.015,
            ...     'trend': 0.8,
            ...     'liquidity': 0.6,
            ...     'regime': 'bull',
            ...     'anomaly_score': 0.1
            ... }
            >>> weights = router.route(snapshot)
            >>> print(weights)
            {
                'momentum_1h': 0.45,      # High trend → momentum gets 45%
                'breakout_4h': 0.30,      # Bull regime → breakout gets 30%
                'mean_reversion_15m': 0.15,  # Low volatility → mean-reversion gets 15%
                'market_making': 0.10     # High liquidity → market-making gets 10%
            }
        
        Hysteresis example:
            If regime was 'bull' and current signal is 'transitional', router may
            maintain 'bull' weights for N time steps before switching to avoid
            rapid regime flipping.
        """
        ...


# ============================================================================
# 4. EvolutionEngine Interface
# ============================================================================

@dataclass
class Strategy:
    """Strategy representation for evolution."""
    id: str                          # Unique identifier
    params: Dict[str, Any]           # Strategy parameters
    fitness: float | None = None     # Fitness score (set by fitness_fn)
    genealogy: List[str] | None = None  # Parent IDs
    generation: int = 0              # Generation number


class PopulationUpdater(Protocol):
    """
    Pluggable population update strategy for evolutionary algorithms.
    
    Enables A/B testing of evolution methods (NSGA-II, Spectral GNN, diversity penalties)
    without changing evolution loop.
    
    Example:
        updater = SpectralGNNUpdater()  # or NSGA2Updater() or DiversityPenaltyUpdater()
        
        for generation in range(100):
            # Evaluate fitness
            for strategy in population:
                strategy.fitness = evaluate_strategy(strategy)
            
            # Update population
            population = updater.update(population, fitness_fn=evaluate_strategy)
            
            # Log best strategy
            best = max(population, key=lambda s: s.fitness)
            print(f"Gen {generation}: Best fitness = {best.fitness}")
    """
    
    def update(
        self,
        population: List[Strategy],
        fitness_fn: Callable[[Strategy], float]
    ) -> List[Strategy]:
        """
        Update population for one generation.
        
        Args:
            population: Current population of strategies
            fitness_fn: Function to evaluate strategy fitness
                       - Input: Strategy object
                       - Output: float (higher is better)
                       - Example: Sharpe ratio, deflated Sharpe, multi-objective score
        
        Returns:
            Updated population after selection, crossover, mutation.
            
            Constraints:
                - len(new_population) == len(population) (constant size)
                - Maintains diversity (no premature convergence)
                - Respects elitism (keeps top N strategies)
                - Updates genealogy (tracks parent-child relationships)
        
        Implementation notes:
            - NSGA2Updater: Non-dominated sorting, crowding distance
            - SpectralGNNUpdater: Graph representation, spectral filtering
            - DiversityPenaltyUpdater: Fitness penalized by similarity to others
        
        Example:
            >>> updater = NSGA2Updater(population_size=50, elite_size=5)
            >>> population = [Strategy(id=f"s{i}", params={...}) for i in range(50)]
            >>> 
            >>> def fitness_fn(strategy):
            ...     # Evaluate strategy on historical data
            ...     sharpe = backtest(strategy)
            ...     return sharpe
            >>> 
            >>> for gen in range(100):
            ...     population = updater.update(population, fitness_fn)
            ...     best = max(population, key=lambda s: s.fitness)
            ...     print(f"Gen {gen}: Best Sharpe = {best.fitness:.3f}")
            Gen 0: Best Sharpe = 1.234
            Gen 1: Best Sharpe = 1.456
            ...
            Gen 99: Best Sharpe = 2.789
        """
        ...


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    """
    Example usage showing how interfaces enable A/B testing.
    """
    
    # Example 1: A/B test forecasting sensors
    from src.sensory.what.lstm_sensor import LSTMSensor
    from src.sensory.what.itransformer_sensor import iTransformerSensor
    
    sensors = {
        'baseline': LSTMSensor(),
        'candidate': iTransformerSensor()
    }
    
    window = np.random.randn(100, 5)
    meta = {'symbol': 'EUR/USD', 'regime': 'bull', 'timestamp': datetime.now()}
    
    for name, sensor in sensors.items():
        forecast = sensor.predict(window, meta)
        print(f"{name}: mu={forecast['mu']:.4f}, conf={forecast['conf']:.2f}")
    
    
    # Example 2: A/B test execution policies
    from src.execution.almgren_chriss import AlmgrenChrissPolicy
    from src.execution.rl_execution import RLExecutionPolicy
    
    policies = {
        'baseline': AlmgrenChrissPolicy(),
        'candidate': RLExecutionPolicy()
    }
    
    order = Order('EUR/USD', 'buy', 10000, urgency=0.5, alpha_half_life=300)
    liquidity = {'bid_depth': 50000, 'ask_depth': 60000, 'bid_ask_spread': 1.5}
    impact_params = {'permanent': 0.1, 'temporary': 0.05, 'eta': 0.5}
    
    for name, policy in policies.items():
        children = policy.schedule(order, liquidity, impact_params)
        print(f"{name}: {len(children)} child orders, total qty = {sum(c.quantity for c in children)}")
    
    
    # Example 3: A/B test regime routers
    from src.thinking.moe_router import MoERouter
    from src.thinking.logistic_router import LogisticRouter
    
    routers = {
        'baseline': LogisticRouter(),
        'candidate': MoERouter()
    }
    
    snapshot = {
        'volatility': 0.015,
        'trend': 0.8,
        'liquidity': 0.6,
        'regime': 'bull',
        'anomaly_score': 0.1
    }
    
    for name, router in routers.items():
        weights = router.route(snapshot)
        print(f"{name}: {weights}")
    
    
    # Example 4: A/B test evolution updaters
    from src.evolution.nsga2 import NSGA2Updater
    from src.evolution.spectral_gnn import SpectralGNNUpdater
    
    updaters = {
        'baseline': NSGA2Updater(population_size=50, elite_size=5),
        'candidate': SpectralGNNUpdater(population_size=50, elite_size=5)
    }
    
    population = [Strategy(id=f"s{i}", params={'param1': i}) for i in range(50)]
    
    def fitness_fn(strategy):
        # Dummy fitness function
        return np.random.rand()
    
    for name, updater in updaters.items():
        new_pop = updater.update(population, fitness_fn)
        avg_fitness = np.mean([s.fitness for s in new_pop if s.fitness is not None])
        print(f"{name}: avg fitness = {avg_fitness:.3f}")

