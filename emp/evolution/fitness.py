"""
FitnessEvaluator: Comprehensive fitness evaluation for trading genomes.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class FitnessScore:
    """Comprehensive fitness score for a genome"""
    genome_id: str
    
    # Core performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # v2.0: Multi-objective fitness metrics
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    consistency_score: float = 0.0
    
    # Robustness metrics
    volatility_adjusted_return: float = 0.0
    tail_risk_score: float = 0.0
    stress_test_score: float = 0.0
    
    # Adaptability metrics
    regime_adaptation_score: float = 0.0
    learning_rate_score: float = 0.0
    
    # Efficiency metrics
    trade_frequency: float = 0.0
    transaction_cost_impact: float = 0.0
    
    # Antifragility metrics
    adversarial_performance: float = 0.0
    black_swan_resilience: float = 0.0
    
    # Composite scores
    returns_score: float = 0.0
    robustness_score: float = 0.0
    adaptability_score: float = 0.0
    efficiency_score: float = 0.0
    antifragility_score: float = 0.0
    
    # Final weighted score
    total_fitness: float = 0.0
    
    # v2.0: Regime-specific scores for triathlon analysis
    regime_scores: Optional[Dict[str, float]] = None
    
    # Metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    trades_analyzed: int = 0
    simulation_duration: timedelta = field(default_factory=lambda: timedelta(0))

class FitnessEvaluator:
    """
    Comprehensive fitness evaluator for trading genomes.
    
    Evaluates genomes across multiple dimensions:
    - Returns performance
    - Risk management
    - Adaptability to different market regimes
    - Efficiency and cost management
    - Antifragility and robustness
    """
    
    def __init__(self, data_storage, 
                 evaluation_period_days: int = 30,
                 adversarial_intensity: float = 0.7,
                 commission_rate: float = 0.0001,  # 1 pip commission
                 slippage_bps: float = 0.5):       # 0.5 bps slippage
        
        self.data_storage = data_storage
        self.evaluation_period_days = evaluation_period_days
        self.adversarial_intensity = adversarial_intensity
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        
        # Regime datasets for triathlon evaluation
        self.regime_datasets: Dict[str, Dict] = {}
        self.synthetic_regimes: Dict[str, Dict] = {}
        
        # Performance tracking
        self.evaluation_history: List[FitnessScore] = []
        self.fitness_distribution: Dict[str, List[float]] = {}
        
        # Initialize regime datasets
        self._identify_regime_datasets()
        self._create_synthetic_regimes()
        
        logger.info("Initialized FitnessEvaluator")
    
    def _identify_regime_datasets(self):
        """Identify different market regime datasets for evaluation."""
        try:
            # Load regime configuration
            regime_config = self.data_storage.load_regime_config()
            
            if regime_config:
                for regime_name, regime_data in regime_config.items():
                    self.regime_datasets[regime_name] = {
                        'config': regime_data,
                        'data': None,  # Will be loaded on demand
                        'description': regime_data.get('description', ''),
                        'characteristics': regime_data.get('characteristics', [])
                    }
            
            logger.info(f"Identified {len(self.regime_datasets)} regime datasets")
            
        except Exception as e:
            logger.warning(f"Could not load regime datasets: {e}")
            # Create default regimes
            self._create_default_regimes()
    
    def _create_default_regimes(self):
        """Create default regime datasets if none are available."""
        default_regimes = {
            'trending': {
                'description': 'Strong directional movement',
                'characteristics': ['high_volatility', 'directional_bias', 'momentum']
            },
            'ranging': {
                'description': 'Sideways consolidation',
                'characteristics': ['low_volatility', 'mean_reversion', 'support_resistance']
            },
            'volatile': {
                'description': 'High volatility with choppy movement',
                'characteristics': ['high_volatility', 'random_walk', 'breakouts']
            }
        }
        
        for regime_name, regime_data in default_regimes.items():
            self.regime_datasets[regime_name] = {
                'config': regime_data,
                'data': None,
                'description': regime_data['description'],
                'characteristics': regime_data['characteristics']
            }
    
    def _create_synthetic_regimes(self):
        """Create synthetic regime datasets for comprehensive testing."""
        # Generate synthetic data for different market conditions
        synthetic_regimes = {
            'bull_trend': {
                'description': 'Synthetic bull market',
                'volatility': 0.02,
                'trend_strength': 0.8,
                'mean_reversion': 0.1
            },
            'bear_trend': {
                'description': 'Synthetic bear market',
                'volatility': 0.025,
                'trend_strength': -0.7,
                'mean_reversion': 0.1
            },
            'sideways': {
                'description': 'Synthetic sideways market',
                'volatility': 0.015,
                'trend_strength': 0.0,
                'mean_reversion': 0.8
            },
            'crisis': {
                'description': 'Synthetic crisis market',
                'volatility': 0.05,
                'trend_strength': -0.9,
                'mean_reversion': 0.0
            }
        }
        
        for regime_name, regime_data in synthetic_regimes.items():
            self.synthetic_regimes[regime_name] = regime_data
        
        logger.info(f"Created {len(self.synthetic_regimes)} synthetic regimes")
    
    def evaluate_genome(self, genome) -> FitnessScore:
        """
        Evaluate a genome's fitness across multiple dimensions.
        
        Args:
            genome: DecisionGenome to evaluate
            
        Returns:
            FitnessScore with comprehensive metrics
        """
        logger.info(f"Evaluating genome {genome.genome_id}")
        
        # Create fitness score object
        fitness_score = FitnessScore(genome_id=genome.genome_id)
        
        # Evaluate across different regimes
        regime_results = {}
        
        # Evaluate on real regime data
        for regime_name, regime_data in self.regime_datasets.items():
            try:
                result = self._run_simulation_for_regime(genome, regime_data['config'])
                if result:
                    regime_results[regime_name] = result
            except Exception as e:
                logger.warning(f"Failed to evaluate regime {regime_name}: {e}")
        
        # Evaluate on synthetic regimes
        for regime_name, regime_config in self.synthetic_regimes.items():
            try:
                result = self._run_simulation_for_regime(genome, regime_config)
                if result:
                    regime_results[f"synthetic_{regime_name}"] = result
            except Exception as e:
                logger.warning(f"Failed to evaluate synthetic regime {regime_name}: {e}")
        
        # Calculate comprehensive scores
        self._calculate_comprehensive_scores(fitness_score, regime_results, genome)
        
        # Store evaluation history
        self.evaluation_history.append(fitness_score)
        
        # Update fitness distribution
        self._update_fitness_distribution(fitness_score)
        
        logger.info(f"Genome {genome.genome_id} fitness: {fitness_score.total_fitness:.4f}")
        
        return fitness_score
    
    def _run_simulation_for_regime(self, genome, regime_config: Dict) -> Optional[Dict[str, Any]]:
        """Run simulation for a specific market regime."""
        try:
            # Create market simulator for this regime
            from emp.simulation.market import MarketSimulator
            from emp.agent.sensory import SensoryCortex
            from emp.simulation.adversary import AdversarialEngine
            
            # Initialize components
            simulator = MarketSimulator(
                data_storage=self.data_storage,
                initial_balance=100000.0,
                leverage=1.0
            )
            
            # Add adversarial engine
            adversary = AdversarialEngine(difficulty_level=self.adversarial_intensity)
            simulator.add_adversarial_callback(adversary.apply_adversarial_effects)
            
            # Initialize sensory cortex
            sensory_cortex = SensoryCortex(symbol="EURUSD", data_storage=self.data_storage)
            
            # Load data for evaluation period
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.evaluation_period_days)
            
            # Load data (use synthetic data if no real data available)
            if regime_config.get('synthetic', False):
                # Generate synthetic data based on regime characteristics
                data = self._generate_synthetic_data(regime_config, start_time, end_time)
                simulator.load_data_from_dataframe(data)
            else:
                # Load real data
                simulator.load_data("EURUSD", start_time, end_time)
            
            # Calibrate sensory cortex
            sensory_cortex.calibrate(start_time, end_time)
            
            # Run simulation
            simulation_results = self._run_simulation(genome, simulator, sensory_cortex)
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Failed to run simulation for regime: {e}")
            return None
    
    def _generate_synthetic_data(self, regime_config: Dict, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Generate synthetic market data based on regime characteristics."""
        # Generate timestamps
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1T')
        n_ticks = len(timestamps)
        
        # Base price
        base_price = 1.1000
        
        # Generate price series based on regime characteristics
        volatility = regime_config.get('volatility', 0.02)
        trend_strength = regime_config.get('trend_strength', 0.0)
        mean_reversion = regime_config.get('mean_reversion', 0.0)
        
        # Generate returns
        np.random.seed(42)  # For reproducibility
        
        # Random walk component
        random_returns = np.random.normal(0, volatility, n_ticks)
        
        # Trend component
        trend_returns = np.linspace(0, trend_strength, n_ticks) / n_ticks
        
        # Mean reversion component
        if mean_reversion > 0:
            mean_reversion_returns = np.zeros(n_ticks)
            for i in range(1, n_ticks):
                mean_reversion_returns[i] = -mean_reversion * (random_returns[i-1])
        else:
            mean_reversion_returns = np.zeros(n_ticks)
        
        # Combine components
        total_returns = random_returns + trend_returns + mean_reversion_returns
        
        # Calculate prices
        prices = [base_price]
        for ret in total_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create tick data
        tick_data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            spread = price * 0.0001  # 1 pip spread
            bid = price - spread / 2
            ask = price + spread / 2
            
            tick_data.append({
                'timestamp': timestamp,
                'bid': bid,
                'ask': ask,
                'bid_volume': np.random.uniform(1000, 10000),
                'ask_volume': np.random.uniform(1000, 10000)
            })
        
        return pd.DataFrame(tick_data)
    
    def _calculate_regime_fitness(self, simulation_results: Dict, genome) -> float:
        """Calculate fitness score for a specific regime."""
        if not simulation_results:
            return 0.0
        
        # Extract key metrics
        total_return = simulation_results.get('total_return', 0.0)
        sharpe_ratio = simulation_results.get('sharpe_ratio', 0.0)
        max_drawdown = simulation_results.get('max_drawdown', 0.0)
        win_rate = simulation_results.get('win_rate', 0.0)
        
        # Calculate regime-specific fitness
        # Weight different metrics based on regime characteristics
        fitness = (
            total_return * 0.4 +
            sharpe_ratio * 0.3 +
            (1 - max_drawdown) * 0.2 +
            win_rate * 0.1
        )
        
        return max(0.0, fitness)
    
    def _calculate_sortino_ratio(self, simulation_results: Dict) -> float:
        """Calculate Sortino ratio (risk-adjusted return using downside deviation)."""
        returns = simulation_results.get('returns', [])
        if not returns:
            return 0.0
        
        returns = np.array(returns)
        mean_return = np.mean(returns)
        
        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        sortino_ratio = mean_return / downside_deviation
        return sortino_ratio
    
    def _calculate_calmar_ratio(self, simulation_results: Dict) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        total_return = simulation_results.get('total_return', 0.0)
        max_drawdown = simulation_results.get('max_drawdown', 0.0)
        
        if max_drawdown == 0:
            return 0.0 if total_return == 0 else float('inf')
        
        calmar_ratio = total_return / max_drawdown
        return calmar_ratio
    
    def _calculate_profit_factor(self, simulation_results: Dict) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        trades = simulation_results.get('trades', [])
        if not trades:
            return 0.0
        
        gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        profit_factor = gross_profit / gross_loss
        return profit_factor
    
    def _calculate_consistency_score(self, simulation_results: Dict) -> float:
        """Calculate consistency score based on return stability."""
        returns = simulation_results.get('returns', [])
        if not returns:
            return 0.0
        
        returns = np.array(returns)
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if mean_return == 0:
            return 0.0
        
        cv = std_return / abs(mean_return)
        consistency_score = 1.0 / (1.0 + cv)  # Normalize to 0-1
        
        return consistency_score
    
    def _calculate_complexity_penalty(self, genome) -> float:
        """Calculate complexity penalty for genome."""
        complexity = genome.get_complexity()
        nodes = complexity.get('nodes', 0)
        depth = complexity.get('depth', 0)
        
        # Penalty increases with complexity
        complexity_penalty = (nodes * 0.001) + (depth * 0.01)
        return complexity_penalty
    
    def _calculate_robustness_score(self, simulation_results: Dict) -> float:
        """Calculate robustness score based on performance stability."""
        if not simulation_results:
            return 0.0
        
        # Extract metrics
        sharpe_ratio = simulation_results.get('sharpe_ratio', 0.0)
        sortino_ratio = simulation_results.get('sortino_ratio', 0.0)
        max_drawdown = simulation_results.get('max_drawdown', 0.0)
        consistency = simulation_results.get('consistency_score', 0.0)
        
        # Calculate robustness components
        risk_adjusted_return = (sharpe_ratio + sortino_ratio) / 2
        drawdown_penalty = 1.0 - max_drawdown
        stability_score = consistency
        
        # Combine into robustness score
        robustness_score = (
            risk_adjusted_return * 0.4 +
            drawdown_penalty * 0.3 +
            stability_score * 0.3
        )
        
        return max(0.0, robustness_score)
    
    def _calculate_comprehensive_scores(self, fitness_score: FitnessScore, 
                                      regime_results: Dict[str, Dict], 
                                      genome):
        """Calculate comprehensive fitness scores."""
        if not regime_results:
            fitness_score.total_fitness = 0.0
            return
        
        # Calculate regime-specific scores
        regime_scores = {}
        for regime_name, results in regime_results.items():
            regime_fitness = self._calculate_regime_fitness(results, genome)
            regime_scores[regime_name] = regime_fitness
        
        fitness_score.regime_scores = regime_scores
        
        # Calculate aggregate metrics across all regimes
        all_returns = []
        all_trades = []
        all_drawdowns = []
        
        for results in regime_results.values():
            all_returns.extend(results.get('returns', []))
            all_trades.extend(results.get('trades', []))
            all_drawdowns.append(results.get('max_drawdown', 0.0))
        
        # Core performance metrics
        if all_returns:
            fitness_score.total_return = np.sum(all_returns)
            fitness_score.sharpe_ratio = np.mean(all_returns) / np.std(all_returns) if np.std(all_returns) > 0 else 0.0
            fitness_score.max_drawdown = max(all_drawdowns) if all_drawdowns else 0.0
        
        if all_trades:
            winning_trades = [t for t in all_trades if t.get('pnl', 0) > 0]
            fitness_score.win_rate = len(winning_trades) / len(all_trades) if all_trades else 0.0
            fitness_score.profit_factor = self._calculate_profit_factor({'trades': all_trades})
        
        # Advanced metrics
        if all_returns:
            fitness_score.sortino_ratio = self._calculate_sortino_ratio({'returns': all_returns})
            fitness_score.calmar_ratio = self._calculate_calmar_ratio({
                'total_return': fitness_score.total_return,
                'max_drawdown': fitness_score.max_drawdown
            })
            fitness_score.consistency_score = self._calculate_consistency_score({'returns': all_returns})
        
        # Calculate component scores
        fitness_score.returns_score = self._calculate_returns_fitness({
            'total_return': fitness_score.total_return,
            'sharpe_ratio': fitness_score.sharpe_ratio,
            'win_rate': fitness_score.win_rate
        })
        
        fitness_score.robustness_score = self._calculate_robustness_score({
            'sharpe_ratio': fitness_score.sharpe_ratio,
            'sortino_ratio': fitness_score.sortino_ratio,
            'max_drawdown': fitness_score.max_drawdown,
            'consistency_score': fitness_score.consistency_score
        })
        
        fitness_score.adaptability_score = self._calculate_adaptability_fitness(regime_results, genome)
        fitness_score.efficiency_score = self._calculate_efficiency_fitness({
            'trades': all_trades,
            'total_return': fitness_score.total_return
        })
        fitness_score.antifragility_score = self._calculate_antifragility_fitness({
            'trades': all_trades,
            'max_drawdown': fitness_score.max_drawdown,
            'regime_scores': regime_scores
        })
        
        # Calculate final weighted fitness
        weights = {
            'returns': 0.3,
            'robustness': 0.25,
            'adaptability': 0.2,
            'efficiency': 0.15,
            'antifragility': 0.1
        }
        
        fitness_score.total_fitness = (
            fitness_score.returns_score * weights['returns'] +
            fitness_score.robustness_score * weights['robustness'] +
            fitness_score.adaptability_score * weights['adaptability'] +
            fitness_score.efficiency_score * weights['efficiency'] +
            fitness_score.antifragility_score * weights['antifragility']
        )
        
        # Apply complexity penalty
        complexity_penalty = self._calculate_complexity_penalty(genome)
        fitness_score.total_fitness = max(0.0, fitness_score.total_fitness - complexity_penalty)
    
    def _run_simulation(self, genome, simulator, sensory_cortex) -> Optional[Dict]:
        """Run a complete simulation for fitness evaluation."""
        try:
            # Initialize simulation
            simulator.reset()
            
            # Simulation parameters
            max_steps = 10000
            step_count = 0
            trades = []
            returns = []
            equity_curve = []
            
            # Run simulation
            while step_count < max_steps:
                # Get market state
                market_state = simulator.step()
                if not market_state:
                    break
                
                # Get sensory reading
                sensory_reading = sensory_cortex.perceive(market_state)
                
                # Get genome decision
                action = genome.decide(sensory_reading)
                
                # Execute action
                if action:
                    trade_result = self._execute_action_with_costs(simulator, action, market_state)
                    if trade_result:
                        trades.append(trade_result)
                
                # Record metrics
                account_summary = simulator.get_account_summary()
                current_equity = account_summary['equity']
                equity_curve.append(current_equity)
                
                if len(equity_curve) > 1:
                    daily_return = (current_equity - equity_curve[-2]) / equity_curve[-2]
                    returns.append(daily_return)
                
                step_count += 1
            
            # Calculate final metrics
            if equity_curve:
                initial_equity = equity_curve[0]
                final_equity = equity_curve[-1]
                total_return = (final_equity - initial_equity) / initial_equity
                
                # Calculate drawdown
                peak = initial_equity
                max_drawdown = 0.0
                for equity in equity_curve:
                    if equity > peak:
                        peak = equity
                    drawdown = (peak - equity) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Calculate Sharpe ratio
                if returns and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
                
                # Calculate win rate
                if trades:
                    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
                    win_rate = len(winning_trades) / len(trades)
                else:
                    win_rate = 0.0
                
                return {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'trades': trades,
                    'returns': returns,
                    'equity_curve': equity_curve,
                    'sortino_ratio': self._calculate_sortino_ratio({'returns': returns}),
                    'consistency_score': self._calculate_consistency_score({'returns': returns})
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return None
    
    def _execute_action_with_costs(self, simulator, action, market_state) -> Optional[Dict[str, Any]]:
        """Execute a trading action with realistic costs."""
        try:
            # Calculate position size
            base_size = 10000  # Base position size
            position_size = base_size * action.size_factor
            
            # Execute action
            if action.action_type.value == 'buy':
                order_id = simulator.place_order(
                    symbol=market_state.symbol,
                    side='buy',
                    order_type='market',
                    quantity=position_size
                )
            elif action.action_type.value == 'sell':
                order_id = simulator.place_order(
                    symbol=market_state.symbol,
                    side='sell',
                    order_type='market',
                    quantity=position_size
                )
            elif action.action_type.value == 'close':
                # Close all positions
                positions = simulator.get_account_summary().get('positions', {})
                for symbol, position in positions.items():
                    if position['quantity'] != 0:
                        side = 'sell' if position['quantity'] > 0 else 'buy'
                        simulator.place_order(
                            symbol=symbol,
                            side=side,
                            order_type='market',
                            quantity=abs(position['quantity'])
                        )
            else:
                # HOLD or other actions
                return None
            
            # Get trade result
            account_summary = simulator.get_account_summary()
            
            return {
                'timestamp': market_state.timestamp,
                'action': action.action_type.value,
                'size': position_size,
                'price': market_state.mid_price,
                'pnl': account_summary.get('unrealized_pnl', 0.0),
                'equity': account_summary.get('equity', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return None
    
    def _calculate_returns_fitness(self, simulation_results: Dict) -> float:
        """Calculate returns-based fitness score."""
        total_return = simulation_results.get('total_return', 0.0)
        sharpe_ratio = simulation_results.get('sharpe_ratio', 0.0)
        win_rate = simulation_results.get('win_rate', 0.0)
        
        # Normalize metrics to 0-1 range
        normalized_return = min(max(total_return, -1.0), 1.0)  # Cap at ±100%
        normalized_sharpe = min(max(sharpe_ratio / 2.0, 0.0), 1.0)  # Cap at 2.0 Sharpe
        normalized_winrate = win_rate  # Already 0-1
        
        # Weighted combination
        returns_fitness = (
            normalized_return * 0.5 +
            normalized_sharpe * 0.3 +
            normalized_winrate * 0.2
        )
        
        return max(0.0, returns_fitness)
    
    def _calculate_adaptability_fitness(self, regime_results: Dict, genome) -> float:
        """Calculate adaptability fitness based on performance across regimes."""
        if not regime_results:
            return 0.0
        
        # Calculate performance variance across regimes
        regime_performances = []
        for regime_name, results in regime_results.items():
            regime_fitness = self._calculate_regime_fitness(results, genome)
            regime_performances.append(regime_fitness)
        
        if not regime_performances:
            return 0.0
        
        # Adaptability is inversely related to performance variance
        performance_std = np.std(regime_performances)
        performance_mean = np.mean(regime_performances)
        
        if performance_mean == 0:
            return 0.0
        
        # Coefficient of variation (lower is better)
        cv = performance_std / performance_mean
        adaptability_score = 1.0 / (1.0 + cv)
        
        return adaptability_score
    
    def _calculate_efficiency_fitness(self, simulation_results: Dict) -> float:
        """Calculate efficiency fitness based on cost management."""
        trades = simulation_results.get('trades', [])
        total_return = simulation_results.get('total_return', 0.0)
        
        if not trades:
            return 0.0
        
        # Calculate efficiency metrics
        trade_count = len(trades)
        avg_trade_size = np.mean([abs(t.get('size', 0)) for t in trades]) if trades else 0
        
        # Efficiency score (fewer trades, larger sizes, higher returns)
        efficiency_score = (
            (1.0 / (1.0 + trade_count / 100)) * 0.4 +  # Fewer trades
            min(avg_trade_size / 10000, 1.0) * 0.3 +   # Larger sizes
            max(min(total_return, 1.0), 0.0) * 0.3     # Higher returns
        )
        
        return efficiency_score
    
    def _calculate_antifragility_fitness(self, simulation_results: Dict) -> float:
        """Calculate antifragility fitness based on stress resilience."""
        trades = simulation_results.get('trades', [])
        max_drawdown = simulation_results.get('max_drawdown', 0.0)
        regime_scores = simulation_results.get('regime_scores', {})
        
        # Antifragility components
        drawdown_resilience = 1.0 - max_drawdown
        
        # Crisis performance (if available)
        crisis_performance = 0.0
        if 'crisis' in regime_scores:
            crisis_performance = regime_scores['crisis']
        elif 'synthetic_crisis' in regime_scores:
            crisis_performance = regime_scores['synthetic_crisis']
        
        # Volatility performance
        volatility_performance = 0.0
        if 'volatile' in regime_scores:
            volatility_performance = regime_scores['volatile']
        
        # Combine into antifragility score
        antifragility_score = (
            drawdown_resilience * 0.4 +
            crisis_performance * 0.3 +
            volatility_performance * 0.3
        )
        
        return max(0.0, antifragility_score)
    
    def _update_fitness_distribution(self, fitness_score: FitnessScore):
        """Update fitness distribution statistics."""
        metrics = [
            'total_fitness', 'returns_score', 'robustness_score',
            'adaptability_score', 'efficiency_score', 'antifragility_score'
        ]
        
        for metric in metrics:
            if metric not in self.fitness_distribution:
                self.fitness_distribution[metric] = []
            
            value = getattr(fitness_score, metric, 0.0)
            self.fitness_distribution[metric].append(value)
    
    def get_fitness_distribution(self) -> Dict:
        """Get fitness distribution statistics."""
        distribution_stats = {}
        
        for metric, values in self.fitness_distribution.items():
            if values:
                distribution_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
        
        return distribution_stats 