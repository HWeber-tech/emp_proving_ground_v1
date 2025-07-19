"""
Simulation Orchestrator v1.0 - High-Fidelity Simulation Envelope

Implements SIM-01 ticket requirements for comprehensive simulation management.
Orchestrates market simulation, execution, and fitness evaluation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from .market_simulator import MarketSimulator
from .execution.simulation_execution_engine import SimulationExecutionEngine
from ..core.event_bus import EventBus
from ..core.events import TradeIntent, FitnessReport, SimulationComplete
from ..sensory.integration.sensory_cortex import SensoryCortex
from ..evolution.engine.genetic_engine import GeneticEngine
from ..evolution.fitness.trading_fitness_evaluator import TradingFitnessEvaluator

logger = logging.getLogger(__name__)


class SimulationOrchestrator:
    """
    Central orchestrator for the high-fidelity simulation envelope.
    Manages market simulation, trade execution, and fitness evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation orchestrator.
        
        Args:
            config: Configuration dictionary for simulation parameters
        """
        self.config = config
        self.event_bus = EventBus()
        self.sensory_cortex = SensoryCortex(self.event_bus)
        self.market_simulator = MarketSimulator(self.event_bus, self.sensory_cortex)
        self.execution_engine = SimulationExecutionEngine(
            self.event_bus,
            initial_capital=Decimal(str(config.get('initial_capital', 100000)))
        )
        self.genetic_engine = GeneticEngine(self.event_bus)
        self.fitness_evaluator = TradingFitnessEvaluator()
        
        self.is_running = False
        self.current_generation = 0
        self.simulation_results: List[Dict[str, Any]] = []
        
    async def initialize(self) -> bool:
        """
        Initialize all simulation components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load historical data
            data_path = self.config.get('historical_data_path', 'data/historical_market_data.csv')
            if not await self.market_simulator.load_data(data_path):
                logger.error("Failed to load historical data")
                return False
                
            # Initialize components
            await self.execution_engine.start()
            await self.sensory_cortex.initialize()
            await self.genetic_engine.initialize()
            
            logger.info("Simulation orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing simulation orchestrator: {e}")
            return False
            
    async def run_simulation(self, generations: int = 1) -> Dict[str, Any]:
        """
        Run the complete simulation for specified generations.
        
        Args:
            generations: Number of generations to simulate
            
        Returns:
            Dictionary with simulation results
        """
        if not self.is_running:
            await self.start()
            
        try:
            logger.info(f"Starting simulation for {generations} generations")
            
            for generation in range(generations):
                self.current_generation = generation
                logger.info(f"Running generation {generation + 1}/{generations}")
                
                # Reset execution engine for new generation
                self.execution_engine = SimulationExecutionEngine(
                    self.event_bus,
                    initial_capital=Decimal(str(self.config.get('initial_capital', 100000)))
                )
                await self.execution_engine.start()
                
                # Run market simulation
                await self.market_simulator.run()
                
                # Wait for simulation to complete
                await self._wait_for_simulation_complete()
                
                # Evaluate fitness
                fitness_score = await self._evaluate_fitness()
                
                # Store results
                result = {
                    'generation': generation,
                    'fitness_score': float(fitness_score),
                    'portfolio_summary': self.execution_engine.get_portfolio_summary(),
                    'trade_history': self.execution_engine.get_trade_history()
                }
                self.simulation_results.append(result)
                
                # Publish fitness report
                await self._publish_generation_fitness(fitness_score, generation)
                
            await self.stop()
            
            return {
                'success': True,
                'generations': generations,
                'results': self.simulation_results,
                'summary': self._generate_summary()
            }
            
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            await self.stop()
            return {
                'success': False,
                'error': str(e),
                'results': self.simulation_results
            }
            
    async def start(self) -> None:
        """Start the simulation orchestrator."""
        self.is_running = True
        logger.info("Simulation orchestrator started")
        
    async def stop(self) -> None:
        """Stop the simulation orchestrator."""
        self.is_running = False
        await self.execution_engine.stop()
        logger.info("Simulation orchestrator stopped")
        
    async def _wait_for_simulation_complete(self) -> None:
        """Wait for market simulation to complete."""
        max_wait = self.config.get('max_simulation_wait', 300)  # 5 minutes
        start_time = datetime.now()
        
        while self.market_simulator.is_running:
            if (datetime.now() - start_time).total_seconds() > max_wait:
                logger.warning("Simulation timeout reached")
                await self.market_simulator.stop()
                break
            await asyncio.sleep(0.1)
            
    async def _evaluate_fitness(self) -> Decimal:
        """Evaluate fitness based on portfolio performance."""
        portfolio_summary = self.execution_engine.get_portfolio_summary()
        
        # Calculate fitness score based on multiple metrics
        final_value = Decimal(str(portfolio_summary['total_value']))
        initial_capital = Decimal(str(self.config.get('initial_capital', 100000)))
        
        # Basic fitness: total return
        total_return = (final_value - initial_capital) / initial_capital
        
        # Additional metrics
        trade_history = self.execution_engine.get_trade_history()
        total_trades = len(trade_history)
        
        # Penalty for excessive trading
        trading_penalty = Decimal('0.001') * Decimal(str(total_trades))
        
        # Final fitness score
        fitness_score = total_return - trading_penalty
        
        return max(fitness_score, Decimal('-1.0'))  # Cap at -100%
        
    async def _publish_generation_fitness(self, fitness_score: Decimal, generation: int) -> None:
        """Publish fitness report for a generation."""
        from ..core.events import EvolutionEvent
        
        evolution_event = EvolutionEvent(
            event_id=f"evolution_{generation}_{datetime.now().isoformat()}",
            timestamp=datetime.now(),
            source="SimulationOrchestrator",
            generation=generation,
            population_size=1,  # Single genome per generation in this simulation
            best_fitness=fitness_score,
            average_fitness=fitness_score,
            diversity_score=Decimal('0'),  # No diversity in single genome
            metadata={
                'portfolio_summary': self.execution_engine.get_portfolio_summary(),
                'total_trades': len(self.execution_engine.get_trade_history())
            }
        )
        
        await self.event_bus.publish(evolution_event)
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate simulation summary."""
        if not self.simulation_results:
            return {'message': 'No simulation results available'}
            
        best_generation = max(self.simulation_results, key=lambda x: x['fitness_score'])
        worst_generation = min(self.simulation_results, key=lambda x: x['fitness_score'])
        
        return {
            'total_generations': len(self.simulation_results),
            'best_fitness': best_generation['fitness_score'],
            'worst_fitness': worst_generation['fitness_score'],
            'average_fitness': sum(r['fitness_score'] for r in self.simulation_results) / len(self.simulation_results),
            'best_generation': best_generation,
            'worst_generation': worst_generation
        }
        
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get current simulation configuration."""
        return {
            'initial_capital': float(self.config.get('initial_capital', 100000)),
            'historical_data_path': self.config.get('historical_data_path', 'data/historical_market_data.csv'),
            'slippage_rate': float(self.execution_engine.slippage_rate),
            'commission_rate': float(self.execution_engine.commission_rate),
            'minimum_commission': float(self.execution_engine.minimum_commission)
        }
        
    async def run_single_backtest(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single backtest with specific strategy configuration.
        
        Args:
            strategy_config: Strategy configuration for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Running single backtest")
        
        # Configure strategy
        self.config.update(strategy_config)
        
        # Run simulation
        return await self.run_simulation(generations=1)
        
    def export_results(self, output_path: str) -> bool:
        """
        Export simulation results to file.
        
        Args:
            output_path: Path to save results
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            import json
            
            output_data = {
                'config': self.get_simulation_config(),
                'results': self.simulation_results,
                'summary': self._generate_summary(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
                
            logger.info(f"Simulation results exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
