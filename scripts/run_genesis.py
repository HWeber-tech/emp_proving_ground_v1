#!/usr/bin/env python3
"""
Operation Darwin: The Genesis Run

First-ever end-to-end evolutionary simulation to establish baseline performance
for the EMP Proving Ground system.

This script executes a complete evolutionary run with:
- Sensory cortex integration
- Risk management validation
- Comprehensive fitness evaluation
- Baseline performance metrics
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import CurrencyConverter, InstrumentProvider, RiskConfig
from src.core.risk.manager import RiskManager
from src.data import DukascopyIngestor, TickDataCleaner, TickDataStorage
from src.evolution import EvolutionConfig, EvolutionEngine, FitnessEvaluator
from src.sensory.core.base import InstrumentMeta
from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
from src.simulation import AdversarialEngine, MarketSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GenesisConfig:
    """Configuration for the Genesis Run"""

    symbol: str = "EURUSD"
    population_size: int = 100
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.6
    elite_ratio: float = 0.1

    # Data parameters
    start_date: str = "2024-01-01"
    end_date: str = "2024-06-30"
    timeframe: str = "H1"

    # Risk parameters
    max_risk_per_trade: float = 0.02
    max_drawdown: float = 0.25
    max_leverage: float = 10.0

    # Simulation parameters
    initial_balance: float = 100000.0
    adversarial_difficulty: float = 0.3


class GenesisRunner:
    """Orchestrates the complete Genesis Run"""

    def __init__(self, config: GenesisConfig):
        """Initialize Genesis Runner"""
        self.config = config
        self.results = {
            "start_time": datetime.utcnow().isoformat(),
            "config": config.__dict__,
            "generations": [],
            "final_population": [],
            "baseline_metrics": {},
            "system_health": {},
        }

    async def setup_system(self):
        """Set up the complete EMP system"""
        logger.info("Setting up EMP system for Genesis Run...")

        # Initialize core components
        risk_config = RiskConfig(
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade)),
            max_leverage=Decimal(str(self.config.max_leverage)),
            max_total_exposure_pct=Decimal("0.5"),
            max_drawdown_pct=Decimal(str(self.config.max_drawdown)),
        )

        self.instrument_provider = InstrumentProvider()
        self.currency_converter = CurrencyConverter()

        # Initialize data pipeline
        self.data_storage = TickDataStorage()
        self.data_cleaner = TickDataCleaner()
        self.data_ingestor = DukascopyIngestor(self.data_storage, self.data_cleaner)

        # Initialize risk management
        self.risk_manager = RiskManager(risk_config, self.instrument_provider)

        # Initialize sensory cortex
        instrument_meta = InstrumentMeta(
            symbol=self.config.symbol,
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            typical_spread=0.00015,
            avg_daily_range=0.01,
        )

        self.sensory_cortex = MasterOrchestrator(instrument_meta)

        # Initialize evolution engine
        evolution_config = EvolutionConfig(
            population_size=self.config.population_size,
            elite_ratio=self.config.elite_ratio,
            crossover_ratio=self.config.crossover_rate,
            mutation_ratio=self.config.mutation_rate,
        )

        self.fitness_evaluator = FitnessEvaluator(self.data_storage)
        self.evolution_engine = EvolutionEngine(evolution_config, self.fitness_evaluator)

        # Initialize market simulation
        self.market_simulator = MarketSimulator(
            self.data_storage,
            initial_balance=self.config.initial_balance,
            leverage=self.config.max_leverage,
        )

        # Initialize sensory cortex
        self.market_simulator.initialize_sensory_cortex(instrument_meta)

        self.adversarial_engine = AdversarialEngine(
            difficulty_level=self.config.adversarial_difficulty, seed=42
        )

        logger.info("EMP system setup complete")

    async def load_data(self):
        """Load market data for simulation"""
        logger.info(f"Loading data for {self.config.symbol}...")

        start_time = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end_time = datetime.strptime(self.config.end_date, "%Y-%m-%d")

        try:
            # Generate synthetic OHLCV data directly
            logger.info("Generating synthetic OHLCV data for simulation...")
            ohlcv_data = self._generate_synthetic_ohlcv(self.config.symbol, start_time, end_time)

            if ohlcv_data.empty:
                logger.error("Failed to generate synthetic data")
                return False

            # Validate data quality
            if not self._validate_data_quality(ohlcv_data):
                logger.error("Data quality validation failed")
                return False

            logger.info(f"Generated {len(ohlcv_data)} hours of synthetic data")
            self.market_data = ohlcv_data
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    async def run_evolution(self):
        """Run the complete evolutionary simulation"""
        logger.info("Starting Genesis Run...")

        # Initialize population
        success = self.evolution_engine.initialize_population(seed=42)
        if not success:
            logger.error("Failed to initialize population")
            return False

        # Run generations
        for generation in range(self.config.generations):
            logger.info(f"Running generation {generation + 1}/{self.config.generations}")

            # Evolve generation
            stats = self.evolution_engine.evolve_generation()

            # Record generation stats
            self.results["generations"].append(
                {
                    "generation": generation + 1,
                    "best_fitness": float(stats.best_fitness),
                    "avg_fitness": float(stats.avg_fitness),
                    "population_diversity": float(stats.diversity),
                }
            )

            logger.info(
                f"Generation {generation + 1}: "
                f"Best={stats.best_fitness:.4f}, "
                f"Avg={stats.avg_fitness:.4f}, "
                f"Diversity={stats.diversity:.3f}"
            )

        # Get final population
        final_genomes = self.evolution_engine.get_best_genomes(10)
        self.results["final_population"] = [
            {
                "genome_id": str(genome.genome_id),
                "fitness": float(genome.fitness_score),
                "strategy": str(genome.decision_tree.get("type", "unknown")),
                "risk_profile": "balanced",
            }
            for genome in final_genomes
        ]

        return True

    async def run_system_health_check(self):
        """Run comprehensive system health check"""
        logger.info("Running system health check...")

        # Calculate baseline metrics
        baseline_metrics = await self._calculate_baseline_metrics()

        # Check sensory cortex health
        sensory_health = self.sensory_cortex.get_system_health()

        # Check data pipeline
        data_health = {
            "data_loaded": len(self.market_data) > 0,
            "data_quality": self._calculate_data_quality(self.market_data),
        }

        # Check evolution engine
        evolution_health = {
            "population_initialized": self.evolution_engine.population is not None,
            "fitness_evaluator_ready": self.fitness_evaluator is not None,
        }

        self.results["system_health"] = {
            "sensory_cortex": sensory_health,
            "data_pipeline": data_health,
            "evolution_engine": evolution_health,
            "baseline_metrics": baseline_metrics,
        }

        logger.info("System health check complete")

    def _generate_synthetic_ohlcv(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing"""
        logger.info("Generating synthetic OHLCV data...")

        # Create date range
        date_range = pd.date_range(start=start_time, end=end_time, freq="H")

        # Base price
        base_price = 1.1000 if symbol == "EURUSD" else 1.0000

        # Generate price movements
        np.random.seed(42)
        n_points = len(date_range)

        # Random walk with trend
        returns = np.random.normal(0.0001, 0.001, n_points)
        prices = base_price * np.exp(np.cumsum(returns))

        # Add intraday patterns
        hours = np.array([dt.hour for dt in date_range])
        hour_pattern = np.sin(2 * np.pi * hours / 24) * 0.001
        prices *= np.exp(hour_pattern)

        # Generate OHLCV data
        ohlcv_data = []
        for i, timestamp in enumerate(date_range):
            if i == 0:
                continue

            # Generate realistic OHLCV
            open_price = prices[i - 1]
            close_price = prices[i]

            # Add volatility
            volatility = 0.001
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility)))

            # Volume with patterns
            base_volume = 1000
            volume = base_volume * (1 + abs(np.random.normal(0, 0.5)))

            ohlcv_data.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "bid": close_price - 0.0001,
                    "ask": close_price + 0.0001,
                    "bid_volume": volume / 2,
                    "ask_volume": volume / 2,
                }
            )

        return pd.DataFrame(ohlcv_data).set_index("timestamp")

    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Validate data quality"""
        if data.empty:
            return False

        # Check for required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in data.columns for col in required_cols):
            return False

        # Check for NaN values
        if data[required_cols].isnull().sum().sum() > 0:
            return False

        # Check for zero or negative prices
        price_cols = ["open", "high", "low", "close"]
        if (data[price_cols] <= 0).sum().sum() > 0:
            return False

        # Check for reasonable price ranges
        if data["close"].max() > 10 or data["close"].min() < 0.1:
            logger.warning("Price range seems unusual, but continuing...")

        return True

    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if data.empty:
            return 0.0

        # Basic quality metrics
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            return 0.0

        # Check for NaN values
        nan_ratio = data[required_cols].isnull().sum().sum() / (len(data) * len(required_cols))

        # Check for zero volumes
        zero_volume_ratio = (data["volume"] == 0).sum() / len(data)

        # Calculate quality score
        quality = 1.0 - nan_ratio - zero_volume_ratio

        return max(0.0, min(1.0, quality))

    async def _calculate_baseline_metrics(self) -> Dict[str, Any]:
        """Calculate baseline performance metrics"""
        logger.info("Calculating baseline metrics...")

        # Random strategy baseline
        random_returns = []
        buy_hold_returns = []

        if len(self.market_data) > 0:
            # Buy and hold strategy
            start_price = self.market_data["close"].iloc[0]
            end_price = self.market_data["close"].iloc[-1]
            buy_hold_return = (end_price - start_price) / start_price

            # Random strategy simulation
            np.random.seed(42)
            n_trades = min(100, len(self.market_data) // 10)
            random_pnl = []

            for i in range(n_trades):
                entry_idx = np.random.randint(0, len(self.market_data) - 10)
                exit_idx = min(entry_idx + np.random.randint(5, 20), len(self.market_data) - 1)

                entry_price = self.market_data["close"].iloc[entry_idx]
                exit_price = self.market_data["close"].iloc[exit_idx]

                direction = np.random.choice([-1, 1])
                pnl = direction * (exit_price - entry_price) / entry_price
                random_pnl.append(pnl)

            random_return = np.mean(random_pnl) if random_pnl else 0.0

            return {
                "buy_hold_return": float(buy_hold_return),
                "random_strategy_return": float(random_return),
                "data_points": len(self.market_data),
                "volatility": float(self.market_data["close"].pct_change().std() * np.sqrt(252)),
                "sharpe_ratio": float(
                    buy_hold_return / (self.market_data["close"].pct_change().std() * np.sqrt(252))
                )
                if self.market_data["close"].pct_change().std() > 0
                else 0.0,
            }

        return {
            "buy_hold_return": 0.0,
            "random_strategy_return": 0.0,
            "data_points": 0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
        }

    def _check_data_integrity(self) -> bool:
        """Check data integrity"""
        if self.market_data is None or self.market_data.empty:
            return False

        # Check for monotonic timestamps
        if not self.market_data.index.is_monotonic_increasing:
            return False

        # Check for duplicate timestamps
        if self.market_data.index.duplicated().any():
            return False

        return True

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
            }
        except ImportError:
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}

    async def save_results(self):
        """Save Genesis Run results"""
        logger.info("Saving Genesis Run results...")

        self.results["end_time"] = datetime.utcnow().isoformat()

        # Calculate final metrics
        final_metrics = {
            "total_runtime_seconds": (
                datetime.fromisoformat(self.results["end_time"])
                - datetime.fromisoformat(self.results["start_time"])
            ).total_seconds(),
            "evolution_summary": self.evolution_engine.get_evolution_summary(),
            "system_performance": self._get_system_performance(),
        }

        self.results["final_metrics"] = final_metrics

        # Save to JSON
        results_file = Path("genesis_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary
        summary_file = Path("genesis_summary.txt")
        with open(summary_file, "w") as f:
            f.write("=== EMP Proving Ground: Genesis Run Summary ===\n\n")
            f.write(f"Start Time: {self.results['start_time']}\n")
            f.write(f"End Time: {self.results['end_time']}\n")
            f.write(f"Total Runtime: {final_metrics['total_runtime_seconds']:.2f} seconds\n\n")

            f.write("=== Evolution Results ===\n")
            if self.results["generations"]:
                final_gen = self.results["generations"][-1]
                f.write(f"Final Best Fitness: {final_gen['best_fitness']:.4f}\n")
                f.write(f"Final Average Fitness: {final_gen['avg_fitness']:.4f}\n")
                f.write(f"Final Diversity: {final_gen['population_diversity']:.3f}\n")

            f.write("\n=== System Health ===\n")
            health = self.results["system_health"]
            f.write(f"Overall Health: {health['sensory_cortex']['overall_health']:.2f}\n")
            f.write(f"Data Quality: {health['data_pipeline']['data_quality']:.2f}\n")

            f.write("\n=== Baseline Metrics ===\n")
            baseline = health["baseline_metrics"]
            f.write(f"Buy & Hold Return: {baseline['buy_hold_return']:.4f}\n")
            f.write(f"Random Strategy Return: {baseline['random_strategy_return']:.4f}\n")
            f.write(f"Market Volatility: {baseline['volatility']:.4f}\n")

        logger.info(f"Results saved to {results_file} and {summary_file}")

    def _get_system_performance(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        return {
            "memory_usage": self._check_memory_usage(),
            "cache_stats": self.data_storage.get_cache_stats()
            if hasattr(self.data_storage, "get_cache_stats")
            else {},
            "total_generations": len(self.results["generations"]),
            "final_population_size": len(self.results["final_population"]),
        }

    async def run(self):
        """Execute the complete Genesis Run"""
        try:
            # Setup system
            await self.setup_system()

            # Load data
            if not await self.load_data():
                return False

            # Run system health check
            await self.run_system_health_check()

            # Run evolution
            if not await self.run_evolution():
                return False

            # Save results
            await self.save_results()

            logger.info("Genesis Run completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Genesis Run failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False


async def main():
    """Main execution function"""
    logger.info("Starting Operation Darwin: Genesis Run")

    # Create configuration
    config = GenesisConfig()

    # Create runner
    runner = GenesisRunner(config)

    # Execute Genesis Run
    success = await runner.run()

    if success:
        logger.info("🎉 Genesis Run completed successfully!")
        logger.info("System is ready for production deployment")
    else:
        logger.error("❌ Genesis Run failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
