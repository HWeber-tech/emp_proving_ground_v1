"""
Market GAN System
Generative Adversarial Network for creating challenging market scenarios.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
import uuid

try:
    from src.core.events import MarketScenario, StrategyTestResult  # legacy
except Exception:  # pragma: no cover
    MarketScenario = StrategyTestResult = object  # type: ignore
from src.thinking.prediction.predictive_market_modeler import PredictiveMarketModeler
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


class MarketDataGenerator:
    """Generator network for creating synthetic market scenarios."""
    
    def __init__(self):
        self.difficulty_levels = {
            'easy': {'volatility': 0.01, 'noise': 0.02},
            'medium': {'volatility': 0.02, 'noise': 0.04},
            'hard': {'volatility': 0.04, 'noise': 0.08},
            'extreme': {'volatility': 0.08, 'noise': 0.15}
        }
        
    async def generate_scenarios(
        self,
        difficulty_level: str,
        target_strategies: List[str],
        num_scenarios: int = 100
    ) -> List[MarketScenario]:
        """Generate challenging market scenarios."""
        try:
            scenarios = []
            difficulty = self.difficulty_levels.get(difficulty_level, self.difficulty_levels['medium'])
            
            for i in range(num_scenarios):
                # Generate synthetic market data
                scenario = await self._create_challenging_scenario(
                    difficulty,
                    target_strategies
                )
                scenarios.append(scenario)
            
            logger.info(
                f"Generated {len(scenarios)} {difficulty_level} scenarios "
                f"targeting {len(target_strategies)} strategies"
            )
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating scenarios: {e}")
            return []
    
    async def _create_challenging_scenario(
        self,
        difficulty: Dict[str, float],
        target_strategies: List[str]
    ) -> MarketScenario:
        """Create a challenging scenario for target strategies."""
        try:
            # Generate synthetic price data
            base_price = 100.0
            volatility = difficulty['volatility']
            noise_level = difficulty['noise']
            
            # Create price path with challenging patterns
            price_path = self._generate_challenging_price_path(
                base_price,
                volatility,
                noise_level
            )
            
            scenario = MarketScenario(
                scenario_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                scenario_type='adversarial',
                probability=1.0,
                price_path=price_path,
                volatility=volatility,
                direction_bias=0.0,
                confidence=Decimal('0.8')
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error creating challenging scenario: {e}")
            return MarketScenario(
                scenario_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                scenario_type='error',
                probability=0.0,
                price_path=[100.0],
                volatility=0.02,
                direction_bias=0.0,
                confidence=Decimal('0.5')
            )
    
    def _generate_challenging_price_path(
        self,
        base_price: float,
        volatility: float,
        noise_level: float
    ) -> List[float]:
        """Generate a price path designed to challenge strategies."""
        try:
            # Create a path with multiple regime changes
            days = 30
            prices = [base_price]
            
            for day in range(days):
                # Add regime changes
                if day % 7 == 0:
                    # Sudden regime shift
                    regime_change = np.random.normal(0, volatility * 3)
                else:
                    regime_change = 0
                
                # Add noise
                noise = np.random.normal(0, noise_level)
                
                # Calculate new price
                drift = regime_change + noise
                new_price = prices[-1] * (1 + drift)
                
                # Ensure price stays positive
                new_price = max(new_price, base_price * 0.5)
                
                prices.append(new_price)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error generating challenging price path: {e}")
            return [base_price] * 30
    
    async def train_generator(
        self,
        survival_results: List[StrategyTestResult],
        target_failure_rate: float = 0.3
    ) -> bool:
        """Train the generator to create more challenging scenarios."""
        try:
            # Analyze survival rates
            survival_rates = [r.survival_rate for r in survival_results]
            avg_survival = np.mean(survival_rates) if survival_rates else 0.5
            
            # Adjust difficulty based on target failure rate
            if avg_survival > (1 - target_failure_rate):
                # Increase difficulty
                logger.info(
                    f"Increasing generator difficulty: "
                    f"Current survival {avg_survival:.2f}, "
                    f"Target failure {target_failure_rate:.2f}"
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error training generator: {e}")
            return False


class StrategyTester:
    """Discriminator network that tests strategies against scenarios."""
    
    def __init__(self):
        self.test_results = []
        
    async def test_strategies(
        self,
        strategy_population: List[str],
        synthetic_scenarios: List[MarketScenario]
    ) -> List[StrategyTestResult]:
        """Test strategies against synthetic scenarios."""
        try:
            results = []
            
            for strategy_id in strategy_population:
                # Test strategy against all scenarios
                survival_count = 0
                total_scenarios = len(synthetic_scenarios)
                
                for scenario in synthetic_scenarios:
                    # Simulate strategy performance
                    survival = await self._simulate_strategy_performance(
                        strategy_id,
                        scenario
                    )
                    
                    if survival:
                        survival_count += 1
                
                # Calculate survival rate
                survival_rate = survival_count / total_scenarios if total_scenarios > 0 else 0
                
                result = StrategyTestResult(
                    strategy_id=strategy_id,
                    scenario_id=scenario.scenario_id,
                    survival_rate=survival_rate,
                    performance_score=self._calculate_performance_score(survival_rate),
                    timestamp=datetime.utcnow()
                )
                
                results.append(result)
            
            logger.info(
                f"Tested {len(strategy_population)} strategies "
                f"against {len(synthetic_scenarios)} scenarios"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing strategies: {e}")
            return []
    
    async def _simulate_strategy_performance(
        self,
        strategy_id: str,
        scenario: MarketScenario
    ) -> bool:
        """Simulate strategy performance against scenario."""
        try:
            # Simple simulation based on scenario characteristics
            volatility = scenario.volatility
            
            # Higher volatility makes survival harder
            survival_threshold = 0.03  # 3% volatility threshold
            
            return volatility <= survival_threshold
            
        except Exception as e:
            logger.error(f"Error simulating strategy performance: {e}")
            return False
    
    def _calculate_performance_score(self, survival_rate: float) -> float:
        """Calculate performance score from survival rate."""
        try:
            # Convert survival rate to performance score
            return max(0.0, min(1.0, survival_rate))
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0


class AdversarialTrainer:
    """Trains both generator and discriminator in adversarial fashion."""
    
    def __init__(self):
        self.training_history = []
        
    async def train_generator(
        self,
        generator: MarketDataGenerator,
        survival_results: List[StrategyTestResult],
        target_failure_rate: float
    ) -> bool:
        """Train the generator to create more challenging scenarios."""
        return await generator.train_generator(survival_results, target_failure_rate)
    
    async def train_discriminator(
        self,
        strategy_population: List[str],
        synthetic_scenarios: List[MarketScenario],
        survival_results: List[StrategyTestResult]
    ) -> List[str]:
        """Train strategies to survive adversarial scenarios."""
        try:
            # Identify weak strategies
            weak_strategies = [
                r.strategy_id for r in survival_results
                if r.survival_rate < 0.5
            ]
            
            # Improve weak strategies
            improved_strategies = []
            
            for strategy_id in weak_strategies:
                # Create improved version
                improved_strategy = await self._improve_strategy(
                    strategy_id,
                    survival_results
                )
                
                if improved_strategy:
                    improved_strategies.append(improved_strategy)
            
            logger.info(
                f"Improved {len(improved_strategies)} weak strategies"
            )
            
            return improved_strategies
            
        except Exception as e:
            logger.error(f"Error training discriminator: {e}")
            return []
    
    async def _improve_strategy(
        self,
        strategy_id: str,
        survival_results: List[StrategyTestResult]
    ) -> Optional[str]:
        """Improve a strategy based on test results."""
        try:
            # Find strategy's test results
            strategy_results = [
                r for r in survival_results
                if r.strategy_id == strategy_id
            ]
            
            if not strategy_results:
                return None
            
            # Calculate average survival rate
            avg_survival = np.mean([r.survival_rate for r in strategy_results])
            
            # Create improved strategy ID
            improved_id = f"{strategy_id}_improved_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            logger.debug(
                f"Improved strategy {strategy_id} -> {improved_id} "
                f"(survival: {avg_survival:.2f})"
            )
            
            return improved_id
            
        except Exception as e:
            logger.error(f"Error improving strategy: {e}")
            return None


class ScenarioValidator:
    """Validates that synthetic scenarios are realistic."""
    
    def __init__(self):
        self.validation_rules = {
            'volatility_range': (0.001, 0.2),
            'price_range': (0.1, 10.0),
            'trend_consistency': 0.8
        }
    
    async def validate_realism(
        self,
        synthetic_scenarios: List[MarketScenario],
        real_market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that synthetic scenarios are realistic."""
        try:
            validation_results = {
                'valid': 0,
                'invalid': 0,
                'total': len(synthetic_scenarios),
                'validation_score': 0.0
            }
            
            for scenario in synthetic_scenarios:
                is_valid = await self._validate_scenario(scenario, real_market_data)
                
                if is_valid:
                    validation_results['valid'] += 1
                else:
                    validation_results['invalid'] += 1
            
            # Calculate validation score
            if validation_results['total'] > 0:
                validation_results['validation_score'] = (
                    validation_results['valid'] / validation_results['total']
                )
            
            logger.info(
                f"Scenario validation: "
                f"{validation_results['valid']}/{validation_results['total']} valid "
                f"({validation_results['validation_score']:.2f})"
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating scenarios: {e}")
            return {'valid': 0, 'invalid': 0, 'total': 0, 'validation_score': 0.0}
    
    async def _validate_scenario(
        self,
        scenario: MarketScenario,
        real_market_data: Dict[str, Any]
    ) -> bool:
        """Validate a single scenario."""
        try:
            # Check volatility range
            min_vol, max_vol = self.validation_rules['volatility_range']
            if not (min_vol <= scenario.volatility <= max_vol):
                return False
            
            # Check price path validity
            if not scenario.price_path or len(scenario.price_path) < 2:
                return False
            
            # Check for reasonable price movements
            price_changes = [
                abs(scenario.price_path[i] - scenario.price_path[i-1]) /
                scenario.price_path[i-1]
                for i in range(1, len(scenario.price_path))
            ]
            
            max_change = max(price_changes) if price_changes else 0
            if max_change > 0.2:  # More than 20% daily change is unrealistic
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating scenario: {e}")
            return False


class MarketGAN:
    """
    Generative Adversarial Network for training strategies against challenging scenarios.
    
    Features:
    - Generator creates challenging market scenarios
    - Discriminator tests strategies against scenarios
    - Adversarial training improves both components
    - Realistic scenario validation
    """
    
    def __init__(self, state_store: StateStore):
        self.state_store = state_store
        self.generator = MarketDataGenerator()
        self.discriminator = StrategyTester()
        self.adversarial_trainer = AdversarialTrainer()
        self.scenario_validator = ScenarioValidator()
        
        self._training_history_key = "emp:gan_training_history"
        
    async def train_adversarial_strategies(
        self,
        strategy_population: List[str],
        num_epochs: int = 10
    ) -> List[str]:
        """
        Train strategies using adversarial training.
        
        Args:
            strategy_population: List of strategy IDs to train
            num_epochs: Number of training epochs
            
        Returns:
            List of improved strategies
        """
        try:
            improved_strategies = []
            
            for epoch in range(num_epochs):
                logger.info(f"GAN Training Epoch {epoch + 1}/{num_epochs}")
                
                # Step 1: Generator creates challenging scenarios
                difficulty = self._get_difficulty_for_epoch(epoch, num_epochs)
                synthetic_scenarios = await self.generator.generate_scenarios(
                    difficulty,
                    strategy_population,
                    num_scenarios=100
                )
                
                if not synthetic_scenarios:
                    continue
                
                # Step 2: Discriminator tests strategies
                survival_results = await self.discriminator.test_strategies(
                    strategy_population,
                    synthetic_scenarios
                )
                
                # Step 3: Train generator to create more challenging scenarios
                await self.adversarial_trainer.train_generator(
                    self.generator,
                    survival_results,
                    target_failure_rate=0.3
                )
                
                # Step 4: Train discriminator (strategies) to survive
                improved = await self.adversarial_trainer.train_discriminator(
                    strategy_population,
                    synthetic_scenarios,
                    survival_results
                )
                
                improved_strategies.extend(improved)
                
                # Step 5: Validate scenarios are realistic
                real_market_data = await self._get_real_market_data()
                validation = await self.scenario_validator.validate_realism(
                    synthetic_scenarios,
                    real_market_data
                )
                
                # Store training results
                await self._store_training_results(
                    epoch,
                    survival_results,
                    validation,
                    improved
                )
            
            logger.info(
                f"GAN training complete: {len(improved_strategies)} strategies improved"
            )
            
            return improved_strategies
            
        except Exception as e:
            logger.error(f"Error in adversarial training: {e}")
            return []
    
    def _get_difficulty_for_epoch(self, epoch: int, total_epochs: int) -> str:
        """Get difficulty level for training epoch."""
        try:
            progress = epoch / total_epochs
            
            if progress < 0.25:
                return 'easy'
            elif progress < 0.5:
                return 'medium'
            elif progress < 0.75:
                return 'hard'
            else:
                return 'extreme'
                
        except Exception as e:
            logger.error(f"Error getting difficulty: {e}")
            return 'medium'
    
    async def _get_real_market_data(self) -> Dict[str, Any]:
        """Get real market data for validation."""
        try:
            # This would be enhanced with actual market data
            return {
                'volatility': 0.02,
                'price_range': (90, 110),
                'trend_consistency': 0.7
            }
        except Exception as e:
            logger.error(f"Error getting real market data: {e}")
            return {}
    
    async def _store_training_results(
        self,
        epoch: int,
        survival_results: List[StrategyTestResult],
        validation: Dict[str, Any],
        improved_strategies: List[str]
    ) -> None:
        """Store training results for analysis."""
        try:
            if not survival_results:
                return
            
            avg_survival = np.mean([r.survival_rate for r in survival_results])
            
            result = {
                'epoch': epoch,
                'timestamp': datetime.utcnow().isoformat(),
                'strategies_tested': len(survival_results),
                'average_survival_rate': avg_survival,
                'validation_score': validation.get('validation_score', 0),
                'strategies_improved': len(improved_strategies)
            }
            
            key = f"{self._training_history_key}:{epoch}"
            await self.state_store.set(
                key,
                str(result),
                expire=86400 * 30  # 30 days
            )
            
        except Exception as e:
            logger.error(f"Error storing training results: {e}")
    
    async def get_training_stats(self) -> Dict[str, Any]:
        """Get GAN training statistics."""
        try:
            keys = await self.state_store.keys(f"{self._training_history_key}:*")
            
            epochs = []
            for key in keys:
                data = await self.state_store.get(key)
                if data:
                    epochs.append(eval(data))
            
            if not epochs:
                return {
                    'total_epochs': 0,
                    'average_survival_rate': 0.0,
                    'total_strategies_improved': 0,
                    'last_training': None
                }
            
            avg_survival = np.mean([e['average_survival_rate'] for e in epochs])
            total_improved = sum([e['strategies_improved'] for e in epochs])
            
            return {
                'total_epochs': len(epochs),
                'average_survival_rate': avg_survival,
                'total_strategies_improved': total_improved,
                'last_training': max([e['timestamp'] for e in epochs])
            }
            
        except Exception as e:
            logger.error(f"Error getting training stats: {e}")
            return {
                'total_epochs': 0,
                'average_survival_rate': 0.0,
                'total_strategies_improved': 0,
                'last_training': None
            }
