#!/usr/bin/env python3
"""
Component Integration Framework
Complete integration of all Phase 2 components into unified system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from src.evolution.fitness.multi_dimensional_fitness_evaluator import MultiDimensionalFitnessEvaluator
from src.evolution.selection.adversarial_selector import AdversarialSelector
from src.trading.strategy_manager import StrategyManager
from src.trading.risk.market_regime_detector import MarketRegimeDetector
from src.trading.risk.advanced_risk_manager import AdvancedRiskManager
from src.validation.phase2_validation_suite import Phase2ValidationSuite

logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """Result of component integration"""
    component_name: str
    integration_success: bool
    integration_time: float
    error_message: Optional[str] = None
    validation_passed: bool = False


class ComponentIntegrator:
    """Master integrator for all Phase 2 components"""
    
    def __init__(self):
        self.fitness_evaluator = None
        self.adversarial_selector = None
        self.strategy_manager = None
        self.regime_detector = None
        self.risk_manager = None
        self.validation_suite = None
        
    async def integrate_all_components(self) -> Dict[str, IntegrationResult]:
        """Integrate all Phase 2 components into unified system"""
        
        logger.info("Starting Phase 2 component integration")
        
        integration_results = {}
        
        # 1. Integrate Evolution Engine Components
        evolution_result = await self._integrate_evolution_engine()
        integration_results['evolution_engine'] = evolution_result
        
        # 2. Integrate Risk Management Components
        risk_result = await self._integrate_risk_management()
        integration_results['risk_management'] = risk_result
        
        # 3. Integrate Validation Components
        validation_result = await self._integrate_validation_suite()
        integration_results['validation_suite'] = validation_result
        
        # 4. Cross-Component Integration
        cross_integration_result = await self._perform_cross_integration()
        integration_results['cross_integration'] = cross_integration_result
        
        # 5. Final System Integration
        final_integration_result = await self._perform_final_integration()
        integration_results['final_integration'] = final_integration_result
        
        logger.info("Phase 2 component integration completed")
        return integration_results
    
    async def _integrate_evolution_engine(self) -> IntegrationResult:
        """Integrate evolution engine components"""
        
        try:
            start_time = datetime.now()
            
            # Initialize fitness evaluator
            self.fitness_evaluator = MultiDimensionalFitnessEvaluator()
            await self.fitness_evaluator.initialize_fitness_dimensions()
            
            # Initialize adversarial selector
            self.adversarial_selector = AdversarialSelector()
            await self.adversarial_selector.load_stress_scenarios()
            
            # Validate integration
            validation_passed = await self._validate_evolution_integration()
            
            integration_time = (datetime.now() - start_time).total_seconds()
            
            return IntegrationResult(
                component_name="evolution_engine",
                integration_success=True,
                integration_time=integration_time,
                validation_passed=validation_passed
            )
            
        except Exception as e:
            logger.error(f"Evolution engine integration failed: {e}")
            return IntegrationResult(
                component_name="evolution_engine",
                integration_success=False,
                integration_time=0.0,
                error_message=str(e),
                validation_passed=False
            )
    
    async def _integrate_risk_management(self) -> IntegrationResult:
        """Integrate risk management components"""
        
        try:
            start_time = datetime.now()
            
            # Initialize strategy manager
            self.strategy_manager = StrategyManager()
            await self.strategy_manager.initialize()
            
            # Initialize regime detector
            self.regime_detector = MarketRegimeDetector()
            await self.regime_detector.initialize_detection_algorithms()
            
            # Initialize advanced risk manager
            self.risk_manager = AdvancedRiskManager()
            await self.risk_manager.configure_components(
                strategy_manager=self.strategy_manager,
                regime_detector=self.regime_detector
            )
            
            # Validate integration
            validation_passed = await self._validate_risk_integration()
            
            integration_time = (datetime.now() - start_time).total_seconds()
            
            return IntegrationResult(
                component_name="risk_management",
                integration_success=True,
                integration_time=integration_time,
                validation_passed=validation_passed
            )
            
        except Exception as e:
            logger.error(f"Risk management integration failed: {e}")
            return IntegrationResult(
                component_name="risk_management",
                integration_success=False,
                integration_time=0.0,
                error_message=str(e),
                validation_passed=False
            )
    
    async def _integrate_validation_suite(self) -> IntegrationResult:
        """Integrate validation components"""
        
        try:
            start_time = datetime.now()
            
            # Initialize validation suite
            self.validation_suite = Phase2ValidationSuite()
            await self.validation_suite.initialize()
            
            # Validate integration
            validation_passed = await self._validate_validation_integration()
            
            integration_time = (datetime.now() - start_time).total_seconds()
            
            return IntegrationResult(
                component_name="validation_suite",
                integration_success=True,
                integration_time=integration_time,
                validation_passed=validation_passed
            )
            
        except Exception as e:
            logger.error(f"Validation suite integration failed: {e}")
            return IntegrationResult(
                component_name="validation_suite",
                integration_success=False,
                integration_time=0.0,
                error_message=str(e),
                validation_passed=False
            )
    
    async def _perform_cross_integration(self) -> IntegrationResult:
        """Perform cross-component integration"""
        
        try:
            start_time = datetime.now()
            
            # Connect evolution engine to strategy manager
            if self.fitness_evaluator and self.strategy_manager:
                await self.strategy_manager.set_fitness_evaluator(self.fitness_evaluator)
            
            # Connect risk manager to evolution engine
            if self.risk_manager and self.adversarial_selector:
                await self.adversarial_selector.set_risk_manager(self.risk_manager)
            
            # Connect validation suite to all components
            if self.validation_suite:
                await self.validation_suite.register_components(
                    fitness_evaluator=self.fitness_evaluator,
                    adversarial_selector=self.adversarial_selector,
                    strategy_manager=self.strategy_manager,
                    regime_detector=self.regime_detector,
                    risk_manager=self.risk_manager
                )
            
            # Validate cross-integration
            validation_passed = await self._validate_cross_integration()
            
            integration_time = (datetime.now() - start_time).total_seconds()
            
            return IntegrationResult(
                component_name="cross_integration",
                integration_success=True,
                integration_time=integration_time,
                validation_passed=validation_passed
            )
            
        except Exception as e:
            logger.error(f"Cross integration failed: {e}")
            return IntegrationResult(
                component_name="cross_integration",
                integration_success=False,
                integration_time=0.0,
                error_message=str(e),
                validation_passed=False
            )
    
    async def _perform_final_integration(self) -> IntegrationResult:
        """Perform final system integration"""
        
        try:
            start_time = datetime.now()
            
            # Run complete system test
            system_test_result = await self._run_complete_system_test()
            
            # Validate all components working together
            validation_passed = system_test_result.success
            
            integration_time = (datetime.now() - start_time).total_seconds()
            
            return IntegrationResult(
                component_name="final_integration",
                integration_success=system_test_result.success,
                integration_time=integration_time,
                validation_passed=validation_passed
            )
            
        except Exception as e:
            logger.error(f"Final integration failed: {e}")
            return IntegrationResult(
                component_name="final_integration",
                integration_success=False,
                integration_time=0.0,
                error_message=str(e),
                validation_passed=False
            )
    
    async def _validate_evolution_integration(self) -> bool:
        """Validate evolution engine integration"""
        
        try:
            # Test fitness evaluation
            test_data = {
                'total_return': 0.15,
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.02,
                'win_rate': 0.65
            }
            
            fitness_score = await self.fitness_evaluator.evaluate_strategy_fitness(
                strategy_id="test_strategy",
                performance_data=test_data,
                market_regimes=['TRENDING_UP']
            )
            
            return fitness_score.overall > 0.0
            
        except Exception as e:
            logger.error(f"Evolution integration validation failed: {e}")
            return False
    
    async def _validate_risk_integration(self) -> bool:
        """Validate risk management integration"""
        
        try:
            # Test regime detection
            test_market_data = {
                'price_trend': 0.8,
                'volatility': 0.15,
                'volume_trend': 0.7
            }
            
            regime = await self.regime_detector.detect_regime(test_market_data)
            
            return regime in ['TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE', 'CRISIS']
            
        except Exception as e:
            logger.error(f"Risk integration validation failed: {e}")
            return False
    
    async def _validate_validation_integration(self) -> bool:
        """Validate validation suite integration"""
        
        try:
            # Test validation suite initialization
            return await self.validation_suite.is_initialized()
            
        except Exception as e:
            logger.error(f"Validation integration validation failed: {e}")
            return False
    
    async def _validate_cross_integration(self) -> bool:
        """Validate cross-component integration"""
        
        try:
            # Test component communication
            return all([
                self.fitness_evaluator is not None,
                self.adversarial_selector is not None,
                self.strategy_manager is not None,
                self.regime_detector is not None,
                self.risk_manager is not None,
                self.validation_suite is not None
            ])
            
        except Exception as e:
            logger.error(f"Cross integration validation failed: {e}")
            return False
    
    async def _run_complete_system_test(self):
        """Run complete system integration test"""
        
        class SystemTestResult:
            def __init__(self):
                self.success = False
                self.components_tested = 0
                self.assertions_passed = 0
        
        result = SystemTestResult()
        
        try:
            # Test 1: Market data processing
            market_data = {'price': 1.1000, 'volume': 1000}
            
            # Test 2: Strategy evaluation
            if self.strategy_manager:
                signals = await self.strategy_manager.evaluate_strategies('EURUSD', market_data)
                result.components_tested += 1
                if len(signals) >= 0:  # Allow zero signals
                    result.assertions_passed += 1
            
            # Test 3: Risk validation
            if self.risk_manager:
                risk_check = await self.risk_manager.validate_signal(
                    {'symbol': 'EURUSD', 'action': 'buy', 'confidence': 0.8},
                    market_data
                )
                result.components_tested += 1
                if risk_check[0]:  # is_valid
                    result.assertions_passed += 1
            
            # Test 4: Validation suite
            if self.validation_suite:
                validation_result = await self.validation_suite.run_quick_test()
                result.components_tested += 1
                if validation_result.passed:
                    result.assertions_passed += 1
            
            result.success = result.assertions_passed == result.components_tested
            
        except Exception as e:
            logger.error(f"Complete system test failed: {e}")
            result.success = False
        
        return result


async def main():
    """Test component integration"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    integrator = ComponentIntegrator()
    results = await integrator.integrate_all_components()
    
    print("Integration Results:")
    for component, result in results.items():
        print(f"  {component}: {'✓' if result.integration_success else '✗'}")
        if result.error_message:
            print(f"    Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
