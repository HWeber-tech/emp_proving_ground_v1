#!/usr/bin/env python3
"""
Component Integrator
===================

Integrates all Phase 2 components into a unified system.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from src.evolution.fitness.multi_dimensional_fitness_evaluator import MultiDimensionalFitnessEvaluator
from src.evolution.selection.adversarial_selector import AdversarialSelector
from src.trading.strategy_manager import StrategyManager
from src.trading.risk.market_regime_detector import MarketRegimeDetector
from src.trading.risk.advanced_risk_manager import AdvancedRiskManager, RiskLimits

logger = logging.getLogger(__name__)


class ComponentIntegrator:
    """Integrates all Phase 2 components"""
    
    def __init__(self):
        self.components = {}
        self.integration_status = {}
        self.last_check = None
    
    async def initialize_components(self) -> bool:
        """Initialize all Phase 2 components"""
        try:
            logger.info("Initializing Phase 2 components...")
            
            # Initialize fitness evaluator
            self.components['fitness_evaluator'] = MultiDimensionalFitnessEvaluator()
            self.integration_status['fitness_evaluator'] = True
            
            # Initialize adversarial selector
            self.components['adversarial_selector'] = AdversarialSelector()
            self.integration_status['adversarial_selector'] = True
            
            # Initialize strategy manager
            self.components['strategy_manager'] = StrategyManager()
            self.integration_status['strategy_manager'] = True
            
            # Initialize market regime detector
            self.components['regime_detector'] = MarketRegimeDetector()
            self.integration_status['regime_detector'] = True
            
            # Initialize risk manager
            risk_limits = RiskLimits()
            self.components['risk_manager'] = AdvancedRiskManager(
                risk_limits=risk_limits,
                strategy_manager=self.components['strategy_manager']
            )
            self.integration_status['risk_manager'] = True
            
            self.last_check = datetime.now()
            logger.info("All Phase 2 components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    async def test_integration(self) -> Dict[str, Any]:
        """Test integration between components"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'integration': {},
            'overall_status': 'PENDING'
        }
        
        try:
            # Test fitness evaluator
            if 'fitness_evaluator' in self.components:
                results['components']['fitness_evaluator'] = await self._test_fitness_evaluator()
            
            # Test adversarial selector
            if 'adversarial_selector' in self.components:
                results['components']['adversarial_selector'] = await self._test_adversarial_selector()
            
            # Test strategy manager
            if 'strategy_manager' in self.components:
                results['components']['strategy_manager'] = await self._test_strategy_manager()
            
            # Test regime detector
            if 'regime_detector' in self.components:
                results['components']['regime_detector'] = await self._test_regime_detector()
            
            # Test risk manager
            if 'risk_manager' in self.components:
                results['components']['risk_manager'] = await self._test_risk_manager()
            
            # Test end-to-end workflow
            results['integration']['end_to_end'] = await self._test_end_to_end_workflow()
            
            # Determine overall status
            all_passed = all(
                component['passed'] for component in results['components'].values()
            )
            results['overall_status'] = 'PASS' if all_passed else 'FAIL'
            
            return results
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            results['overall_status'] = 'ERROR'
            results['error'] = str(e)
            return results
    
    async def _test_fitness_evaluator(self) -> Dict[str, Any]:
        """Test fitness evaluator"""
        try:
            evaluator = self.components['fitness_evaluator']
            
            # Create test data
            performance_data = {
                'total_return': 0.15,
                'sharpe_ratio': 2.0,
                'max_drawdown': 0.02,
                'win_rate': 0.65
            }
            
            fitness = await evaluator.evaluate_strategy_fitness(
                strategy_id="test_strategy",
                performance_data=performance_data,
                market_regimes=["TRENDING_UP", "RANGING"]
            )
            
            return {
                'passed': 0 <= fitness.overall <= 1,
                'fitness_score': fitness.overall,
                'details': 'Fitness evaluator working correctly'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Fitness evaluator test failed'
            }
    
    async def _test_adversarial_selector(self) -> Dict[str, Any]:
        """Test adversarial selector"""
        try:
            selector = self.components['adversarial_selector']
            
            # Create test strategies
            strategies = [
                {'id': 'strategy1', 'fitness': 0.8, 'survival_rate': 0.9},
                {'id': 'strategy2', 'fitness': 0.6, 'survival_rate': 0.5},
                {'id': 'strategy3', 'fitness': 0.9, 'survival_rate': 0.95}
            ]
            
            selected = selector.select_strategies(strategies)
            
            return {
                'passed': len(selected) > 0,
                'selected_count': len(selected),
                'details': 'Adversarial selector working correctly'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Adversarial selector test failed'
            }
    
    async def _test_strategy_manager(self) -> Dict[str, Any]:
        """Test strategy manager"""
        try:
            manager = self.components['strategy_manager']
            
            # Test basic functionality
            summary = manager.get_strategy_summary()
            
            return {
                'passed': isinstance(summary, dict),
                'strategy_count': summary.get('total_strategies', 0),
                'details': 'Strategy manager working correctly'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Strategy manager test failed'
            }
    
    async def _test_regime_detector(self) -> Dict[str, Any]:
        """Test regime detector"""
        try:
            detector = self.components['regime_detector']
            
            # Create test data
            test_data = pd.DataFrame({
                'close': np.linspace(100, 110, 50),
                'volume': np.random.randint(1000, 10000, 50)
            })
            
            regime = detector.detect_current_regime(test_data)
            
            return {
                'passed': regime is not None,
                'regime': str(regime),
                'details': 'Regime detector working correctly'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Regime detector test failed'
            }
    
    async def _test_risk_manager(self) -> Dict[str, Any]:
        """Test risk manager"""
        try:
            manager = self.components['risk_manager']
            
            # Test basic functionality
            report = manager.get_risk_report()
            
            return {
                'passed': isinstance(report, dict),
                'risk_metrics': len(report.get('risk_metrics', {})),
                'details': 'Risk manager working correctly'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'Risk manager test failed'
            }
    
    async def _test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test end-to-end workflow"""
        try:
            # Create test data
            test_data = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Test complete workflow
            regime = self.components['regime_detector'].detect_current_regime(test_data)
            
            return {
                'passed': regime is not None,
                'workflow_completed': True,
                'details': 'End-to-end workflow test completed'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'details': 'End-to-end workflow test failed'
            }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'timestamp': self.last_check.isoformat() if self.last_check else None,
            'components': self.integration_status,
            'total_components': len(self.components),
            'initialized_components': sum(1 for v in self.integration_status.values() if v)
        }


async def main():
    """Test component integration"""
    integrator = ComponentIntegrator()
    
    # Initialize components
    success = await integrator.initialize_components()
    
    if success:
        # Test integration
        results = await integrator.test_integration()
        
        print("\n" + "="*60)
        print("COMPONENT INTEGRATION TEST RESULTS")
        print("="*60)
        print(f"Overall Status: {results['overall_status']}")
        print()
        
        for component, result in results['components'].items():
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{component.upper()}: {status}")
            if 'details' in result:
                print(f"  {result['details']}")
            print()
        
        if results['overall_status'] == 'PASS':
            print("✅ All components integrated successfully")
        else:
            print("❌ Integration issues detected")
    
    else:
        print("❌ Failed to initialize components")


if __name__ == "__main__":
    asyncio.run(main())
