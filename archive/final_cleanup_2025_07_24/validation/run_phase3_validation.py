#!/usr/bin/env python3
"""
Phase 3 Validation Suite
========================

Comprehensive validation script for Phase 3 intelligence systems.
Tests all components and integration points.

Usage:
    python run_phase3_validation.py
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add src to path
sys.path.append('src')

from src.intelligence import (
    SentientAdaptationEngine,
    PredictiveMarketModeler,
    MarketGAN,
    RedTeamAI,
    SpecializedPredatorEvolution,
    PortfolioEvolutionEngine,
    CompetitiveIntelligenceSystem,
    Phase3IntelligenceOrchestrator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase3Validator:
    """Validates all Phase 3 intelligence systems."""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'systems': {},
            'integration': {},
            'performance': {},
            'success': False
        }
        
    async def validate_sentient_adaptation(self) -> Dict[str, Any]:
        """Validate sentient adaptation engine."""
        logger.info("Validating Sentient Adaptation Engine...")
        
        try:
            engine = SentientAdaptationEngine()
            await engine.initialize()
            
            # Test real-time adaptation
            market_event = {
                'price': 100.0,
                'volume': 1000000,
                'volatility': 0.02,
                'timestamp': datetime.utcnow()
            }
            
            strategy_response = {
                'action': 'BUY',
                'confidence': 0.8,
                'position_size': 0.1
            }
            
            outcome = {
                'pnl': 1000.0,
                'win': True,
                'duration': 3600
            }
            
            adaptation = await engine.adapt_in_real_time(
                market_event, strategy_response, outcome
            )
            
            result = {
                'status': 'success',
                'adaptation_generated': adaptation is not None,
                'learning_signal': adaptation.get('confidence', 0) if adaptation else 0
            }
            
        except Exception as e:
            result = {
                'status': 'failed',
                'error': str(e)
            }
            
        self.test_results['systems']['sentient_adaptation'] = result
        return result
    
    async def validate_predictive_modeling(self) -> Dict[str, Any]:
        """Validate predictive market modeling."""
        logger.info("Validating Predictive Market Modeling...")
        
        try:
            modeler = PredictiveMarketModeler()
            await modeler.initialize()
            
            # Test market prediction
            current_state = {
                'price': 100.0,
                'volume': 1000000,
                'volatility': 0.02,
                'trend': 0.01
            }
            
            predictions = await modeler.predict_market_scenarios(
                current_state, timedelta(hours=24)
            )
            
            result = {
                'status': 'success',
                'predictions_generated': len(predictions) > 0,
                'prediction_accuracy': predictions[0].get('confidence', 0) if predictions else 0
            }
            
        except Exception as e:
            result = {
                'status': 'failed',
                'error': str(e)
            }
            
        self.test_results['systems']['predictive_modeling'] = result
        return result
    
    async def validate_adversarial_training(self) -> Dict[str, Any]:
        """Validate adversarial training system."""
        logger.info("Validating Adversarial Training System...")
        
        try:
            gan = MarketGAN()
            await gan.initialize()
            
            # Test strategy population
            strategy_population = [
                {'id': 'strategy_1', 'fitness': 0.7},
                {'id': 'strategy_2', 'fitness': 0.8},
                {'id': 'strategy_3', 'fitness': 0.6}
            ]
            
            improved_strategies = await gan.train_adversarial_strategies(
                strategy_population
            )
            
            result = {
                'status': 'success',
                'strategies_improved': len(improved_strategies) > 0,
                'survival_rate': len(improved_strategies) / len(strategy_population)
            }
            
        except Exception as e:
            result = {
                'status': 'failed',
                'error': str(e)
            }
            
        self.test_results['systems']['adversarial_training'] = result
        return result
    
    async def validate_red_team_ai(self) -> Dict[str, Any]:
        """Validate red team AI system."""
        logger.info("Validating Red Team AI System...")
        
        try:
            red_team = RedTeamAI()
            await red_team.initialize()
            
            # Test strategy attack
            target_strategy = {
                'id': 'test_strategy',
                'type': 'momentum',
                'parameters': {'period': 20, 'threshold': 0.02}
            }
            
            attack_result = await red_team.attack_strategy(target_strategy)
            
            result = {
                'status': 'success',
                'attack_completed': attack_result is not None,
                'vulnerabilities_found': len(attack_result.get('weaknesses', [])) if attack_result else 0
            }
            
        except Exception as e:
            result = {
                'status': 'failed',
                'error': str(e)
            }
            
        self.test_results['systems']['red_team_ai'] = result
        return result
    
    async def validate_specialized_predators(self) -> Dict[str, Any]:
        """Validate specialized predator evolution."""
        logger.info("Validating Specialized Predator Evolution...")
        
        try:
            evolution = SpecializedPredatorEvolution()
            await evolution.initialize()
            
            # Test predator evolution
            predators = await evolution.evolve_specialized_predators()
            
            result = {
                'status': 'success',
                'predators_evolved': len(predators) > 0,
                'predator_types': list(predators.keys()) if predators else []
            }
            
        except Exception as e:
            result = {
                'status': 'failed',
                'error': str(e)
            }
            
        self.test_results['systems']['specialized_predators'] = result
        return result
    
    async def validate_portfolio_evolution(self) -> Dict[str, Any]:
        """Validate portfolio-level evolution."""
        logger.info("Validating Portfolio Evolution...")
        
        try:
            evolution = PortfolioEvolutionEngine()
            await evolution.initialize()
            
            # Test portfolio evolution
            strategies = [
                {'id': 'strategy_1', 'type': 'momentum', 'fitness': 0.7},
                {'id': 'strategy_2', 'type': 'mean_reversion', 'fitness': 0.8}
            ]
            
            market_data = {
                'price': 100.0,
                'volume': 1000000,
                'volatility': 0.02
            }
            
            portfolio_result = await evolution.evolve_portfolio(strategies, market_data)
            
            result = {
                'status': 'success',
                'portfolio_optimized': portfolio_result is not None,
                'correlation_improved': portfolio_result.get('correlation_score', 0) > 0.5 if portfolio_result else False
            }
            
        except Exception as e:
            result = {
                'status': 'failed',
                'error': str(e)
            }
            
        self.test_results['systems']['portfolio_evolution'] = result
        return result
    
    async def validate_competitive_intelligence(self) -> Dict[str, Any]:
        """Validate competitive intelligence system."""
        logger.info("Validating Competitive Intelligence...")
        
        try:
            intelligence = CompetitiveIntelligenceSystem()
            
            # Test market data
            market_data = {
                'order_sizes': [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.8],
                'frequencies': [150, 200, 180, 220, 160, 190, 170, 210],
                'latencies': [0.005, 0.008, 0.006, 0.009, 0.004, 0.007, 0.005, 0.008]
            }
            
            our_performance = {
                'market_share': 0.15,
                'win_rate': 0.65,
                'profit_factor': 1.3
            }
            
            analysis = await intelligence.analyze_competitive_landscape(
                market_data, our_performance
            )
            
            result = {
                'status': 'success',
                'competitors_identified': len(analysis.get('competitors', [])) > 0,
                'counter_strategies': len(analysis.get('counter_strategies', [])) > 0
            }
            
        except Exception as e:
            result = {
                'status': 'failed',
                'error': str(e)
            }
            
        self.test_results['systems']['competitive_intelligence'] = result
        return result
    
    async def validate_integration(self) -> Dict[str, Any]:
        """Validate complete Phase 3 integration."""
        logger.info("Validating Phase 3 Integration...")
        
        try:
            orchestrator = Phase3IntelligenceOrchestrator()
            await orchestrator.initialize_phase3()
            
            # Test market data
            market_data = {
                'price': 100.0,
                'volume': 1000000,
                'volatility': 0.02,
                'trend': 0.01
            }
            
            # Test strategies
            strategies = [
                {'id': 'strategy_1', 'type': 'momentum'},
                {'id': 'strategy_2', 'type': 'mean_reversion'}
            ]
            
            # Run complete intelligence cycle
            results = await orchestrator.run_intelligence_cycle(market_data, strategies)
            
            result = {
                'status': 'success',
                'cycle_completed': True,
                'systems_active': len([k for k, v in results.items() if v]) > 5
            }
            
        except Exception as e:
            result = {
                'status': 'failed',
                'error': str(e)
            }
            
        self.test_results['integration'] = result
        return result
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all Phase 3 validations."""
        logger.info("Starting Phase 3 validation suite...")
        
        # Validate individual systems
        await self.validate_sentient_adaptation()
        await self.validate_predictive_modeling()
        await self.validate_adversarial_training()
        await self.validate_red_team_ai()
        await self.validate_specialized_predators()
        await self.validate_portfolio_evolution()
        await self.validate_competitive_intelligence()
        
        # Validate integration
        await self.validate_integration()
        
        # Calculate overall success
        system_success = sum(1 for r in self.test_results['systems'].values() 
                           if r.get('status') == 'success')
        total_systems = len(self.test_results['systems'])
        
        integration_success = self.test_results['integration'].get('status') == 'success'
        
        self.test_results['success'] = (
            system_success >= total_systems * 0.8 and integration_success
        )
        
        self.test_results['summary'] = {
            'systems_passed': system_success,
            'total_systems': total_systems,
            'integration_passed': integration_success,
            'overall_success': self.test_results['success']
        }
        
        logger.info("Phase 3 validation complete")
        return self.test_results
    
    def save_results(self, filename: str = "phase3_validation_results.json"):
        """Save validation results to file."""
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to {filename}")


async def main():
    """Main validation function."""
    validator = Phase3Validator()
    results = await validator.run_all_validations()
    
    # Print summary
    print("\n" + "="*60)
    print("PHASE 3 VALIDATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Success: {results['success']}")
    print(f"Systems Passed: {results['summary']['systems_passed']}/{results['summary']['total_systems']}")
    print(f"Integration Passed: {results['summary']['integration_passed']}")
    
    print("\nSystem Results:")
    for system, result in results['systems'].items():
        status = "✅ PASS" if result.get('status') == 'success' else "❌ FAIL"
        print(f"  {system}: {status}")
        if result.get('error'):
            print(f"    Error: {result['error']}")
    
    # Save results
    validator.save_results()
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
