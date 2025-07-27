#!/usr/bin/env python3
"""
Complete System Verification Script
==================================

Comprehensive verification of the entire EMP system including:
- All core interfaces implemented
- Strategy engine operational
- Risk management functional
- Evolution system ready
- Integration tests passing
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any

# Import all components
from src.risk.risk_manager_impl import create_risk_manager
from src.trading.strategy_engine.strategy_engine_impl import create_strategy_engine
from src.trading.strategy_engine.templates.moving_average_strategy import create_moving_average_strategy
from src.evolution.engine.population_manager import PopulationManager
from src.evolution.fitness.real_trading_fitness_evaluator import RealTradingFitnessEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemVerifier:
    """Comprehensive system verification."""
    
    def __init__(self):
        """Initialize the verifier."""
        self.results = {
            'risk_manager': False,
            'strategy_engine': False,
            'strategy_templates': False,
            'evolution_system': False,
            'integration': False,
            'overall': False
        }
        self.errors = []
    
    async def verify_risk_manager(self) -> bool:
        """Verify risk management system."""
        try:
            logger.info("🔍 Verifying Risk Manager...")
            
            # Create risk manager
            risk_manager = create_risk_manager(initial_balance=10000.0)
            
            # Test basic functionality
            assert risk_manager.account_balance == 10000.0
            
            # Test position validation
            position = {
                'symbol': 'EURUSD',
                'size': 1000.0,
                'entry_price': 1.1000
            }
            
            is_valid = await risk_manager.validate_position(position)
            assert is_valid is True
            
            # Test position sizing
            signal = {
                'symbol': 'EURUSD',
                'confidence': 0.7,
                'stop_loss_pct': 0.02
            }
            
            position_size = await risk_manager.calculate_position_size(signal)
            assert position_size > 0
            
            # Test risk summary
            summary = risk_manager.get_risk_summary()
            assert 'account_balance' in summary
            
            logger.info("✅ Risk Manager verified successfully")
            self.results['risk_manager'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk Manager verification failed: {e}")
            self.errors.append(f"Risk Manager: {e}")
            return False
    
    async def verify_strategy_engine(self) -> bool:
        """Verify strategy engine."""
        try:
            logger.info("🔍 Verifying Strategy Engine...")
            
            # Create strategy engine
            risk_manager = create_risk_manager(10000.0)
            strategy_engine = create_strategy_engine(risk_manager)
            
            # Test strategy registration
            strategy = create_moving_average_strategy(
                strategy_id="test_engine",
                symbols=["EURUSD"],
                parameters={'fast_period': 5, 'slow_period': 10}
            )
            
            success = strategy_engine.register_strategy(strategy)
            assert success is True
            
            # Test strategy start
            started = strategy_engine.start_strategy("test_engine")
            assert started is True
            
            # Test strategy execution
            market_data = {
                'symbol': 'EURUSD',
                'close': 1.1000,
                'volume': 1000,
                'timestamp': datetime.now()
            }
            
            result = await strategy_engine.execute_strategy("test_engine", market_data)
            assert result is not None
            
            # Test strategy management
            strategies = strategy_engine.get_all_strategies()
            assert "test_engine" in strategies
            
            logger.info("✅ Strategy Engine verified successfully")
            self.results['strategy_engine'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Strategy Engine verification failed: {e}")
            self.errors.append(f"Strategy Engine: {e}")
            return False
    
    async def verify_strategy_templates(self) -> bool:
        """Verify strategy templates."""
        try:
            logger.info("🔍 Verifying Strategy Templates...")
            
            # Test moving average strategy
            strategy = create_moving_average_strategy(
                strategy_id="template_test",
                symbols=["EURUSD", "GBPUSD"],
                parameters={'fast_period': 10, 'slow_period': 20}
            )
            
            # Test strategy info
            info = strategy.get_strategy_info()
            assert 'strategy_id' in info
            assert 'parameters' in info
            
            # Test parameter update
            success = strategy.update_parameters({'fast_period': 15})
            assert success is True
            
            # Verify update
            params = strategy.get_parameters()
            assert params['fast_period'] == 15
            
            logger.info("✅ Strategy Templates verified successfully")
            self.results['strategy_templates'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Strategy Templates verification failed: {e}")
            self.errors.append(f"Strategy Templates: {e}")
            return False
    
    async def verify_evolution_system(self) -> bool:
        """Verify evolution system."""
        try:
            logger.info("🔍 Verifying Evolution System...")
            
            # Test population manager
            population_manager = PopulationManager(population_size=5)
            assert population_manager.population_size == 5
            
            # Test fitness evaluator
            risk_manager = create_risk_manager(10000.0)
            strategy_engine = create_strategy_engine(risk_manager)
            fitness_evaluator = RealTradingFitnessEvaluator(
                symbol="EURUSD",
                lookback_days=30,
                initial_capital=100000.0
            )
            
            assert fitness_evaluator is not None
            
            logger.info("✅ Evolution System verified successfully")
            self.results['evolution_system'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Evolution System verification failed: {e}")
            self.errors.append(f"Evolution System: {e}")
            return False
    
    async def verify_integration(self) -> bool:
        """Verify system integration."""
        try:
            logger.info("🔍 Verifying System Integration...")
            
            # Create complete system
            risk_manager = create_risk_manager(10000.0)
            strategy_engine = create_strategy_engine(risk_manager)
            
            # Register strategy
            strategy = create_moving_average_strategy(
                strategy_id="integration_test",
                symbols=["EURUSD"],
                parameters={'fast_period': 5, 'slow_period': 10}
            )
            
            strategy_engine.register_strategy(strategy)
            strategy_engine.start_strategy("integration_test")
            
            # Test complete flow
            market_data = {
                'symbol': 'EURUSD',
                'close': 1.1000,
                'volume': 1000,
                'timestamp': datetime.now()
            }
            
            # Execute strategy
            result = await strategy_engine.execute_strategy("integration_test", market_data)
            assert result is not None
            
            # Test risk validation
            if result.signal:
                is_valid = await risk_manager.validate_position({
                    'symbol': result.signal.symbol,
                    'size': result.signal.quantity,
                    'entry_price': result.signal.price
                })
                assert is_valid is True
            
            logger.info("✅ System Integration verified successfully")
            self.results['integration'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ System Integration verification failed: {e}")
            self.errors.append(f"Integration: {e}")
            return False
    
    async def run_all_verifications(self) -> bool:
        """Run all verification tests."""
        logger.info("🚀 Starting Complete System Verification...")
        
        # Run individual verifications
        await self.verify_risk_manager()
        await self.verify_strategy_engine()
        await self.verify_strategy_templates()
        await self.verify_evolution_system()
        await self.verify_integration()
        
        # Calculate overall result
        all_passed = all(self.results.values())
        self.results['overall'] = all_passed
        
        # Print summary
        self.print_summary()
        
        return all_passed
    
    def print_summary(self):
        """Print verification summary."""
        logger.info("\n" + "="*60)
        logger.info("SYSTEM VERIFICATION SUMMARY")
        logger.info("="*60)
        
        for component, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{component.upper():<20} {status}")
        
        logger.info("-"*60)
        
        if self.errors:
            logger.error("ERRORS:")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        overall_status = "✅ ALL TESTS PASSED" if self.results['overall'] else "❌ SOME TESTS FAILED"
        logger.info(f"\nOVERALL: {overall_status}")
        logger.info("="*60)


async def main():
    """Main verification function."""
    verifier = SystemVerifier()
    success = await verifier.run_all_verifications()
    
    if success:
        logger.info("\n🎉 System verification completed successfully!")
        logger.info("The EMP system is ready for production use.")
        return 0
    else:
        logger.error("\n💥 System verification failed!")
        logger.error("Please address the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
