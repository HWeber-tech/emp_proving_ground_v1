#!/usr/bin/env python3
"""
Test Advanced Risk Management - Phase 2.2 Validation

This script tests the advanced risk management system integration with evolved strategies
and live trading, validating portfolio-level risk controls and dynamic position sizing.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.trading.advanced_risk_manager import AdvancedRiskManager, RiskLimits, RiskMetrics
from src.trading.strategy_manager import StrategyManager, StrategySignal
from src.trading.live_trading_executor import LiveTradingExecutor, TradingSignal
from src.trading.mock_ctrader_interface import TradingConfig, TradingMode, Position, OrderSide, MarketData

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedRiskManagementTester:
    """Test the advanced risk management system."""
    
    def __init__(self):
        self.strategy_manager = StrategyManager()
        self.risk_limits = RiskLimits()
        self.advanced_risk_manager = AdvancedRiskManager(self.risk_limits, self.strategy_manager)
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run all advanced risk management tests."""
        logger.info("🛡️ Starting Advanced Risk Management Tests (Phase 2.2)")
        
        tests = [
            ("Test Risk Manager Creation", self.test_risk_manager_creation),
            ("Test Signal Validation", self.test_signal_validation),
            ("Test Position Sizing", self.test_position_sizing),
            ("Test Portfolio State Updates", self.test_portfolio_state_updates),
            ("Test Risk Metrics Calculation", self.test_risk_metrics_calculation),
            ("Test Risk Limits Enforcement", self.test_risk_limits_enforcement),
            ("Test Correlation Analysis", self.test_correlation_analysis),
            ("Test Live Trading Integration", self.test_live_trading_integration),
            ("Test Risk Alerts", self.test_risk_alerts),
            ("Test Kelly Criterion", self.test_kelly_criterion)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n📋 Running: {test_name}")
                
                # Check if test function is async
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                    
                self.test_results[test_name] = result
                
                if result['passed']:
                    logger.info(f"✅ PASSED: {test_name}")
                    passed += 1
                else:
                    logger.error(f"❌ FAILED: {test_name} - {result['error']}")
                    
            except Exception as e:
                logger.error(f"❌ ERROR in {test_name}: {e}")
                self.test_results[test_name] = {'passed': False, 'error': str(e)}
        
        # Print summary
        logger.info(f"\n📊 Test Summary:")
        logger.info(f"Passed: {passed}/{total}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 All tests passed! Advanced risk management is working correctly.")
        else:
            logger.error("⚠️ Some tests failed. Check the logs above for details.")
        
        return passed == total
    
    def test_risk_manager_creation(self):
        """Test advanced risk manager initialization."""
        try:
            # Test basic creation
            risk_manager = AdvancedRiskManager(self.risk_limits, self.strategy_manager)
            assert risk_manager is not None
            assert hasattr(risk_manager, 'risk_limits')
            assert hasattr(risk_manager, 'strategy_manager')
            assert hasattr(risk_manager, 'portfolio_state')
            assert hasattr(risk_manager, 'risk_metrics')
            
            # Test risk limits
            assert risk_manager.risk_limits.max_total_exposure == 0.3
            assert risk_manager.risk_limits.max_leverage == 5.0
            assert risk_manager.risk_limits.max_drawdown == 0.15
            assert risk_manager.risk_limits.position_sizing_method == "kelly"
            
            return {'passed': True, 'message': 'Advanced risk manager created successfully'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_signal_validation(self):
        """Test signal validation against risk management rules."""
        try:
            # Create test signal
            test_signal = StrategySignal(
                strategy_id="test_strategy_001",
                symbol="EURUSD",
                action="buy",
                confidence=0.7,
                entry_price=1.1000,
                stop_loss=1.0980,
                take_profit=1.1040,
                volume=0.01
            )
            
            # Create mock market data
            market_data = {
                "EURUSD": MarketData(
                    symbol_id=1,
                    symbol_name="EURUSD",
                    bid=1.1000,
                    ask=1.1002,
                    timestamp=datetime.now(),
                    digits=5
                )
            }
            
            # Test validation
            is_valid, reason, metadata = self.advanced_risk_manager.validate_signal(test_signal, market_data)
            
            # Should be valid for basic signal
            assert isinstance(is_valid, bool)
            assert isinstance(reason, str)
            assert isinstance(metadata, dict)
            
            # Test invalid signal
            invalid_signal = StrategySignal(
                strategy_id="test_strategy_002",
                symbol="EURUSD",
                action="buy",
                confidence=0.05,  # Too low confidence
                entry_price=1.1000,
                stop_loss=1.0980,
                take_profit=1.1040,
                volume=0.01
            )
            
            is_valid_invalid, reason_invalid, _ = self.advanced_risk_manager.validate_signal(invalid_signal, market_data)
            
            # Should be invalid
            assert not is_valid_invalid
            assert "confidence" in reason_invalid.lower() or "invalid" in reason_invalid.lower()
            
            return {'passed': True, 'message': 'Signal validation working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_position_sizing(self):
        """Test dynamic position sizing calculations."""
        try:
            # Create test signal
            test_signal = StrategySignal(
                strategy_id="test_strategy_001",
                symbol="EURUSD",
                action="buy",
                confidence=0.7,
                entry_price=1.1000,
                stop_loss=1.0980,
                take_profit=1.1040,
                volume=0.01
            )
            
            # Create mock market data
            market_data = {
                "EURUSD": MarketData(
                    symbol_id=1,
                    symbol_name="EURUSD",
                    bid=1.1000,
                    ask=1.1002,
                    timestamp=datetime.now(),
                    digits=5
                )
            }
            
            # Test position sizing
            account_equity = 10000.0
            position_size = self.advanced_risk_manager.calculate_position_size(
                test_signal, account_equity, market_data
            )
            
            # Check position size is reasonable
            assert position_size > 0
            assert position_size <= account_equity * self.risk_limits.max_position_size
            assert position_size >= 0.01  # Minimum size
            
            # Test with different account sizes
            small_account_size = self.advanced_risk_manager.calculate_position_size(
                test_signal, 1000.0, market_data
            )
            large_account_size = self.advanced_risk_manager.calculate_position_size(
                test_signal, 100000.0, market_data
            )
            
            # Larger account should allow larger position (within limits)
            assert large_account_size >= small_account_size
            
            return {'passed': True, 'message': f'Position sizing working correctly (size: {position_size:.4f})'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_portfolio_state_updates(self):
        """Test portfolio state updates."""
        try:
            # Create mock positions
            mock_positions = [
                Position(
                    position_id="pos1",
                    symbol_id=1,
                    side=OrderSide.BUY,
                    volume=0.01,
                    entry_price=1.1000,
                    current_price=1.1010,
                    profit_loss=10.0
                ),
                Position(
                    position_id="pos2",
                    symbol_id=2,
                    side=OrderSide.SELL,
                    volume=0.02,
                    entry_price=1.3000,
                    current_price=1.2990,
                    profit_loss=20.0
                )
            ]
            
            # Update portfolio state
            equity = 10000.0
            margin = 500.0
            orders = []
            
            self.advanced_risk_manager.update_portfolio_state(mock_positions, equity, margin, orders)
            
            # Check state was updated
            assert self.advanced_risk_manager.portfolio_state.total_equity == equity
            assert self.advanced_risk_manager.portfolio_state.total_margin == margin
            assert self.advanced_risk_manager.portfolio_state.free_margin == equity - margin
            assert len(self.advanced_risk_manager.portfolio_state.positions) == 2
            
            return {'passed': True, 'message': 'Portfolio state updates working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        try:
            # Create mock positions
            mock_positions = [
                Position(
                    position_id="pos1",
                    symbol_id=1,
                    side=OrderSide.BUY,
                    volume=0.01,
                    entry_price=1.1000,
                    current_price=1.1010,
                    profit_loss=10.0
                )
            ]
            
            # Create mock market data
            market_data = {
                "EURUSD": MarketData(
                    symbol_id=1,
                    symbol_name="EURUSD",
                    bid=1.1000,
                    ask=1.1002,
                    timestamp=datetime.now(),
                    digits=5
                )
            }
            
            # Update risk metrics
            self.advanced_risk_manager.update_risk_metrics(mock_positions, market_data)
            
            # Check metrics were calculated
            assert hasattr(self.advanced_risk_manager.risk_metrics, 'total_exposure')
            assert hasattr(self.advanced_risk_manager.risk_metrics, 'leverage_ratio')
            assert hasattr(self.advanced_risk_manager.risk_metrics, 'correlation_score')
            assert hasattr(self.advanced_risk_manager.risk_metrics, 'var_95')
            
            return {'passed': True, 'message': 'Risk metrics calculation working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_risk_limits_enforcement(self):
        """Test risk limits enforcement."""
        try:
            # Test exposure limits
            self.advanced_risk_manager.risk_metrics.total_exposure = 0.35  # Above 30% limit
            
            test_signal = StrategySignal(
                strategy_id="test_strategy_001",
                symbol="EURUSD",
                action="buy",
                confidence=0.7,
                entry_price=1.1000,
                stop_loss=1.0980,
                take_profit=1.1040,
                volume=0.01
            )
            
            market_data = {}
            
            is_valid, reason, _ = self.advanced_risk_manager.validate_signal(test_signal, market_data)
            
            # Should be rejected due to high exposure
            assert not is_valid
            assert "exposure" in reason.lower()
            
            # Reset exposure
            self.advanced_risk_manager.risk_metrics.total_exposure = 0.1
            
            # Test leverage limits
            self.advanced_risk_manager.risk_metrics.leverage_ratio = 6.0  # Above 5:1 limit
            
            is_valid, reason, _ = self.advanced_risk_manager.validate_signal(test_signal, market_data)
            
            # Should be rejected due to high leverage
            assert not is_valid
            assert "leverage" in reason.lower()
            
            return {'passed': True, 'message': 'Risk limits enforcement working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_correlation_analysis(self):
        """Test correlation analysis functionality."""
        try:
            # Test correlation calculation
            self.advanced_risk_manager.risk_metrics.correlation_score = 0.8  # High correlation
            
            test_signal = StrategySignal(
                strategy_id="test_strategy_001",
                symbol="EURUSD",
                action="buy",
                confidence=0.7,
                entry_price=1.1000,
                stop_loss=1.0980,
                take_profit=1.1040,
                volume=0.01
            )
            
            market_data = {}
            
            is_valid, reason, _ = self.advanced_risk_manager.validate_signal(test_signal, market_data)
            
            # Should be rejected due to high correlation
            assert not is_valid
            assert "correlation" in reason.lower()
            
            # Reset correlation
            self.advanced_risk_manager.risk_metrics.correlation_score = 0.3
            
            return {'passed': True, 'message': 'Correlation analysis working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def test_live_trading_integration(self):
        """Test integration with live trading executor."""
        try:
            # Create trading config
            config = TradingConfig(
                client_id="test_client",
                client_secret="test_secret",
                access_token="test_token",
                refresh_token="test_refresh",
                account_id=12345,
                mode=TradingMode.DEMO
            )
            
            # Create live trading executor
            executor = LiveTradingExecutor(
                config=config,
                symbols=['EURUSD'],
                max_positions=3,
                max_risk_per_trade=0.02
            )
            
            # Check that advanced risk manager is initialized
            assert hasattr(executor, 'advanced_risk_manager')
            assert isinstance(executor.advanced_risk_manager, AdvancedRiskManager)
            
            # Check that it's connected to strategy manager
            assert executor.advanced_risk_manager.strategy_manager == executor.strategy_manager
            
            return {'passed': True, 'message': 'Live trading integration working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_risk_alerts(self):
        """Test risk alert generation."""
        try:
            # Set high risk metrics to trigger alerts
            self.advanced_risk_manager.risk_metrics.total_exposure = 0.25  # 80% of 30% limit
            self.advanced_risk_manager.risk_metrics.leverage_ratio = 4.0  # 80% of 5:1 limit
            self.advanced_risk_manager.risk_metrics.correlation_score = 0.6  # 80% of 70% limit
            self.advanced_risk_manager.risk_metrics.var_95 = 0.016  # 80% of 2% limit
            self.advanced_risk_manager.portfolio_state.daily_pnl = -400.0  # 80% of 5% daily loss
            
            # Get risk report
            risk_report = self.advanced_risk_manager.get_risk_report()
            
            # Check that alerts were generated
            assert 'alerts' in risk_report
            assert len(risk_report['alerts']) > 0
            
            # Check alert content
            alert_text = ' '.join(risk_report['alerts']).lower()
            assert 'exposure' in alert_text or 'leverage' in alert_text or 'correlation' in alert_text
            
            return {'passed': True, 'message': f'Risk alerts working correctly ({len(risk_report["alerts"])} alerts)'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_kelly_criterion(self):
        """Test Kelly criterion position sizing."""
        try:
            # Add a strategy with known performance
            from src.evolution.real_genetic_engine import TradingStrategy
            
            strategy = TradingStrategy(
                id="kelly_test_001",
                name="Kelly Test Strategy",
                parameters={'buy_threshold': 0.5},
                indicators=['SMA'],
                entry_rules=['sma_crossover'],
                exit_rules=['stop_loss'],
                risk_management={'max_risk': 0.02},
                fitness_score=0.6,
                generation=1
            )
            
            # Convert to genome and add to strategy manager
            from src.core.interfaces import DecisionGenome
            
            genome = DecisionGenome(
                genome_id=strategy.id,
                decision_tree={
                    'parameters': strategy.parameters,
                    'indicators': strategy.indicators,
                    'entry_rules': strategy.entry_rules,
                    'exit_rules': strategy.exit_rules,
                    'risk_management': strategy.risk_management
                },
                fitness_score=strategy.fitness_score,
                generation=strategy.generation,
                parent_ids=strategy.parent_ids
            )
            
            self.strategy_manager.add_strategy(genome)
            
            # Update strategy performance
            self.strategy_manager.update_strategy_performance(strategy.id, 100.0)  # Win
            self.strategy_manager.update_strategy_performance(strategy.id, 100.0)  # Win
            self.strategy_manager.update_strategy_performance(strategy.id, -50.0)  # Loss
            self.strategy_manager.update_strategy_performance(strategy.id, 100.0)  # Win
            self.strategy_manager.update_strategy_performance(strategy.id, -50.0)  # Loss
            
            # Create test signal
            test_signal = StrategySignal(
                strategy_id=strategy.id,
                symbol="EURUSD",
                action="buy",
                confidence=0.7,
                entry_price=1.1000,
                stop_loss=1.0980,
                take_profit=1.1040,
                volume=0.01
            )
            
            # Test Kelly position sizing
            account_equity = 10000.0
            market_data = {}
            
            position_size = self.advanced_risk_manager.calculate_position_size(
                test_signal, account_equity, market_data
            )
            
            # Kelly should produce reasonable position size
            assert position_size > 0
            assert position_size <= account_equity * self.risk_limits.max_position_size
            
            return {'passed': True, 'message': f'Kelly criterion working correctly (size: {position_size:.4f})'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}


async def main():
    """Run the advanced risk management tests."""
    print("🛡️ EMP Advanced Risk Management Test Suite")
    print("=" * 50)
    
    tester = AdvancedRiskManagementTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Phase 2.2: Advanced Risk Management - COMPLETED SUCCESSFULLY!")
        print("\n✅ Key Achievements:")
        print("   • Portfolio-Level Risk Controls: Comprehensive exposure and leverage management")
        print("   • Dynamic Position Sizing: Kelly criterion and volatility-adjusted sizing")
        print("   • Correlation Analysis: Real-time portfolio correlation monitoring")
        print("   • Risk Alerts: Automated risk monitoring and alerting")
        print("   • Live Trading Integration: Seamless integration with trading executor")
        print("   • Strategy-Specific Controls: Performance-based risk adjustments")
        print("\n🚀 Ready to proceed to Phase 2.3: Performance Tracking")
    else:
        print("\n❌ Phase 2.2: Advanced Risk Management - FAILED")
        print("Please check the test results above and fix any issues.")
    
    return success


if __name__ == "__main__":
    asyncio.run(main()) 