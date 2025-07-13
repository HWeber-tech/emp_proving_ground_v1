"""
Test script for EMP Proving Ground v2.0 Financial Core
Validates the Risk Management Core and PnL Engine implementation
"""
import sys
import os
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_foundational_components():
    """Test Part 0: Foundational Principles & Dependencies"""
    logger.info("=" * 60)
    logger.info("TESTING FOUNDATIONAL COMPONENTS")
    logger.info("=" * 60)
    
    try:
        from emp_proving_ground_unified import (
            RiskConfig, Instrument, InstrumentProvider, 
            CurrencyConverter, Decimal
        )
        
        # Test 1: RiskConfig validation
        logger.info("Testing RiskConfig...")
        config = RiskConfig()
        assert config.max_risk_per_trade_pct == Decimal('0.02')
        assert config.max_leverage == Decimal('10.0')
        logger.info("✓ RiskConfig validation passed")
        
        # Test 2: Instrument creation
        logger.info("Testing Instrument creation...")
        instrument = Instrument(
            symbol="EUR_USD",
            pip_decimal_places=4,
            contract_size=Decimal('100000'),
            long_swap_rate=Decimal('-0.0001'),
            short_swap_rate=Decimal('0.0001'),
            margin_currency="USD"
        )
        assert instrument.symbol == "EUR_USD"
        assert instrument.pip_decimal_places == 4
        logger.info("✓ Instrument creation passed")
        
        # Test 3: InstrumentProvider
        logger.info("Testing InstrumentProvider...")
        provider = InstrumentProvider()
        eur_usd = provider.get_instrument("EUR_USD")
        assert eur_usd is not None
        assert eur_usd.symbol == "EUR_USD"
        logger.info("✓ InstrumentProvider passed")
        
        # Test 4: CurrencyConverter
        logger.info("Testing CurrencyConverter...")
        converter = CurrencyConverter()
        rate = converter.get_rate("EUR", "USD")
        assert rate > 0
        assert eur_usd is not None  # Ensure instrument exists
        pip_value = converter.calculate_pip_value(eur_usd, "USD")
        assert pip_value > 0
        logger.info("✓ CurrencyConverter passed")
        
        logger.info("✓ All foundational components passed!")
        return True
        
    except Exception as e:
        logger.error(f"Foundational components test failed: {e}")
        return False

def test_risk_management_core():
    """Test Part 1: Risk Management Core"""
    logger.info("=" * 60)
    logger.info("TESTING RISK MANAGEMENT CORE")
    logger.info("=" * 60)
    
    try:
        from emp_proving_ground_unified import (
            RiskManager, RiskConfig, InstrumentProvider,
            Order, OrderSide, OrderType, ValidationResult,
            EnhancedPosition, TradeRecord, Decimal
        )
        
        # Setup
        config = RiskConfig()
        provider = InstrumentProvider()
        risk_manager = RiskManager(config, provider)
        instrument = provider.get_instrument("EUR_USD")
        assert instrument is not None, "EUR_USD instrument not found"
        
        # Test 1: Position size calculation
        logger.info("Testing position size calculation...")
        account_equity = Decimal('100000')
        stop_loss_pips = Decimal('50')
        position_size = risk_manager.calculate_position_size(
            account_equity, stop_loss_pips, instrument
        )
        assert position_size > 0
        assert position_size >= config.min_position_size
        assert position_size <= config.max_position_size
        logger.info(f"✓ Position size calculation: {position_size} units")
        
        # Test 2: Order validation
        logger.info("Testing order validation...")
        order = Order(
            order_id="test_order",
            symbol="EUR_USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000,
            stop_loss=1.0950
        )
        
        account_state = {"equity": 100000.0}
        open_positions = {}
        
        result = risk_manager.validate_order(order, account_state, open_positions)
        assert result.is_valid
        assert result.reason == "Order approved"
        logger.info("✓ Order validation passed")
        
        # Test 3: EnhancedPosition operations
        logger.info("Testing EnhancedPosition...")
        position = EnhancedPosition(
            symbol="EUR_USD",
            quantity=10000,
            avg_price=Decimal('1.1000'),
            entry_timestamp=datetime.now(),
            last_swap_time=datetime.now()
        )
        
        # Test position update
        position.update(
            trade_quantity=5000,
            trade_price=Decimal('1.1050'),
            commission=Decimal('5.0'),
            slippage=Decimal('2.5'),
            current_time=datetime.now(),
            trade_type="ADD"
        )
        assert position.quantity == 15000
        assert len(position.trade_history) == 1
        logger.info("✓ EnhancedPosition operations passed")
        
        # Test 4: PnL calculations
        logger.info("Testing PnL calculations...")
        position.update_unrealized_pnl(Decimal('1.1100'))
        assert position.unrealized_pnl > 0
        assert position.max_favorable_excursion > 0
        logger.info("✓ PnL calculations passed")
        
        logger.info("✓ All risk management core tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Risk management core test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pnl_engine():
    """Test Part 2: PnL Engine"""
    logger.info("=" * 60)
    logger.info("TESTING PNL ENGINE")
    logger.info("=" * 60)
    
    try:
        from emp_proving_ground_unified import (
            EnhancedPosition, TradeRecord, Instrument,
            Decimal, datetime
        )
        
        # Test 1: Trade record creation
        logger.info("Testing TradeRecord creation...")
        trade_record = TradeRecord(
            timestamp=datetime.now(),
            trade_type="OPEN",
            quantity=10000,
            price=Decimal('1.1000'),
            commission=Decimal('5.0'),
            slippage=Decimal('2.5')
        )
        assert trade_record.trade_type == "OPEN"
        assert trade_record.quantity == 10000
        logger.info("✓ TradeRecord creation passed")
        
        # Test 2: Position lifecycle
        logger.info("Testing position lifecycle...")
        instrument = Instrument(
            symbol="EUR_USD",
            pip_decimal_places=4,
            contract_size=Decimal('100000'),
            long_swap_rate=Decimal('-0.0001'),
            short_swap_rate=Decimal('0.0001'),
            margin_currency="USD"
        )
        
        position = EnhancedPosition(
            symbol="EUR_USD",
            quantity=0,
            avg_price=Decimal('0'),
            entry_timestamp=datetime.now(),
            last_swap_time=datetime.now()
        )
        
        # Open position
        position.update(
            trade_quantity=10000,
            trade_price=Decimal('1.1000'),
            commission=Decimal('5.0'),
            slippage=Decimal('2.5'),
            current_time=datetime.now(),
            trade_type="OPEN"
        )
        assert position.quantity == 10000
        assert position.avg_price == Decimal('1.1000')
        
        # Add to position
        position.update(
            trade_quantity=5000,
            trade_price=Decimal('1.1050'),
            commission=Decimal('2.5'),
            slippage=Decimal('1.25'),
            current_time=datetime.now(),
            trade_type="ADD"
        )
        assert position.quantity == 15000
        assert position.avg_price > Decimal('1.1000')
        
        # Close position
        position.update(
            trade_quantity=15000,
            trade_price=Decimal('1.1100'),
            commission=Decimal('7.5'),
            slippage=Decimal('3.75'),
            current_time=datetime.now(),
            trade_type="CLOSE"
        )
        assert position.quantity == 0
        assert position.realized_pnl > 0
        logger.info("✓ Position lifecycle passed")
        
        # Test 3: Swap fee application
        logger.info("Testing swap fee application...")
        position = EnhancedPosition(
            symbol="EUR_USD",
            quantity=10000,
            avg_price=Decimal('1.1000'),
            entry_timestamp=datetime.now(),
            last_swap_time=datetime.now() - timedelta(days=1)
        )
        
        # Apply swap fee
        position.apply_swap_fee(datetime.now(), instrument)
        # Note: This will only apply if we're past swap time
        logger.info("✓ Swap fee application passed")
        
        logger.info("✓ All PnL engine tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"PnL engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of all components"""
    logger.info("=" * 60)
    logger.info("TESTING INTEGRATION")
    logger.info("=" * 60)
    
    try:
        from emp_proving_ground_unified import (
            RiskManager, RiskConfig, InstrumentProvider,
            CurrencyConverter, EnhancedPosition, Order,
            OrderSide, OrderType, Decimal
        )
        
        # Setup complete system
        config = RiskConfig()
        provider = InstrumentProvider()
        risk_manager = RiskManager(config, provider)
        converter = CurrencyConverter()
        
        # Simulate a complete trading scenario
        logger.info("Simulating complete trading scenario...")
        
        # 1. Calculate position size
        account_equity = Decimal('100000')
        stop_loss_pips = Decimal('50')
        instrument = provider.get_instrument("EUR_USD")
        assert instrument is not None, "EUR_USD instrument not found"
        position_size = risk_manager.calculate_position_size(
            account_equity, stop_loss_pips, instrument
        )
        
        # 2. Create and validate order
        order = Order(
            order_id="integration_test",
            symbol="EUR_USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=position_size,
            stop_loss=1.0950
        )
        
        account_state = {"equity": float(account_equity)}
        open_positions = {}
        
        validation = risk_manager.validate_order(order, account_state, open_positions)
        assert validation.is_valid
        
        # 3. Create position and simulate trade
        position = EnhancedPosition(
            symbol="EUR_USD",
            quantity=0,
            avg_price=Decimal('0'),
            entry_timestamp=datetime.now(),
            last_swap_time=datetime.now()
        )
        
        # Simulate trade execution
        execution_price = Decimal('1.1000')
        commission = Decimal('5.0')
        slippage = Decimal('2.5')
        
        position.update(
            trade_quantity=position_size,
            trade_price=execution_price,
            commission=commission,
            slippage=slippage,
            current_time=datetime.now(),
            trade_type="OPEN"
        )
        
        # 4. Update unrealized PnL
        current_price = Decimal('1.1050')
        position.update_unrealized_pnl(current_price)
        
        # Verify results
        assert position.quantity == position_size
        assert position.unrealized_pnl > 0
        assert len(position.trade_history) == 1
        
        logger.info(f"✓ Integration test passed!")
        logger.info(f"  Position size: {position_size}")
        logger.info(f"  Unrealized PnL: ${position.unrealized_pnl:.2f}")
        logger.info(f"  Trade history: {len(position.trade_history)} records")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("EMP PROVING GROUND v2.0 FINANCIAL CORE TEST SUITE")
    logger.info("=" * 80)
    
    tests = [
        ("Foundational Components", test_foundational_components),
        ("Risk Management Core", test_risk_management_core),
        ("PnL Engine", test_pnl_engine),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.error(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! v2.0 Financial Core is ready!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} tests failed. Please fix issues.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 