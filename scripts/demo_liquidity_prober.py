"""
Liquidity Prober Demonstration Script
Demonstrates PROBE-40: The Predator's Sonar functionality

This script demonstrates the LiquidityProber in action, showing how it
actively probes the order book for hidden liquidity and icebergs.
"""

import asyncio
import logging
from decimal import Decimal
from unittest.mock import Mock

from src.trading.execution.liquidity_prober import LiquidityProber
from src.trading.integration.fix_broker_interface import FIXBrokerInterface
from src.trading.risk.position_sizer import PositionSizer
from src.trading.risk.risk_gateway import RiskGateway

# Legacy type import (provide local fallback if unavailable)
try:
    from src.core.events import TradeIntent  # type: ignore
except Exception:
    from dataclasses import dataclass  # local-only fallback
    @dataclass
    class TradeIntent:  # type: ignore
        symbol: str
        side: str
        quantity: Decimal
        order_type: str
        price: Decimal
        metadata: dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DummyEventBus:
    async def emit(self, event_name: str, payload):
        logger.info(f"Event emitted: {event_name} -> {payload}")


class DummyInitiator:
    def __init__(self):
        self.sent_messages = []
    def send_message(self, msg) -> bool:
        self.sent_messages.append(msg)
        return True


async def demonstrate_liquidity_probing():
    """Demonstrate the LiquidityProber functionality"""
    print("\n" + "="*60)
    print("🦈 PREDATOR'S SONAR DEMONSTRATION")
    print("="*60)
    
    # Create broker and liquidity prober (paper via DummyInitiator)
    trade_queue = asyncio.Queue()
    broker = FIXBrokerInterface(DummyEventBus(), trade_queue, DummyInitiator())
    await broker.start()
    liquidity_prober = LiquidityProber(broker, config={
        'probe_size': 0.001,
        'timeout_seconds': 2.0,
        'max_concurrent_probes': 5
    })
    
    # Test scenarios
    scenarios = [
        {
            'symbol': 'EURUSD',
            'side': 'buy',
            'price_levels': [1.1000, 1.1001, 1.1002, 1.1003, 1.1004],
            'description': 'Normal market conditions'
        },
        {
            'symbol': 'EURUSD',
            'side': 'sell',
            'price_levels': [1.0998, 1.0997, 1.0996, 1.0995, 1.0994],
            'description': 'Sell-side probing'
        },
        {
            'symbol': 'GBPUSD',
            'side': 'buy',
            'price_levels': [1.2500, 1.2501, 1.2502, 1.2503, 1.2504],
            'description': 'Different symbol'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['description']}")
        print(f"Symbol: {scenario['symbol']}, Side: {scenario['side']}")
        print(f"Price levels: {scenario['price_levels']}")
        
        # Perform liquidity probing
        results = await liquidity_prober.probe_liquidity(
            scenario['symbol'],
            scenario['price_levels'],
            scenario['side']
        )
        
        # Display results
        print("\n🔍 Probe Results:")
        for price, volume in results.items():
            print(f"  Price {price}: {volume:.6f} lots filled")
        
        # Calculate confidence scores for different intended volumes
        volumes = [0.5, 1.0, 2.0]
        print("\n📈 Liquidity Confidence Scores:")
        for volume in volumes:
            score = liquidity_prober.calculate_liquidity_confidence_score(results, volume)
            print(f"  Volume {volume} lots: {score:.3f} ({score*100:.1f}%)")
        
        # Show summary
        summary = liquidity_prober.get_probe_summary(results)
        print(f"\n📋 Summary:")
        print(f"  Total levels probed: {summary['total_levels']}")
        print(f"  Total liquidity found: {summary['total_liquidity']:.6f} lots")
        print(f"  Average per level: {summary['avg_liquidity']:.6f} lots")
        print(f"  Best levels: {summary['best_levels']}")
        print(f"  Empty levels: {summary['empty_levels']}")
        
        print("-" * 60)


async def demonstrate_risk_gateway_integration():
    """Demonstrate LiquidityProber integration with RiskGateway"""
    print("\n" + "="*60)
    print("🛡️ RISK GATEWAY INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Create components
    trade_queue = asyncio.Queue()
    broker = FIXBrokerInterface(DummyEventBus(), trade_queue, DummyInitiator())
    await broker.start()
    liquidity_prober = LiquidityProber(broker, config={
        'probe_size': 0.001,
        'timeout_seconds': 2.0,
        'max_concurrent_probes': 3
    })
    
    # Mock dependencies
    mock_strategy_registry = Mock()
    mock_position_sizer = PositionSizer(risk_percentage=0.02)
    mock_portfolio_monitor = Mock()
    
    # Create RiskGateway with liquidity probing
    risk_gateway = RiskGateway(
        mock_strategy_registry,
        mock_position_sizer,
        mock_portfolio_monitor,
        liquidity_prober=liquidity_prober,
        liquidity_probe_threshold=1.0  # Trigger for trades >= 1 lot
    )
    
    # Test trade intents
    test_trades = [
        {
            'symbol': 'EURUSD',
            'side': 'BUY',
            'quantity': Decimal('0.5'),  # Small trade - no probing
            'price': Decimal('1.1000'),
            'description': 'Small trade (below threshold)'
        },
        {
            'symbol': 'EURUSD',
            'side': 'BUY',
            'quantity': Decimal('2.0'),  # Large trade - triggers probing
            'price': Decimal('1.1000'),
            'description': 'Large trade (above threshold)'
        }
    ]
    
    for trade in test_trades:
        print(f"\n🎯 Testing: {trade['description']}")
        print(f"Trade: {trade['side']} {trade['quantity']} {trade['symbol']} @ {trade['price']}")
        
        # Create trade intent
        intent = TradeIntent(
            symbol=trade['symbol'],
            side=trade['side'],
            quantity=trade['quantity'],
            order_type='2',  # Limit order
            price=trade['price'],
            metadata={'stop_loss_pips': 50}
        )
        
        # Mock portfolio state
        portfolio_state = {
            'equity': 10000.0,
            'current_price': 1.1000,
            'pip_value': 0.0001,
            'current_daily_drawdown': 0.01,
            'open_positions_count': 2
        }
        
        # Validate trade intent
        validated_intent = await risk_gateway.validate_trade_intent(intent, portfolio_state)
        
        if validated_intent:
            print(f"✅ Trade APPROVED")
            print(f"   Liquidity confidence: {validated_intent.liquidity_confidence_score}")
            print(f"   Final quantity: {validated_intent.quantity}")
        else:
            print(f"❌ Trade REJECTED")
        
        print("-" * 60)


async def demonstrate_iceberg_detection():
    """Demonstrate detection of iceberg orders"""
    print("\n" + "="*60)
    print("🧊 ICEBERG DETECTION DEMONSTRATION")
    print("="*60)
    
    # Create mock broker with iceberg simulation
    class IcebergMockBroker:
        def __init__(self):
            self.iceberg_levels = {1.1002: 0.1}  # Hidden liquidity at 1.1002
            self.orders = {}
            self.order_counter = 0
        
        async def place_market_order(self, symbol: str, side: str, quantity: float):
            self.order_counter += 1
            order_id = f"iceberg_order_{self.order_counter}"
            price = 1.1002
            if abs(price - 1.1002) < 0.0001:
                visible_fill = min(quantity * 0.3, 0.001)
            else:
                visible_fill = quantity * 0.8
            self.orders[order_id] = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'status': 'FILLED',
                'filled_qty': visible_fill,
                'avg_px': price,
            }
            return order_id
    
    iceberg_broker = IcebergMockBroker()
    liquidity_prober = LiquidityProber(iceberg_broker, config={
        'probe_size': 0.001,
        'timeout_seconds': 2.0,
        'max_concurrent_probes': 5
    })
    
    # Probe for iceberg
    price_levels = [1.1000, 1.1001, 1.1002, 1.1003, 1.1004]
    results = await liquidity_prober.probe_liquidity("EURUSD", price_levels, "buy")
    
    print("\n🔍 Iceberg Detection Results:")
    for price, volume in results.items():
        print(f"  Price {price}: {volume:.6f} lots (visible)")
    
    # Analyze for iceberg patterns
    print("\n🧊 Iceberg Analysis:")
    expected_pattern = [0.0008, 0.0008, 0.0003, 0.0008, 0.0008]  # Lower at 1.1002
    
    for i, (price, actual) in enumerate(results.items()):
        expected = expected_pattern[i]
        deviation = abs(actual - expected) / expected if expected > 0 else 0
        if deviation > 0.5:
            print(f"  ⚠️  Potential iceberg at {price}: {actual:.6f} vs expected {expected:.6f}")
        else:
            print(f"  ✅ Normal liquidity at {price}: {actual:.6f}")


async def main():
    """Main demonstration function"""
    print("🦈 Starting Predator's Sonar Demonstration...")
    
    try:
        await demonstrate_liquidity_probing()
        await demonstrate_risk_gateway_integration()
        await demonstrate_iceberg_detection()
        
        print("\n" + "="*60)
        print("✅ DEMONSTRATION COMPLETE")
        print("="*60)
        print("The Predator's Sonar is now active and ready to interrogate the market!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
