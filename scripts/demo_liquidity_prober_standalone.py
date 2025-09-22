"""
Standalone Liquidity Prober Demonstration
Demonstrates PROBE-40: The Predator's Sonar functionality
"""

import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class TradeIntent:
    """Simplified trade intent"""

    symbol: str
    side: str
    quantity: Decimal
    order_type: str = "LIMIT"
    price: Optional[Decimal] = None
    liquidity_confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockBroker:
    """Mock broker for demonstration"""

    def __init__(self):
        self.orders = {}
        self.order_counter = 0

    async def place_order(
        self,
        symbol_name: str,
        order_type: str,
        side: str,
        volume: float,
        price: Optional[float] = None,
    ) -> Optional[str]:
        """Mock order placement"""
        self.order_counter += 1
        order_id = f"mock_order_{self.order_counter}"

        # Simulate different fill rates based on price level
        if price:
            if abs(price - 1.1002) < 0.0001:  # Simulate iceberg
                fill_rate = 0.3  # Hidden liquidity
            elif price > 1.1000:
                fill_rate = 0.8  # Good liquidity
            else:
                fill_rate = 0.4  # Lower liquidity
        else:
            fill_rate = 0.6

        filled_volume = volume * fill_rate

        self.orders[order_id] = {
            "order_id": order_id,
            "symbol": symbol_name,
            "side": side,
            "volume": volume,
            "filled_volume": filled_volume,
            "status": "filled",
            "price": price,
        }

        logger.info(
            f"Mock order: {side} {volume} {symbol_name} @ {price} -> filled {filled_volume}"
        )
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        """Mock order cancellation"""
        if order_id in self.orders:
            logger.info(f"Cancelled order: {order_id}")
            return True
        return False

    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders"""
        return list(self.orders.values())


class LiquidityProber:
    """Core LiquidityProber implementation"""

    def __init__(self, broker: MockBroker, config: Optional[Dict[str, Any]] = None):
        self.broker = broker
        self.config = config or {
            "probe_size": 0.001,
            "timeout_seconds": 2.0,
            "max_concurrent_probes": 5,
        }

    async def probe_liquidity(
        self, symbol: str, price_levels: List[float], side: str
    ) -> Dict[float, float]:
        """Probe liquidity at specified price levels"""
        logger.info(f"Probing liquidity for {symbol} {side} at levels: {price_levels}")

        results = {}

        for price in price_levels:
            try:
                # Place IOC order (Immediate Or Cancel)
                order_id = await self.broker.place_order(
                    symbol_name=symbol,
                    order_type="MARKET",
                    side=side.upper(),
                    volume=self.config["probe_size"],
                    price=price,
                )

                if order_id:
                    # Get filled volume
                    orders = self.broker.get_orders()
                    order = next((o for o in orders if o["order_id"] == order_id), None)
                    if order:
                        filled_volume = float(order["filled_volume"])
                        results[price] = filled_volume
                        logger.info(f"  Price {price}: {filled_volume:.6f} lots filled")
                    else:
                        results[price] = 0.0
                        logger.info(f"  Price {price}: 0.0 lots (order not found)")
                else:
                    results[price] = 0.0
                    logger.info(f"  Price {price}: 0.0 lots (order failed)")

                # Cancel any remaining order
                if order_id:
                    await self.broker.cancel_order(order_id)

            except Exception as e:
                logger.error(f"Error probing price {price}: {e}")
                results[price] = 0.0

        return results

    def calculate_liquidity_confidence_score(
        self, probe_results: Dict[float, float], intended_volume: float
    ) -> float:
        """Calculate liquidity confidence score"""
        if not probe_results or intended_volume <= 0:
            return 0.0

        total_liquidity = sum(probe_results.values())
        if total_liquidity <= 0:
            return 0.0

        # Coverage ratio
        coverage_ratio = min(total_liquidity / intended_volume, 1.0)

        # Distribution quality (how evenly liquidity is distributed)
        values = list(probe_results.values())
        if len(values) <= 1:
            distribution_quality = 0.5
        else:
            avg = sum(values) / len(values)
            variance = sum((v - avg) ** 2 for v in values) / len(values)
            distribution_quality = max(0.0, 1.0 - (variance / (avg**2 + 1e-10)))

        # Combined score
        score = coverage_ratio * 0.7 + distribution_quality * 0.3

        logger.info(
            f"Liquidity confidence score: {score:.3f} (coverage: {coverage_ratio:.3f}, distribution: {distribution_quality:.3f})"
        )
        return score

    def get_probe_summary(self, probe_results: Dict[float, float]) -> Dict[str, Any]:
        """Get summary of probe results"""
        if not probe_results:
            return {
                "total_levels": 0,
                "total_liquidity": 0.0,
                "avg_liquidity": 0.0,
                "best_levels": [],
                "empty_levels": 0,
            }

        total_liquidity = sum(probe_results.values())
        total_levels = len(probe_results)
        avg_liquidity = total_liquidity / total_levels

        # Sort by liquidity (descending)
        sorted_levels = sorted(probe_results.items(), key=lambda x: x[1], reverse=True)
        best_levels = [{"price": price, "volume": volume} for price, volume in sorted_levels[:3]]
        empty_levels = sum(1 for v in probe_results.values() if v == 0)

        return {
            "total_levels": total_levels,
            "total_liquidity": total_liquidity,
            "avg_liquidity": avg_liquidity,
            "best_levels": best_levels,
            "empty_levels": empty_levels,
        }


class RiskGateway:
    """Simplified RiskGateway for demonstration"""

    def __init__(self, liquidity_prober: LiquidityProber, probe_threshold: float = 1.0):
        self.liquidity_prober = liquidity_prober
        self.probe_threshold = probe_threshold

    async def validate_trade_intent(
        self, intent: TradeIntent, portfolio_state: Dict[str, Any]
    ) -> Optional[TradeIntent]:
        """Validate trade intent with liquidity checking"""
        quantity = float(intent.quantity)

        if quantity < self.probe_threshold:
            logger.info(
                f"Trade below threshold ({quantity} < {self.probe_threshold}), skipping liquidity probe"
            )
            return intent

        # Generate price levels around intended price
        current_price = float(intent.price) if intent.price else 1.1000
        spread = 0.001

        if intent.side.upper() == "BUY":
            price_levels = [current_price + i * spread for i in range(-2, 3)]
        else:
            price_levels = [current_price - i * spread for i in range(-2, 3)]

        # Perform liquidity probing
        probe_results = await self.liquidity_prober.probe_liquidity(
            intent.symbol, price_levels, intent.side
        )

        # Calculate confidence score
        confidence_score = self.liquidity_prober.calculate_liquidity_confidence_score(
            probe_results, quantity
        )

        # Update intent with confidence score
        intent.liquidity_confidence_score = confidence_score

        # Log results
        summary = self.liquidity_prober.get_probe_summary(probe_results)
        logger.info(f"Trade validation complete:")
        logger.info(f"  Confidence score: {confidence_score:.3f}")
        logger.info(f"  Total liquidity: {summary['total_liquidity']:.6f} lots")
        logger.info(f"  Best levels: {summary['best_levels']}")

        # Reject if confidence too low
        if confidence_score < 0.3:
            logger.warning(
                f"Trade REJECTED: insufficient liquidity (score: {confidence_score:.3f})"
            )
            return None

        return intent


async def demonstrate_liquidity_probing():
    """Demonstrate liquidity probing functionality"""
    print("\n" + "=" * 70)
    print("🦈 PREDATOR'S SONAR - LIQUIDITY PROBING DEMONSTRATION")
    print("=" * 70)

    # Create components
    broker = MockBroker()
    liquidity_prober = LiquidityProber(broker)

    # Test scenarios
    scenarios = [
        {
            "symbol": "EURUSD",
            "side": "buy",
            "price_levels": [1.0998, 1.0999, 1.1000, 1.1001, 1.1002],
            "description": "Normal market probing",
        },
        {
            "symbol": "EURUSD",
            "side": "sell",
            "price_levels": [1.1002, 1.1001, 1.1000, 1.0999, 1.0998],
            "description": "Sell-side probing",
        },
    ]

    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['description']}")
        print(f"Symbol: {scenario['symbol']}, Side: {scenario['side']}")

        # Perform liquidity probing
        results = await liquidity_prober.probe_liquidity(
            scenario["symbol"], scenario["price_levels"], scenario["side"]
        )

        # Display results
        print("\n🔍 Probe Results:")
        for price, volume in results.items():
            print(f"  Price {price}: {volume:.6f} lots")

        # Calculate confidence scores
        volumes = [0.5, 1.0, 2.0]
        print("\n📈 Liquidity Confidence Scores:")
        for volume in volumes:
            score = liquidity_prober.calculate_liquidity_confidence_score(results, volume)
            print(f"  Volume {volume} lots: {score:.3f} ({score * 100:.1f}%)")

        # Show summary
        summary = liquidity_prober.get_probe_summary(results)
        print(f"\n📋 Summary:")
        print(f"  Total levels: {summary['total_levels']}")
        print(f"  Total liquidity: {summary['total_liquidity']:.6f} lots")
        print(f"  Average: {summary['avg_liquidity']:.6f} lots")
        print(f"  Best levels: {summary['best_levels']}")
        print(f"  Empty levels: {summary['empty_levels']}")

        print("-" * 70)


async def demonstrate_risk_gateway_integration():
    """Demonstrate RiskGateway integration"""
    print("\n" + "=" * 70)
    print("🛡️ RISK GATEWAY INTEGRATION DEMONSTRATION")
    print("=" * 70)

    # Create components
    broker = MockBroker()
    liquidity_prober = LiquidityProber(broker)
    risk_gateway = RiskGateway(liquidity_prober, probe_threshold=1.0)

    # Test trade intents
    test_trades = [
        {
            "symbol": "EURUSD",
            "side": "BUY",
            "quantity": Decimal("0.5"),  # Below threshold
            "price": Decimal("1.1000"),
            "description": "Small trade (no probing)",
        },
        {
            "symbol": "EURUSD",
            "side": "BUY",
            "quantity": Decimal("2.0"),  # Above threshold
            "price": Decimal("1.1000"),
            "description": "Large trade (triggers probing)",
        },
        {
            "symbol": "EURUSD",
            "side": "SELL",
            "quantity": Decimal("3.0"),  # Large trade
            "price": Decimal("1.1001"),
            "description": "Large sell trade",
        },
    ]

    for trade in test_trades:
        print(f"\n🎯 Testing: {trade['description']}")
        print(f"Trade: {trade['side']} {trade['quantity']} {trade['symbol']} @ {trade['price']}")

        # Create trade intent
        intent = TradeIntent(
            symbol=trade["symbol"],
            side=trade["side"],
            quantity=trade["quantity"],
            price=trade["price"],
        )

        # Mock portfolio state
        portfolio_state = {"equity": 10000.0, "current_price": 1.1000, "pip_value": 0.0001}

        # Validate trade intent
        validated_intent = await risk_gateway.validate_trade_intent(intent, portfolio_state)

        if validated_intent:
            print(f"✅ Trade APPROVED")
            if validated_intent.liquidity_confidence_score:
                print(f"   Liquidity confidence: {validated_intent.liquidity_confidence_score:.3f}")
        else:
            print(f"❌ Trade REJECTED")

        print("-" * 70)


async def demonstrate_iceberg_detection():
    """Demonstrate iceberg detection"""
    print("\n" + "=" * 70)
    print("🧊 ICEBERG DETECTION DEMONSTRATION")
    print("=" * 70)

    # Create broker with iceberg simulation
    class IcebergBroker(MockBroker):
        def __init__(self):
            super().__init__()
            self.iceberg_price = 1.1002

        async def place_order(self, symbol_name, order_type, side, volume, price=None):
            # Simulate iceberg at specific price
            if price and abs(price - self.iceberg_price) < 0.0001:
                fill_rate = 0.1  # Very low visible liquidity (iceberg)
            else:
                fill_rate = 0.8  # Normal liquidity

            self.order_counter += 1
            order_id = f"iceberg_order_{self.order_counter}"

            filled_volume = volume * fill_rate

            self.orders[order_id] = {
                "order_id": order_id,
                "symbol": symbol_name,
                "side": side,
                "volume": volume,
                "filled_volume": filled_volume,
                "status": "filled",
                "price": price,
            }

            logger.info(
                f"Iceberg order: {side} {volume} @ {price} -> {filled_volume} (fill rate: {fill_rate})"
            )
            return order_id

    iceberg_broker = IcebergBroker()
    liquidity_prober = LiquidityProber(iceberg_broker)

    # Probe for iceberg
    price_levels = [1.1000, 1.1001, 1.1002, 1.1003, 1.1004]
    results = await liquidity_prober.probe_liquidity("EURUSD", price_levels, "buy")

    print("\n🔍 Iceberg Detection Results:")
    for price, volume in results.items():
        print(f"  Price {price}: {volume:.6f} lots")

    # Analyze for iceberg patterns
    print("\n🧊 Iceberg Analysis:")
    avg_liquidity = sum(results.values()) / len(results)

    for price, volume in results.items():
        deviation = abs(volume - avg_liquidity) / avg_liquidity if avg_liquidity > 0 else 0
        if deviation > 0.5 and volume < avg_liquidity:
            print(f"  ⚠️  Potential iceberg at {price}: {volume:.6f} (deviation: {deviation:.2f})")
        else:
            print(f"  ✅ Normal liquidity at {price}: {volume:.6f}")


async def main():
    """Main demonstration function"""
    print("🦈 Starting Predator's Sonar Demonstration...")
    print("This demonstrates the LiquidityProber actively interrogating the market")

    try:
        await demonstrate_liquidity_probing()
        await demonstrate_risk_gateway_integration()
        await demonstrate_iceberg_detection()

        print("\n" + "=" * 70)
        print("✅ DEMONSTRATION COMPLETE")
        print("=" * 70)
        print("The Predator's Sonar is now active!")
        print("- ✅ LiquidityProber sends IOC orders to probe market depth")
        print("- ✅ RiskGateway integrates liquidity validation")
        print("- ✅ Iceberg orders can be detected through probing")
        print("- ✅ Confidence scores guide trade decisions")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
