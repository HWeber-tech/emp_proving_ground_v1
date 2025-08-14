# ruff: noqa: I001
"""
Sensory Cortex v2.2 - Production Example

Demonstrates real-world usage of the sensory cortex system in a production-like environment.
Shows proper initialization, data feeding, and result interpretation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict

import numpy as np

import os
import sys

# Ensure project root on path for local run
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensory.core.base import InstrumentMeta, MarketData, OrderBookSnapshot
from src.sensory.orchestration.master_orchestrator import MasterOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDataSimulator:
    """
    Simulates realistic production data feed for demonstration.
    In real production, this would be replaced with actual market data feeds.
    """
    
    def __init__(self, symbol: str = "EURUSD"):
        """Initialize production data simulator."""
        self.symbol = symbol
        self.current_price = 1.1000
        self.current_time = datetime.utcnow()
        self.tick_count = 0
        
        # Market session characteristics
        self.session_volatility = {
            'asian': 0.0002,
            'london': 0.0005,
            'ny': 0.0007,
            'overlap': 0.0009
        }
        
    def get_current_session(self) -> str:
        """Determine current market session."""
        hour = self.current_time.hour
        
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 13:
            return 'london'
        elif 13 <= hour < 17:
            return 'overlap'  # London-NY overlap
        elif 17 <= hour < 22:
            return 'ny'
        else:
            return 'asian'
    
    def generate_tick(self) -> MarketData:
        """Generate realistic market tick."""
        session = self.get_current_session()
        volatility = self.session_volatility[session]
        
        # Generate price movement
        price_change = np.random.normal(0, volatility)
        self.current_price += price_change
        
        # Generate volume based on session
        base_volume = {
            'asian': 500,
            'london': 1500,
            'ny': 2000,
            'overlap': 3000
        }[session]
        
        volume = np.random.exponential(base_volume)
        
        # Generate spread based on volatility
        spread = max(0.00015, volatility * 2 + np.random.uniform(0.00005, 0.00015))
        
        # Create market data
        market_data = MarketData(
            symbol=self.symbol,
            timestamp=self.current_time,
            open=self.current_price - 0.0002,
            high=self.current_price + 0.0003,
            low=self.current_price - 0.0005,
            close=self.current_price,
            volume=volume,
            bid=self.current_price - spread/2,
            ask=self.current_price + spread/2,
            spread=spread,
            mid_price=self.current_price
        )
        
        # Advance time
        self.current_time += timedelta(seconds=np.random.uniform(1, 10))
        self.tick_count += 1
        
        return market_data
    
    def generate_order_book(self, mid_price: float) -> OrderBookSnapshot:
        """Generate realistic order book."""
        from core.base import OrderBookLevel
        
        depth = 10
        bids = []
        asks = []
        
        for i in range(depth):
            # Price levels
            bid_price = mid_price - (i + 1) * 0.00005
            ask_price = mid_price + (i + 1) * 0.00005
            
            # Volume decreases with distance from mid
            base_volume = 1000000
            volume_decay = 0.8 ** i
            bid_volume = base_volume * volume_decay * np.random.uniform(0.5, 1.5)
            ask_volume = base_volume * volume_decay * np.random.uniform(0.5, 1.5)
            
            bids.append(OrderBookLevel(price=bid_price, volume=bid_volume, order_count=max(1, int(volume_decay * 10))))
            asks.append(OrderBookLevel(price=ask_price, volume=ask_volume, order_count=max(1, int(volume_decay * 10))))
        
        return OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=self.current_time,
            bids=bids,
            asks=asks
        )


class ProductionMonitor:
    """
    Monitors system performance and health in production environment.
    """
    
    def __init__(self):
        """Initialize production monitor."""
        self.start_time = datetime.utcnow()
        self.tick_count = 0
        self.synthesis_results = []
        self.performance_metrics = {
            'processing_times': [],
            'confidence_levels': [],
            'consensus_levels': [],
            'signal_strengths': [],
            'regime_changes': 0,
            'warnings_count': 0,
            'errors_count': 0
        }
        self.last_regime = None
        
    def record_synthesis(self, result) -> None:
        """Record synthesis result for monitoring."""
        self.tick_count += 1
        self.synthesis_results.append(result)
        
        # Update performance metrics
        self.performance_metrics['processing_times'].append(result.processing_time_ms)
        self.performance_metrics['confidence_levels'].append(result.confidence)
        self.performance_metrics['consensus_levels'].append(result.consensus_level)
        self.performance_metrics['signal_strengths'].append(abs(result.signal_strength))
        self.performance_metrics['warnings_count'] += len(result.warnings)
        
        # Track regime changes
        if self.last_regime and self.last_regime != result.regime:
            self.performance_metrics['regime_changes'] += 1
        self.last_regime = result.regime
        
        # Keep only recent results
        if len(self.synthesis_results) > 1000:
            self.synthesis_results.pop(0)
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        if not self.performance_metrics['processing_times']:
            return {'status': 'No data available'}
        
        runtime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'runtime_seconds': runtime,
            'total_ticks': self.tick_count,
            'ticks_per_second': self.tick_count / max(runtime, 1),
            'avg_processing_time_ms': np.mean(self.performance_metrics['processing_times']),
            'max_processing_time_ms': np.max(self.performance_metrics['processing_times']),
            'avg_confidence': np.mean(self.performance_metrics['confidence_levels']),
            'avg_consensus': np.mean(self.performance_metrics['consensus_levels']),
            'avg_signal_strength': np.mean(self.performance_metrics['signal_strengths']),
            'regime_changes': self.performance_metrics['regime_changes'],
            'warnings_count': self.performance_metrics['warnings_count'],
            'errors_count': self.performance_metrics['errors_count']
        }
    
    def print_status(self) -> None:
        """Print current status."""
        summary = self.get_performance_summary()
        
        print(f"\n📊 Production Monitor Status")
        print(f"   Runtime: {summary.get('runtime_seconds', 0):.1f}s")
        print(f"   Ticks Processed: {summary.get('total_ticks', 0)}")
        print(f"   Processing Rate: {summary.get('ticks_per_second', 0):.1f} ticks/sec")
        print(f"   Avg Processing Time: {summary.get('avg_processing_time_ms', 0):.1f}ms")
        print(f"   Avg Confidence: {summary.get('avg_confidence', 0):.3f}")
        print(f"   Avg Consensus: {summary.get('avg_consensus', 0):.3f}")
        print(f"   Regime Changes: {summary.get('regime_changes', 0)}")
        print(f"   Warnings: {summary.get('warnings_count', 0)}")


class ProductionTradingSystem:
    """
    Example production trading system using the sensory cortex.
    Demonstrates proper integration and usage patterns.
    """
    
    def __init__(self, symbol: str = "EURUSD"):
        """Initialize production trading system."""
        self.symbol = symbol
        
        # Create instrument metadata
        self.instrument_meta = InstrumentMeta(
            symbol=symbol,
            base_currency=symbol[:3],
            quote_currency=symbol[3:],
            pip_size=0.0001,
            lot_size=100000,
            min_lot=0.01,
            max_lot=100.0,
            spread_typical=1.5,
            session_start="00:00",
            session_end="23:59"
        )
        
        # Initialize sensory cortex
        self.sensory_cortex = MasterOrchestrator(self.instrument_meta)
        
        # Initialize components
        self.data_simulator = ProductionDataSimulator(symbol)
        self.monitor = ProductionMonitor()
        
        # Trading state
        self.position = 0.0  # Current position size
        self.entry_price = 0.0
        self.pnl = 0.0
        
        # Trading parameters
        self.min_confidence = 0.6
        self.min_consensus = 0.5
        self.position_size = 0.1  # 0.1 lots
        
        logger.info(f"Production Trading System initialized for {symbol}")
    
    async def process_tick(self, market_data: MarketData, order_book: OrderBookSnapshot) -> None:
        """Process single market tick."""
        try:
            # Update sensory cortex
            synthesis = await self.sensory_cortex.update(market_data, order_book)
            
            # Record for monitoring
            self.monitor.record_synthesis(synthesis)
            
            # Make trading decision
            await self._make_trading_decision(synthesis, market_data)
            
            # Log significant events
            if synthesis.warnings:
                logger.warning(f"Synthesis warnings: {synthesis.warnings}")
            
            if synthesis.confidence > 0.8:
                logger.info(f"High confidence signal: {synthesis.signal_strength:.3f} "
                           f"(confidence: {synthesis.confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            self.monitor.performance_metrics['errors_count'] += 1
    
    async def _make_trading_decision(self, synthesis, market_data: MarketData) -> None:
        """Make trading decision based on synthesis."""
        signal = synthesis.signal_strength
        confidence = synthesis.confidence
        consensus = synthesis.consensus_level
        
        # Check if signal meets criteria
        if confidence < self.min_confidence or consensus < self.min_consensus:
            return  # No action
        
        current_price = market_data.close
        
        # Entry logic
        if self.position == 0.0:  # No position
            if signal > 0.5:  # Strong bullish signal
                self._enter_position(1.0, current_price, "Bullish signal")
            elif signal < -0.5:  # Strong bearish signal
                self._enter_position(-1.0, current_price, "Bearish signal")
        
        # Exit logic
        elif self.position != 0.0:  # Have position
            # Exit on opposite signal
            if (self.position > 0 and signal < -0.3) or (self.position < 0 and signal > 0.3):
                self._exit_position(current_price, "Signal reversal")
            
            # Exit on low confidence
            elif confidence < 0.3:
                self._exit_position(current_price, "Low confidence")
    
    def _enter_position(self, direction: float, price: float, reason: str) -> None:
        """Enter trading position."""
        self.position = direction * self.position_size
        self.entry_price = price
        
        logger.info(f"ENTER: {self.position:.2f} lots at {price:.5f} - {reason}")
    
    def _exit_position(self, price: float, reason: str) -> None:
        """Exit trading position."""
        if self.position != 0.0:
            # Calculate P&L
            if self.position > 0:  # Long position
                pnl_pips = (price - self.entry_price) / self.instrument_meta.pip_size
            else:  # Short position
                pnl_pips = (self.entry_price - price) / self.instrument_meta.pip_size
            
            position_pnl = pnl_pips * abs(self.position) * 10  # $10 per pip per lot
            self.pnl += position_pnl
            
            logger.info(f"EXIT: {self.position:.2f} lots at {price:.5f} - {reason} "
                       f"(P&L: ${position_pnl:.2f}, Total: ${self.pnl:.2f})")
            
            self.position = 0.0
            self.entry_price = 0.0
    
    async def run_production_simulation(self, duration_minutes: int = 60) -> None:
        """Run production simulation."""
        logger.info(f"Starting production simulation for {duration_minutes} minutes")
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        tick_count = 0
        last_status_time = start_time
        
        while datetime.utcnow() < end_time:
            try:
                # Generate market data
                market_data = self.data_simulator.generate_tick()
                order_book = self.data_simulator.generate_order_book(market_data.close)
                
                # Process tick
                await self.process_tick(market_data, order_book)
                
                tick_count += 1
                
                # Print status every 30 seconds
                if (datetime.utcnow() - last_status_time).total_seconds() > 30:
                    self.monitor.print_status()
                    self._print_trading_status()
                    last_status_time = datetime.utcnow()
                
                # Small delay to simulate realistic tick rate
                await asyncio.sleep(0.1)
                
            except KeyboardInterrupt:
                logger.info("Simulation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in simulation: {e}")
                continue
        
        # Final status
        logger.info(f"Production simulation completed - {tick_count} ticks processed")
        self.monitor.print_status()
        self._print_trading_status()
        self._print_final_summary()
    
    def _print_trading_status(self) -> None:
        """Print current trading status."""
        print(f"\n💰 Trading Status")
        print(f"   Position: {self.position:.2f} lots")
        if self.position != 0.0:
            print(f"   Entry Price: {self.entry_price:.5f}")
        print(f"   Total P&L: ${self.pnl:.2f}")
    
    def _print_final_summary(self) -> None:
        """Print final summary."""
        print(f"\n🎯 Final Summary")
        print(f"   Symbol: {self.symbol}")
        print(f"   Total P&L: ${self.pnl:.2f}")
        
        # System health
        health = self.sensory_cortex.get_system_health()
        print(f"   System Health: {health['overall_health']:.3f}")
        print(f"   Healthy Engines: {health['healthy_engines']}/{health['total_engines']}")
        
        # Performance summary
        perf = self.monitor.get_performance_summary()
        print(f"   Avg Processing Time: {perf.get('avg_processing_time_ms', 0):.1f}ms")
        print(f"   Avg Confidence: {perf.get('avg_confidence', 0):.3f}")
        print(f"   Regime Changes: {perf.get('regime_changes', 0)}")


async def main():
    """Main production example."""
    print("🚀 Sensory Cortex v2.2 - Production Example")
    print("=" * 60)
    
    # Create production trading system
    trading_system = ProductionTradingSystem("EURUSD")
    
    # Run simulation
    try:
        await trading_system.run_production_simulation(duration_minutes=5)  # 5 minute demo
    except KeyboardInterrupt:
        print("\n⏹️  Simulation stopped by user")
    
    print("\n✅ Production example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

