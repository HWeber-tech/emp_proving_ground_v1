#!/usr/bin/env python3
"""
Legacy: Live Trading Executor (OpenAPI/Mock-oriented) - Disabled in FIX-only build.
This module references cTrader interfaces not used in the FIX-only architecture.
"""

raise ImportError("LiveTradingExecutor is deprecated and disabled in FIX-only builds.")

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd

from .mock_ctrader_interface import (
    CTraderInterface, TradingConfig, MarketData, Order, Position,
    TradingMode, OrderType, OrderSide
)
from src.evolution.real_genetic_engine import RealGeneticEngine
from src.sensory.dimensions.enhanced_when_dimension import TemporalAnalyzer
from src.sensory.dimensions.enhanced_anomaly_dimension import PatternRecognitionDetector
from .strategy_manager import StrategyManager, StrategySignal
try:
    from .advanced_risk_manager import AdvancedRiskManager, RiskLimits  # deprecated
except Exception:  # pragma: no cover
    AdvancedRiskManager = None  # type: ignore
    class RiskLimits:  # type: ignore
        pass
from .performance_tracker import PerformanceTracker
from .order_book_analyzer import OrderBookAnalyzer
try:
    from src.core.interfaces import DecisionGenome  # legacy
except Exception:  # pragma: no cover
    DecisionGenome = object  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal from evolutionary strategy."""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: float = 0.01  # Default 0.01 lots
    timestamp: Optional[datetime] = None


@dataclass
class TradingPerformance:
    """Trading performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0


class LiveTradingExecutor:
    """
    Live trading executor that integrates evolutionary strategies with cTrader.
    
    This class manages:
    - Real-time strategy execution
    - Risk management
    - Performance tracking
    - Market analysis integration
    """
    
    def __init__(self, config: TradingConfig, symbols: List[str], 
                 max_positions: int = 5, max_risk_per_trade: float = 0.02):
        """
        Initialize the live trading executor.
        
        Args:
            config: Trading configuration
            symbols: List of symbols to trade
            max_positions: Maximum concurrent positions
            max_risk_per_trade: Maximum risk per trade (2% default)
        """
        self.config = config
        self.symbols = symbols
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        
        # Initialize components
        self.ctrader = CTraderInterface(config)
        self.genetic_engine = RealGeneticEngine(data_source="real")  # Use real data source
        self.regime_detector = TemporalAnalyzer() # Use TemporalAnalyzer for regime detection
        self.pattern_recognition = PatternRecognitionDetector() # Use PatternRecognitionDetector for pattern recognition
        self.strategy_manager = StrategyManager()  # Strategy integration
        
        # Advanced risk management
        risk_limits = RiskLimits()
        self.advanced_risk_manager = AdvancedRiskManager(risk_limits, self.strategy_manager) if AdvancedRiskManager else None
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker(initial_balance=100000.0)  # Default balance
        
        # Order book analysis
        self.order_book_analyzer = OrderBookAnalyzer(max_levels=20, history_window=1000)
        
        # State management
        self.connected = False
        self.running = False
        self.performance = TradingPerformance()
        self.signals = []
        self.risk_manager = LiveRiskManager(max_risk_per_trade)
        
        # Market data cache
        self.market_data = {}
        self.last_analysis = {}
        
        logger.info(f"Live trading executor initialized for {len(symbols)} symbols")
    
    async def start(self) -> bool:
        """
        Start the live trading executor.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            logger.info("Starting live trading executor...")
            
            # Connect to cTrader
            if not await self.ctrader.connect():
                logger.error("Failed to connect to cTrader")
                return False
            
            self.connected = True
            
            # Subscribe to market data
            for symbol in self.symbols:
                await self.ctrader.subscribe_to_symbol(symbol)
                logger.info(f"Subscribed to {symbol}")
            
            # Set up callbacks
            self.ctrader.add_callback('price_update', self._on_price_update)
            self.ctrader.add_callback('order_update', self._on_order_update)
            self.ctrader.add_callback('position_update', self._on_position_update)
            self.ctrader.add_callback('error', self._on_error)
            
            # Load evolved strategies from genetic engine
            await self._load_evolved_strategies()
            
            self.running = True
            logger.info("Live trading executor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start live trading executor: {e}")
            return False
    
    async def stop(self):
        """Stop the live trading executor."""
        self.running = False
        if self.connected:
            await self.ctrader.disconnect()
            self.connected = False
        logger.info("Live trading executor stopped")
    
    async def run_trading_cycle(self):
        """Run a complete trading cycle."""
        if not self.running:
            return
        
        try:
            # Step 1: Update market data
            await self._update_market_data()
            
            # Step 2: Update order book analysis
            await self._update_order_book_analysis()
            
            # Step 3: Perform market analysis
            await self._perform_market_analysis()
            
            # Step 4: Generate trading signals
            signals = await self._generate_trading_signals()
            
            # Step 5: Execute signals
            await self._execute_trading_signals(signals)
            
            # Step 6: Update performance and risk metrics
            self._update_performance()
            self._update_risk_metrics()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def _update_market_data(self):
        """Update market data for all symbols."""
        for symbol in self.symbols:
            market_data = self.ctrader.get_market_data(symbol)
            if market_data:
                self.market_data[symbol] = market_data
    
    async def _update_order_book_analysis(self):
        """Update order book analysis for all symbols."""
        for symbol in self.symbols:
            # Get order book data from cTrader (mock implementation)
            order_book_data = await self._get_order_book_data(symbol)
            if order_book_data:
                bids, asks = order_book_data
                self.order_book_analyzer.update_order_book(symbol, bids, asks)
                
                # Log order book insights
                analysis = self.order_book_analyzer.get_market_analysis(symbol)
                if analysis:
                    logger.debug(f"Order book analysis for {symbol}: spread={analysis['current']['spread']:.5f}, "
                               f"liquidity={analysis['current']['total_liquidity']:.1f}")
    
    async def _get_order_book_data(self, symbol: str) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        """Get order book data for a symbol (mock implementation)."""
        try:
            # Mock order book data - in real implementation, this would come from cTrader
            market_data = self.market_data.get(symbol)
            if not market_data:
                return None
            
            # Generate realistic order book around current price
            current_price = (market_data.bid + market_data.ask) / 2
            spread = market_data.ask - market_data.bid
            
            # Generate bid levels (descending prices)
            bids = []
            for i in range(10):
                price = current_price - (i * spread * 0.5)
                volume = np.random.uniform(0.1, 5.0)  # Random volume between 0.1 and 5 lots
                bids.append((price, volume))
            
            # Generate ask levels (ascending prices)
            asks = []
            for i in range(10):
                price = current_price + (i * spread * 0.5)
                volume = np.random.uniform(0.1, 5.0)  # Random volume between 0.1 and 5 lots
                asks.append((price, volume))
            
            return bids, asks
            
        except Exception as e:
            logger.error(f"Error getting order book data for {symbol}: {e}")
            return None
    
    async def _perform_market_analysis(self):
        """Perform market analysis for all symbols."""
        for symbol in self.symbols:
            if symbol in self.market_data:
                market_data = self.market_data[symbol]
                
                # Update regime detector with market data
                self.regime_detector.update_market_data(market_data)
                self.regime_detector.update_temporal_data(market_data)
                
                # Update pattern recognition with market data
                self.pattern_recognition.update_data(market_data)
                
                # Detect market regime
                regime = self.regime_detector.detect_market_regime()
                
                # Detect patterns
                patterns = self.pattern_recognition.detect_patterns(market_data)
                
                self.last_analysis[symbol] = {
                    'regime': regime,
                    'patterns': patterns,
                    'timestamp': datetime.now()
                }
    
    async def _generate_trading_signals(self) -> List[TradingSignal]:
        """Generate trading signals using evolved strategies."""
        signals = []
        
        for symbol in self.symbols:
            if symbol in self.market_data:
                market_data = self.market_data[symbol]
                
                # Convert market data to dictionary format for strategy manager
                market_data_dict = {
                    'symbol': symbol,
                    'timestamp': market_data.timestamp,
                    'open': market_data.open,
                    'high': market_data.high,
                    'low': market_data.low,
                    'close': market_data.close,
                    'volume': market_data.volume,
                    'bid': market_data.bid,
                    'ask': market_data.ask
                }
                
                # Use strategy manager to evaluate evolved strategies
                strategy_signal = self.strategy_manager.select_best_strategy(symbol, market_data_dict)
                
                if strategy_signal:
                    # Convert StrategySignal to TradingSignal
                    signal = TradingSignal(
                        symbol=strategy_signal.symbol,
                        action=strategy_signal.action,
                        confidence=strategy_signal.confidence,
                        entry_price=strategy_signal.entry_price,
                        stop_loss=strategy_signal.stop_loss,
                        take_profit=strategy_signal.take_profit,
                        volume=strategy_signal.volume,
                        timestamp=strategy_signal.timestamp
                    )
                    signals.append(signal)
        
        return signals
    
    async def _generate_signal_for_symbol(self, symbol: str, market_data: MarketData, 
                                        analysis: Dict) -> Optional[TradingSignal]:
        """Generate trading signal for a specific symbol."""
        try:
            # Get current price
            current_price = (market_data.bid + market_data.ask) / 2
            
            # Analyze market conditions
            regime = analysis['regime']
            patterns = analysis['patterns']
            
            # Use genetic engine to evaluate strategy
            # This is a simplified version - in practice, you'd use the evolved strategies
            signal = self._evaluate_market_conditions(symbol, current_price, regime, patterns)
            
            if signal:
                signal.timestamp = datetime.now()
                return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
        
        return None
    
    def _evaluate_market_conditions(self, symbol: str, price: float, regime, patterns) -> Optional[TradingSignal]:
        """Evaluate market conditions and generate signal."""
        # This is a simplified signal generation logic
        # In practice, you'd use the evolved strategies from the genetic engine
        
        # Check for bullish conditions
        bullish_conditions = 0
        bearish_conditions = 0
        
        # Regime analysis
        if regime.regime.value in ['trending_up', 'breakout']:
            bullish_conditions += 1
        elif regime.regime.value in ['trending_down']:
            bearish_conditions += 1
        
        # Pattern analysis
        for pattern in patterns[:3]:  # Top 3 patterns
            if pattern.pattern_type.value in ['ascending_triangle', 'bull_flag', 'double_bottom']:
                bullish_conditions += 1
            elif pattern.pattern_type.value in ['descending_triangle', 'bear_flag', 'double_top']:
                bearish_conditions += 1
        
        # Generate signal based on conditions
        if bullish_conditions > bearish_conditions and bullish_conditions >= 2:
            return TradingSignal(
                symbol=symbol,
                action='buy',
                confidence=min(0.9, 0.5 + (bullish_conditions * 0.1)),
                entry_price=price,
                stop_loss=price * 0.99,  # 1% stop loss
                take_profit=price * 1.02,  # 2% take profit
                volume=0.01
            )
        elif bearish_conditions > bullish_conditions and bearish_conditions >= 2:
            return TradingSignal(
                symbol=symbol,
                action='sell',
                confidence=min(0.9, 0.5 + (bearish_conditions * 0.1)),
                entry_price=price,
                stop_loss=price * 1.01,  # 1% stop loss
                take_profit=price * 0.98,  # 2% take profit
                volume=0.01
            )
        
        return None
    
    def _adjust_signal_with_order_book(self, signal: TradingSignal, order_book_analysis: Dict[str, Any]) -> TradingSignal:
        """Adjust trading signal based on order book analysis."""
        try:
            current = order_book_analysis.get('current', {})
            signals = order_book_analysis.get('signals', {})
            
            # Adjust confidence based on order book signals
            confidence_adjustment = 0.0
            
            # Liquidity signal
            if signals.get('liquidity_signal') == 'positive':
                confidence_adjustment += 0.1
            elif signals.get('liquidity_signal') == 'negative':
                confidence_adjustment -= 0.1
            
            # Spread signal
            if signals.get('spread_signal') == 'positive':
                confidence_adjustment += 0.05
            elif signals.get('spread_signal') == 'negative':
                confidence_adjustment -= 0.05
            
            # Imbalance signal
            imbalance_signal = signals.get('imbalance_signal')
            if imbalance_signal == 'buy' and signal.action == 'buy':
                confidence_adjustment += 0.1
            elif imbalance_signal == 'sell' and signal.action == 'sell':
                confidence_adjustment += 0.1
            elif imbalance_signal and imbalance_signal != signal.action:
                confidence_adjustment -= 0.1
            
            # Pressure signal
            pressure_signal = signals.get('pressure_signal')
            if pressure_signal == signal.action:
                confidence_adjustment += 0.05
            elif pressure_signal and pressure_signal != signal.action:
                confidence_adjustment -= 0.05
            
            # Apply confidence adjustment
            new_confidence = max(0.1, min(0.95, signal.confidence + confidence_adjustment))
            signal.confidence = new_confidence
            
            # Adjust volume based on liquidity
            liquidity = current.get('total_liquidity', 0)
            if liquidity > 100:  # High liquidity
                signal.volume = min(signal.volume * 1.2, 0.05)  # Increase volume up to 0.05 lots
            elif liquidity < 20:  # Low liquidity
                signal.volume = max(signal.volume * 0.8, 0.005)  # Decrease volume, minimum 0.005 lots
            
            # Adjust stop loss based on spread
            spread = current.get('spread', 0)
            if spread > 0.0005 and signal.entry_price:  # Wide spread
                # Widen stop loss to account for spread
                if signal.action == 'buy':
                    signal.stop_loss = signal.entry_price * 0.985  # 1.5% stop loss
                else:
                    signal.stop_loss = signal.entry_price * 1.015  # 1.5% stop loss
            
            logger.debug(f"Signal adjusted for {signal.symbol}: confidence={new_confidence:.3f}, "
                        f"volume={signal.volume:.3f}, spread={spread:.5f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error adjusting signal with order book: {e}")
            return signal
    
    async def _execute_trading_signals(self, signals: List[TradingSignal]):
        """Execute trading signals."""
        for signal in signals:
            if await self._should_execute_signal(signal):
                await self._execute_signal(signal)
    
    async def _should_execute_signal(self, signal: TradingSignal) -> bool:
        """Check if signal should be executed based on advanced risk management."""
        # Check if we have too many positions
        current_positions = len(self.ctrader.get_positions())
        if current_positions >= self.max_positions:
            logger.info(f"Max positions reached ({current_positions}), skipping signal")
            return False
        
        # Convert TradingSignal to StrategySignal for advanced risk validation
        strategy_signal = StrategySignal(
            strategy_id=signal.symbol,  # Use symbol as strategy ID for now
            symbol=signal.symbol,
            action=signal.action,
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            volume=signal.volume,
            timestamp=signal.timestamp or datetime.now()
        )
        
        # Advanced risk validation
        is_valid, reason, risk_metadata = (True, "", {}) if not self.advanced_risk_manager else self.advanced_risk_manager.validate_signal(
            strategy_signal, self.market_data
        )
        
        if not is_valid:
            logger.info(f"Signal rejected by advanced risk manager: {signal.symbol} - {reason}")
            return False
        
        # Check confidence threshold
        if signal.confidence < 0.6:
            logger.info(f"Signal confidence too low ({signal.confidence:.2%}): {signal.symbol}")
            return False
        
        # Log risk metadata
        logger.debug(f"Risk metadata for {signal.symbol}: {risk_metadata}")
        
        return True
    
    async def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal."""
        try:
            # Determine order type and side
            order_type = OrderType.MARKET
            side = OrderSide.BUY if signal.action == 'buy' else OrderSide.SELL
            
            # Place order
            order_id = await self.ctrader.place_order(
                symbol_name=signal.symbol,
                order_type=order_type,
                side=side,
                volume=signal.volume,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if order_id:
                logger.info(f"Order placed: {signal.action} {signal.volume} {signal.symbol} (ID: {order_id})")
                self.signals.append(signal)
                
                # Record trade in performance tracker
                trade_data = {
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'entry_price': signal.entry_price,
                    'size': signal.volume,
                    'strategy': 'evolutionary',
                    'entry_time': datetime.now(),
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit
                }
                self.performance_tracker.record_trade(trade_data)
            else:
                logger.error(f"Failed to place order for {signal.symbol}")
                
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    async def _load_evolved_strategies(self):
        """Load evolved strategies from the genetic engine."""
        try:
            # Get best strategies from genetic engine
            best_strategies = self.genetic_engine.get_best_strategies(count=5)
            
            for strategy in best_strategies:
                # Convert TradingStrategy to DecisionGenome format
                genome = self._convert_strategy_to_genome(strategy)
                success = self.strategy_manager.add_strategy(genome)
                if success:
                    logger.info(f"Loaded evolved strategy: {strategy.id} (fitness: {strategy.fitness_score:.3f})")
                else:
                    logger.warning(f"Failed to load strategy: {strategy.id}")
            
            logger.info(f"Loaded {len(best_strategies)} evolved strategies")
            
        except Exception as e:
            logger.error(f"Error loading evolved strategies: {e}")
    
    def _convert_strategy_to_genome(self, strategy) -> DecisionGenome:
        """Convert TradingStrategy to DecisionGenome format."""
        # Create decision tree from strategy parameters
        decision_tree = {
            'parameters': strategy.parameters,
            'indicators': strategy.indicators,
            'entry_rules': strategy.entry_rules,
            'exit_rules': strategy.exit_rules,
            'risk_management': strategy.risk_management
        }
        
        # Create DecisionGenome
        genome = DecisionGenome(
            genome_id=strategy.id,
            decision_tree=decision_tree,
            fitness_score=strategy.fitness_score,
            generation=strategy.generation,
            parent_ids=strategy.parent_ids
        )
        
        return genome
    
    async def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data for analysis."""
        # This would integrate with your data source
        # For now, return None to indicate no historical data
        return None
    
    def _on_price_update(self, market_data: MarketData):
        """Handle price updates."""
        symbol = market_data.symbol_name
        self.market_data[symbol] = market_data
        
        # Update position P&L
        self._update_position_pnl(symbol, market_data)
    
    def _on_order_update(self, order: Order):
        """Handle order updates."""
        logger.info(f"Order update: {order.order_id} - {order.status}")
    
    def _on_position_update(self, position: Position):
        """Handle position updates."""
        logger.info(f"Position update: {position.position_id} - P&L: {position.profit_loss}")
        
        # Update performance tracker
        position_data = {
            'symbol': position.symbol_id,
            'volume': position.volume,
            'entry_price': position.entry_price,
            'current_price': position.current_price,
            'pnl': position.profit_loss,
            'side': position.side.value
        }
        self.performance_tracker.update_position(position_data)
    
    def _on_error(self, error_msg: str):
        """Handle errors."""
        logger.error(f"Trading error: {error_msg}")
    
    def _update_position_pnl(self, symbol: str, market_data: MarketData):
        """Update position P&L for a symbol."""
        positions = self.ctrader.get_positions()
        for position in positions:
            if self.ctrader._get_symbol_name(position.symbol_id) == symbol:
                # Update P&L calculation
                # This would need to be implemented based on position structure
                pass
    
    def _update_performance(self):
        """Update performance metrics using performance tracker."""
        try:
            # Update daily equity
            current_equity = self.performance_tracker.current_balance
            self.performance_tracker.update_daily_equity(current_equity)
            
            # Calculate comprehensive metrics
            metrics = self.performance_tracker.calculate_metrics()
            
            # Update legacy performance object for compatibility
            self.performance.total_trades = metrics.total_trades
            self.performance.winning_trades = metrics.winning_trades
            self.performance.losing_trades = metrics.losing_trades
            self.performance.win_rate = metrics.win_rate
            self.performance.avg_win = metrics.avg_win
            self.performance.avg_loss = metrics.avg_loss
            self.performance.max_drawdown = metrics.max_drawdown
            self.performance.sharpe_ratio = metrics.sharpe_ratio
            
            # Log performance alerts
            alerts = self.performance_tracker.get_performance_alerts()
            for alert in alerts:
                logger.warning(f"Performance Alert: {alert['message']}")
                
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def _update_risk_metrics(self):
        """Update advanced risk metrics."""
        try:
            positions = self.ctrader.get_positions()
            orders = self.ctrader.get_orders()
            
            # Get account state (mock values for now)
            equity = 10000.0  # This would come from account info
            margin = sum(abs(p.volume * p.entry_price) for p in positions)
            
            # Update portfolio state
            if self.advanced_risk_manager:
                self.advanced_risk_manager.update_portfolio_state(
                positions=positions,
                equity=equity,
                margin=margin,
                orders=orders
                )
            
            # Update risk metrics
            if self.advanced_risk_manager:
                self.advanced_risk_manager.update_risk_metrics(positions, self.market_data)
            
            # Log risk alerts
            risk_report = self.advanced_risk_manager.get_risk_report() if self.advanced_risk_manager else {"alerts": []}
            if risk_report['alerts']:
                for alert in risk_report['alerts']:
                    logger.warning(f"Risk Alert: {alert}")
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_trades': self.performance.total_trades,
            'winning_trades': self.performance.winning_trades,
            'losing_trades': self.performance.losing_trades,
            'win_rate': f"{self.performance.win_rate:.2%}",
            'net_profit': f"${self.performance.net_profit:.2f}",
            'total_profit': f"${self.performance.total_profit:.2f}",
            'total_loss': f"${self.performance.total_loss:.2f}",
            'avg_win': f"${self.performance.avg_win:.2f}",
            'avg_loss': f"${self.performance.avg_loss:.2f}",
            'max_drawdown': f"{self.performance.max_drawdown:.2%}",
            'sharpe_ratio': f"{self.performance.sharpe_ratio:.2f}"
        }
    
    def get_comprehensive_performance_report(self, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Get comprehensive performance report using performance tracker."""
        return self.performance_tracker.generate_report(report_type)
    
    def export_performance_data(self, format: str = "json") -> str:
        """Export performance data."""
        return self.performance_tracker.export_data(format)
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts."""
        return self.performance_tracker.get_performance_alerts()
    
    def get_order_book_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get order book analysis for a symbol."""
        return self.order_book_analyzer.get_market_analysis(symbol)
    
    def get_liquidity_analysis(self, symbol: str, volume: float) -> Dict[str, Any]:
        """Get liquidity analysis for a specific volume."""
        return self.order_book_analyzer.get_liquidity_analysis(symbol, volume)
    
    def get_order_book_snapshot(self, symbol: str):
        """Get the latest order book snapshot for a symbol."""
        return self.order_book_analyzer.get_order_book_snapshot(symbol)
    
    def export_order_book_data(self, symbol: str, format: str = "json") -> str:
        """Export order book data for analysis."""
        return self.order_book_analyzer.export_order_book_data(symbol, format)


class LiveRiskManager:
    """Risk management for live trading."""
    
    def __init__(self, max_risk_per_trade: float = 0.02):
        self.max_risk_per_trade = max_risk_per_trade
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.daily_loss = 0.0
        self.last_reset = datetime.now().date()
    
    def check_signal(self, signal: TradingSignal) -> bool:
        """Check if signal meets risk management criteria."""
        # Reset daily loss if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_loss = 0.0
            self.last_reset = current_date
        
        # Check daily loss limit
        if self.daily_loss >= self.daily_loss_limit:
            return False
        
        # Check position sizing
        if signal.volume > self.max_risk_per_trade:
            return False
        
        return True
    
    def update_daily_loss(self, loss: float):
        """Update daily loss tracking."""
        self.daily_loss += abs(loss)


async def main():
    """Test the live trading executor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test live trading executor")
    parser.add_argument("--config", required=True, help="Path to trading config file")
    parser.add_argument("--symbols", nargs="+", default=["EURUSD"], help="Symbols to trade")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    config = TradingConfig(**config_data)
    
    # Create executor
    executor = LiveTradingExecutor(config, args.symbols)
    
    # Start trading
    if await executor.start():
        print(f"Live trading executor started. Running for {args.duration} seconds...")
        
        # Run trading cycles
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < args.duration:
            await executor.run_trading_cycle()
            await asyncio.sleep(5)  # 5-second cycles
        
        # Stop and show performance
        await executor.stop()
        
        print("\nTrading Performance Summary:")
        performance = executor.get_performance_summary()
        for key, value in performance.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to start live trading executor")


if __name__ == "__main__":
    asyncio.run(main()) 
