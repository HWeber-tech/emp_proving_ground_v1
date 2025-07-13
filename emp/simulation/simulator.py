"""
MarketSimulator: Realistic market simulation with correct PnL engine.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float  # Positive for long, negative for short
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

@dataclass
class MarketState:
    """Current market state"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    spread: float
    mid_price: float
    atr: Optional[float] = None
    volatility: Optional[float] = None

@dataclass
class Trade:
    """Represents a completed trade"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    spread_cost: float

class PositionManager:
    """
    Manages trading positions and calculates PnL.
    """
    
    def __init__(self, initial_balance: float = 100000.0, leverage: float = 1.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.trade_counter = 0
        
    def place_order(self, order: Order, market_state: MarketState) -> Optional[Trade]:
        """
        Place and execute an order.
        
        Args:
            order: Order to execute
            market_state: Current market state
            
        Returns:
            Trade object if order was filled, None otherwise
        """
        # Calculate fill price with realistic costs
        fill_price, commission, slippage, spread_cost = self._calculate_fill_price(
            order, market_state
        )
        
        if fill_price is None:
            order.status = OrderStatus.REJECTED
            return None
        
        # Update order
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.slippage = slippage
        order.status = OrderStatus.FILLED
        order.timestamp = market_state.timestamp
        
        # Create trade
        trade = Trade(
            trade_id=f"trade_{self.trade_counter}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=market_state.timestamp,
            commission=commission,
            slippage=slippage,
            spread_cost=spread_cost
        )
        
        self.trade_counter += 1
        self.trades.append(trade)
        
        # Update position
        self._update_position(order, trade)
        
        # Update balance
        self._update_balance(trade)
        
        return trade
    
    def _calculate_fill_price(self, order: Order, market_state: MarketState) -> tuple:
        """
        Calculate realistic fill price with all costs.
        
        Args:
            order: Order to execute
            market_state: Current market state
            
        Returns:
            Tuple of (fill_price, commission, slippage, spread_cost)
        """
        # Base price (bid for sells, ask for buys)
        if order.side == OrderSide.BUY:
            base_price = market_state.ask
        else:
            base_price = market_state.bid
        
        # Calculate spread cost
        spread_cost = market_state.spread * order.quantity
        
        # Calculate slippage (function of volatility and order size)
        slippage_bps = self._calculate_slippage_bps(order, market_state)
        slippage = (slippage_bps / 10000) * base_price * order.quantity
        
        # Calculate commission (fixed per trade)
        commission = 0.0001 * order.quantity  # 1 pip commission
        
        # Calculate final fill price
        if order.side == OrderSide.BUY:
            fill_price = base_price + (slippage_bps / 10000) * base_price
        else:
            fill_price = base_price - (slippage_bps / 10000) * base_price
        
        return fill_price, commission, slippage, spread_cost
    
    def _calculate_slippage_bps(self, order: Order, market_state: MarketState) -> float:
        """
        Calculate slippage in basis points.
        
        Args:
            order: Order to execute
            market_state: Current market state
            
        Returns:
            Slippage in basis points
        """
        # Base slippage
        base_slippage_bps = 0.5
        
        # Volatility factor
        volatility_factor = 0.0
        if market_state.volatility is not None:
            volatility_factor = market_state.volatility * 0.2
        
        # Size impact factor
        size_impact_factor = 0.1 * np.log(max(order.quantity, 1.0))
        
        # ATR factor
        atr_factor = 0.0
        if market_state.atr is not None:
            atr_factor = (market_state.atr / market_state.mid_price) * 10000 * 0.2
        
        total_slippage_bps = base_slippage_bps + volatility_factor + size_impact_factor + atr_factor
        
        return max(0.0, total_slippage_bps)
    
    def _update_position(self, order: Order, trade: Trade):
        """
        Update position after trade execution.
        
        Args:
            order: Executed order
            trade: Completed trade
        """
        symbol = order.symbol
        
        if symbol not in self.positions:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                avg_price=0.0
            )
        
        position = self.positions[symbol]
        
        if order.side == OrderSide.BUY:
            # Buying - increase long position
            if position.quantity >= 0:
                # Adding to long position
                total_cost = position.quantity * position.avg_price + trade.quantity * trade.price
                position.quantity += trade.quantity
                position.avg_price = total_cost / position.quantity
            else:
                # Covering short position
                if abs(position.quantity) <= trade.quantity:
                    # Complete cover
                    realized_pnl = (position.avg_price - trade.price) * abs(position.quantity)
                    position.realized_pnl += realized_pnl
                    position.quantity += trade.quantity
                    if position.quantity > 0:
                        position.avg_price = trade.price
                else:
                    # Partial cover
                    realized_pnl = (position.avg_price - trade.price) * trade.quantity
                    position.realized_pnl += realized_pnl
                    position.quantity += trade.quantity
        else:
            # Selling - increase short position
            if position.quantity <= 0:
                # Adding to short position
                total_cost = abs(position.quantity) * position.avg_price + trade.quantity * trade.price
                position.quantity -= trade.quantity
                position.avg_price = total_cost / abs(position.quantity)
            else:
                # Reducing long position
                if position.quantity <= trade.quantity:
                    # Complete exit
                    realized_pnl = (trade.price - position.avg_price) * position.quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity
                    if position.quantity < 0:
                        position.avg_price = trade.price
                else:
                    # Partial exit
                    realized_pnl = (trade.price - position.avg_price) * trade.quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity
        
        # Update costs
        position.total_commission += trade.commission
        position.total_slippage += trade.slippage
        
        # Remove position if quantity is zero
        if abs(position.quantity) < 1e-8:
            del self.positions[symbol]
    
    def _update_balance(self, trade: Trade):
        """
        Update account balance after trade.
        
        Args:
            trade: Completed trade
        """
        # Calculate trade cost
        trade_cost = trade.commission + trade.slippage + trade.spread_cost
        
        # Update balance
        self.balance -= trade_cost
    
    def update_unrealized_pnl(self, market_state: MarketState):
        """
        Update unrealized PnL for all positions.
        
        Args:
            market_state: Current market state
        """
        for position in self.positions.values():
            if position.symbol == market_state.symbol:
                if position.quantity > 0:
                    # Long position
                    position.unrealized_pnl = (market_state.mid_price - position.avg_price) * position.quantity
                else:
                    # Short position
                    position.unrealized_pnl = (position.avg_price - market_state.mid_price) * abs(position.quantity)
    
    def get_total_pnl(self) -> float:
        """
        Get total PnL (realized + unrealized).
        
        Returns:
            Total PnL
        """
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        return total_realized + total_unrealized
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary.
        
        Returns:
            Account summary dictionary
        """
        total_pnl = self.get_total_pnl()
        total_commission = sum(pos.total_commission for pos in self.positions.values())
        total_slippage = sum(pos.total_slippage for pos in self.positions.values())
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_trades': len(self.trades),
            'open_positions': len(self.positions),
            'return_pct': (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0.0
        }

class MarketSimulator:
    """
    Realistic market simulator with correct PnL engine.
    """
    
    def __init__(self, data_storage, initial_balance: float = 100000.0, leverage: float = 1.0):
        self.data_storage = data_storage
        self.position_manager = PositionManager(initial_balance, leverage)
        self.current_data: Optional[pd.DataFrame] = None
        self.current_index = 0
        self.symbol = None
        self.start_time = None
        self.end_time = None
        
        # Simulation state
        self.is_running = False
        self.current_market_state: Optional[MarketState] = None
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        
    def load_data(self, symbol: str, start_time: datetime, end_time: datetime):
        """
        Load tick data for simulation.
        
        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time
        """
        logger.info(f"Loading data for {symbol} from {start_time} to {end_time}")
        
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        
        # Load tick data
        self.current_data = self.data_storage.load_tick_data(symbol, start_time, end_time)
        
        if self.current_data.empty:
            raise ValueError(f"No data found for {symbol} in specified time range")
        
        # Sort by timestamp
        self.current_data = self.current_data.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate additional indicators
        self._calculate_indicators()
        
        self.current_index = 0
        logger.info(f"Loaded {len(self.current_data)} ticks for simulation")
    
    def _calculate_indicators(self):
        """Calculate technical indicators for the data."""
        if self.current_data is None or self.current_data.empty:
            return
        
        # Calculate ATR (Average True Range)
        self.current_data['atr'] = self._calculate_atr(self.current_data)
        
        # Calculate volatility (rolling standard deviation of returns)
        self.current_data['returns'] = self.current_data['mid_price'].pct_change()
        self.current_data['volatility'] = (
            self.current_data['returns'].rolling(window=100).std() * np.sqrt(252 * 24 * 60)  # Annualized
        )
        
        # Fill NaN values
        self.current_data = self.current_data.fillna(method='bfill')
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['ask']
        low = df['bid']
        close = df['mid_price']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def step(self) -> Optional[MarketState]:
        """
        Process one tick and return current market state.
        
        Returns:
            MarketState if data available, None if simulation complete
        """
        if self.current_data is None or self.current_index >= len(self.current_data):
            return None
        
        # Get current tick
        tick = self.current_data.iloc[self.current_index]
        
        # Create market state
        self.current_market_state = MarketState(
            timestamp=tick['timestamp'],
            symbol=self.symbol,
            bid=tick['bid'],
            ask=tick['ask'],
            bid_volume=tick['bid_volume'],
            ask_volume=tick['ask_volume'],
            spread=tick['spread'],
            mid_price=tick['mid_price'],
            atr=tick.get('atr'),
            volatility=tick.get('volatility')
        )
        
        # Update unrealized PnL
        self.position_manager.update_unrealized_pnl(self.current_market_state)
        
        # Record performance
        self._record_performance()
        
        # Move to next tick
        self.current_index += 1
        
        return self.current_market_state
    
    def _record_performance(self):
        """Record current performance metrics."""
        if self.current_market_state is None:
            return
        
        account_summary = self.position_manager.get_account_summary()
        
        performance = {
            'timestamp': self.current_market_state.timestamp,
            'balance': account_summary['current_balance'],
            'total_pnl': account_summary['total_pnl'],
            'return_pct': account_summary['return_pct'],
            'total_trades': account_summary['total_trades'],
            'open_positions': account_summary['open_positions']
        }
        
        self.performance_history.append(performance)
    
    def place_order(self, side: OrderSide, order_type: OrderType, 
                   quantity: float, price: Optional[float] = None) -> Optional[Trade]:
        """
        Place a trading order.
        
        Args:
            side: Order side (buy/sell)
            order_type: Order type (market/limit/stop)
            quantity: Order quantity
            price: Order price (for limit/stop orders)
            
        Returns:
            Trade object if order was filled, None otherwise
        """
        if self.current_market_state is None:
            logger.warning("Cannot place order: no current market state")
            return None
        
        # Create order
        order_id = f"order_{len(self.position_manager.trades)}_{self.current_index}"
        order = Order(
            order_id=order_id,
            symbol=self.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )
        
        # Execute order
        trade = self.position_manager.place_order(order, self.current_market_state)
        
        if trade:
            logger.info(f"Order executed: {side.value} {quantity} {self.symbol} at {trade.price:.5f}")
        else:
            logger.warning(f"Order rejected: {side.value} {quantity} {self.symbol}")
        
        return trade
    
    def run_simple_strategy(self) -> Dict[str, Any]:
        """
        Run a simple hard-coded strategy for testing.
        
        Returns:
            Strategy results
        """
        logger.info("Running simple strategy test")
        
        # Simple moving average crossover strategy
        window_short = 20
        window_long = 50
        
        # Calculate moving averages
        if self.current_data is not None:
            self.current_data['ma_short'] = self.current_data['mid_price'].rolling(window=window_short).mean()
            self.current_data['ma_long'] = self.current_data['mid_price'].rolling(window=window_long).mean()
        
        position = 0  # 0: no position, 1: long, -1: short
        trades = []
        
        # Run simulation
        while True:
            market_state = self.step()
            if market_state is None:
                break
            
            # Get current moving averages
            if self.current_index > window_long:
                ma_short = self.current_data.iloc[self.current_index - 1]['ma_short']
                ma_long = self.current_data.iloc[self.current_index - 1]['ma_long']
                
                # Trading logic
                if position == 0:  # No position
                    if ma_short > ma_long:  # Golden cross
                        trade = self.place_order(OrderSide.BUY, OrderType.MARKET, 10000)
                        if trade:
                            position = 1
                            trades.append(trade)
                    elif ma_short < ma_long:  # Death cross
                        trade = self.place_order(OrderSide.SELL, OrderType.MARKET, 10000)
                        if trade:
                            position = -1
                            trades.append(trade)
                
                elif position == 1:  # Long position
                    if ma_short < ma_long:  # Death cross
                        trade = self.place_order(OrderSide.SELL, OrderType.MARKET, 10000)
                        if trade:
                            position = 0
                            trades.append(trade)
                
                elif position == -1:  # Short position
                    if ma_short > ma_long:  # Golden cross
                        trade = self.place_order(OrderSide.BUY, OrderType.MARKET, 10000)
                        if trade:
                            position = 0
                            trades.append(trade)
        
        # Get final results
        account_summary = self.position_manager.get_account_summary()
        
        results = {
            'strategy_name': 'Simple MA Crossover',
            'total_trades': len(trades),
            'final_balance': account_summary['current_balance'],
            'total_pnl': account_summary['total_pnl'],
            'return_pct': account_summary['return_pct'],
            'total_commission': account_summary['total_commission'],
            'total_slippage': account_summary['total_slippage'],
            'trades': trades,
            'performance_history': self.performance_history
        }
        
        logger.info(f"Strategy completed: {account_summary['total_trades']} trades, "
                   f"PnL: {account_summary['total_pnl']:.2f}, "
                   f"Return: {account_summary['return_pct']:.2f}%")
        
        return results
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get simulation summary.
        
        Returns:
            Simulation summary
        """
        account_summary = self.position_manager.get_account_summary()
        
        return {
            'symbol': self.symbol,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_ticks': len(self.current_data) if self.current_data is not None else 0,
            'processed_ticks': self.current_index,
            'account_summary': account_summary,
            'open_positions': list(self.position_manager.positions.keys())
        } 