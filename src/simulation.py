"""
Market Simulation for the EMP Proving Ground system.

This module provides:
- MarketSimulator: Realistic market simulation
- AdversarialEngine: Adversarial market events
"""

import logging
import random
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class MarketState:
    """Current market state"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    spread_bps: float
    mid_price: float
    
    # Technical indicators (calculated on demand)
    atr: Optional[float] = None
    volatility: Optional[float] = None
    session: Optional[str] = None


class MarketSimulator:
    """Realistic market simulator with order execution"""
    
    def __init__(self, data_storage, initial_balance: float = 100000.0, leverage: float = 1.0):
        """
        Initialize market simulator
        
        Args:
            data_storage: Data storage instance
            initial_balance: Initial account balance
            leverage: Account leverage
        """
        self.data_storage = data_storage
        self.initial_balance = initial_balance
        self.leverage = leverage
        
        # Market data
        self.current_data = None
        self.current_index = 0
        self.symbol = None
        self.start_time = None
        self.end_time = None
        
        # Account state
        self.balance = initial_balance
        self.equity = initial_balance
        self.margin_used = 0.0
        self.free_margin = initial_balance
        
        # Positions and orders
        self.positions: Dict[str, Any] = {}
        self.orders: Dict[str, Order] = {}
        self.order_id_counter = 0
        
        # Performance tracking
        self.trade_history = []
        self.equity_curve = []
        
        # Adversarial callbacks
        self.adversarial_callbacks: List[Callable] = []
        
        logger.info(f"Initialized market simulator with balance: {initial_balance}")
    
    def load_data(self, symbol: str, start_time: datetime, end_time: datetime):
        """
        Load market data for simulation
        
        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time
        """
        try:
            self.symbol = symbol
            self.start_time = start_time
            self.end_time = end_time
            
            # Load tick data
            self.current_data = self.data_storage.load_tick_data(symbol, start_time, end_time)
            
            if self.current_data.empty:
                logger.error(f"No data available for {symbol}")
                return False
            
            # Prepare data
            self._prepare_data()
            self.current_index = 0
            
            logger.info(f"Loaded {len(self.current_data)} ticks for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _prepare_data(self):
        """Prepare market data for simulation"""
        if self.current_data is None or self.current_data.empty:
            return
        
        # Calculate additional fields
        self.current_data['mid_price'] = (self.current_data['bid'] + self.current_data['ask']) / 2
        self.current_data['spread_bps'] = ((self.current_data['ask'] - self.current_data['bid']) / self.current_data['bid']) * 10000
        
        # Calculate ATR
        self._calculate_atr()
        
        # Calculate sessions
        self._calculate_sessions()
        
        # Calculate volatility
        self.current_data['volatility'] = self.current_data['mid_price'].rolling(window=20).std()
    
    def _calculate_atr(self, period: int = 14):
        """Calculate Average True Range"""
        high_low = self.current_data['bid'].rolling(window=period).max() - self.current_data['bid'].rolling(window=period).min()
        high_close = np.abs(self.current_data['bid'].rolling(window=period).max() - self.current_data['bid'].shift(1))
        low_close = np.abs(self.current_data['bid'].rolling(window=period).min() - self.current_data['bid'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        self.current_data['atr'] = true_range.rolling(window=period).mean()
    
    def _calculate_sessions(self):
        """Calculate trading sessions"""
        def get_session(timestamp: datetime) -> str:
            hour = timestamp.hour
            if 0 <= hour < 8:
                return "asian"
            elif 8 <= hour < 16:
                return "london"
            elif 13 <= hour < 21:
                return "new_york"
            else:
                return "low_activity"
        
        self.current_data['session'] = self.current_data['timestamp'].apply(get_session)
    
    def step(self) -> Optional[MarketState]:
        """
        Advance simulation by one step
        
        Returns:
            Current market state or None if simulation ended
        """
        if (self.current_data is None or 
            self.current_index >= len(self.current_data) or 
            self.current_data.empty):
            return None
        
        # Get current tick
        tick = self.current_data.iloc[self.current_index]
        
        # Create market state
        market_state = MarketState(
            timestamp=tick['timestamp'],
            symbol=self.symbol,
            bid=tick['bid'],
            ask=tick['ask'],
            bid_volume=tick['bid_volume'],
            ask_volume=tick['ask_volume'],
            spread_bps=tick['spread_bps'],
            mid_price=tick['mid_price'],
            atr=tick.get('atr'),
            volatility=tick.get('volatility'),
            session=tick.get('session')
        )
        
        # Apply adversarial effects
        self._apply_adversarial_effects()
        
        # Process orders
        self._process_orders()
        
        # Update positions
        self._update_positions(market_state)
        
        # Update account
        self._update_account()
        
        # Record equity
        self.equity_curve.append({
            'timestamp': market_state.timestamp,
            'equity': self.equity,
            'balance': self.balance,
            'margin_used': self.margin_used
        })
        
        self.current_index += 1
        return market_state
    
    def _apply_adversarial_effects(self):
        """Apply adversarial effects to market"""
        for callback in self.adversarial_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Error in adversarial callback: {e}")
    
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                   quantity: float, price: Optional[float] = None) -> str:
        """
        Place a trading order
        
        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price (for limit/stop orders)
            
        Returns:
            Order ID
        """
        order_id = f"order_{self.order_id_counter}"
        self.order_id_counter += 1
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=datetime.now()
        )
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order {order_id} rejected")
        else:
            self.orders[order_id] = order
            logger.info(f"Placed order {order_id}: {side.value} {quantity} {symbol}")
        
        return order_id
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        # Check if we have enough margin
        required_margin = order.quantity * self.current_data.iloc[self.current_index]['bid'] / self.leverage
        if required_margin > self.free_margin:
            return False
        
        # Check position limits
        if order.symbol in self.positions:
            current_pos = self.positions[order.symbol]
            if order.side == OrderSide.BUY and current_pos.quantity < 0:
                # Closing short position
                pass
            elif order.side == OrderSide.SELL and current_pos.quantity > 0:
                # Closing long position
                pass
            else:
                # Check total exposure
                total_exposure = sum(abs(pos.quantity * pos.avg_price) for pos in self.positions.values())
                if total_exposure > self.equity * 10:  # 10x leverage limit
                    return False
        
        return True
    
    def _process_orders(self):
        """Process pending orders"""
        current_tick = self.current_data.iloc[self.current_index]
        
        for order_id, order in list(self.orders.items()):
            if order.status != OrderStatus.PENDING:
                continue
            
            if order.order_type == OrderType.MARKET:
                self._execute_order(order_id)
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_tick['bid'] <= order.price:
                    self._execute_order(order_id)
                elif order.side == OrderSide.SELL and current_tick['ask'] >= order.price:
                    self._execute_order(order_id)
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_tick['ask'] >= order.price:
                    self._execute_order(order_id)
                elif order.side == OrderSide.SELL and current_tick['bid'] <= order.price:
                    self._execute_order(order_id)
    
    def _execute_order(self, order_id: str):
        """Execute an order"""
        order = self.orders[order_id]
        current_tick = self.current_data.iloc[self.current_index]
        
        # Determine execution price
        if order.side == OrderSide.BUY:
            execution_price = current_tick['ask']
        else:
            execution_price = current_tick['bid']
        
        # Apply execution effects (slippage, etc.)
        execution_price = self._apply_execution_effects(order, execution_price)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        
        # Update position
        self._update_position(order)
        
        # Record trade
        self._record_trade(order)
        
        logger.info(f"Executed order {order_id} at {execution_price}")
    
    def _apply_execution_effects(self, order: Order, base_price: float) -> float:
        """Apply execution effects like slippage"""
        # Simple slippage model
        slippage_bps = random.uniform(0.1, 1.0)  # 0.1-1.0 bps slippage
        
        if order.side == OrderSide.BUY:
            return base_price * (1 + slippage_bps / 10000)
        else:
            return base_price * (1 - slippage_bps / 10000)
    
    def _update_position(self, order: Order):
        """Update position after order execution"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            # Create new position
            self.positions[symbol] = {
                'quantity': order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity,
                'avg_price': order.filled_price,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
        else:
            # Update existing position
            pos = self.positions[symbol]
            old_quantity = pos['quantity']
            old_avg_price = pos['avg_price']
            
            if order.side == OrderSide.BUY:
                new_quantity = old_quantity + order.filled_quantity
                if new_quantity == 0:
                    # Position closed
                    pnl = (order.filled_price - old_avg_price) * old_quantity
                    pos['realized_pnl'] += pnl
                    pos['quantity'] = 0
                else:
                    # Position updated
                    pos['avg_price'] = ((old_quantity * old_avg_price) + (order.filled_quantity * order.filled_price)) / new_quantity
                    pos['quantity'] = new_quantity
            else:
                new_quantity = old_quantity - order.filled_quantity
                if new_quantity == 0:
                    # Position closed
                    pnl = (old_avg_price - order.filled_price) * old_quantity
                    pos['realized_pnl'] += pnl
                    pos['quantity'] = 0
                else:
                    # Position updated
                    pos['quantity'] = new_quantity
    
    def _record_trade(self, order: Order):
        """Record trade in history"""
        trade = {
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.filled_quantity,
            'price': order.filled_price,
            'order_id': order.order_id
        }
        self.trade_history.append(trade)
    
    def _update_positions(self, market_state: MarketState):
        """Update unrealized PnL for all positions"""
        for symbol, position in self.positions.items():
            if position['quantity'] != 0:
                if position['quantity'] > 0:  # Long position
                    pnl = (market_state.mid_price - position['avg_price']) * position['quantity']
                else:  # Short position
                    pnl = (position['avg_price'] - market_state.mid_price) * abs(position['quantity'])
                
                position['unrealized_pnl'] = pnl
    
    def _update_account(self):
        """Update account state"""
        # Calculate total PnL
        total_pnl = sum(pos['realized_pnl'] + pos['unrealized_pnl'] for pos in self.positions.values())
        
        # Update equity
        self.equity = self.balance + total_pnl
        
        # Calculate margin used
        self.margin_used = sum(abs(pos['quantity'] * pos['avg_price']) / self.leverage for pos in self.positions.values())
        
        # Update free margin
        self.free_margin = self.equity - self.margin_used
    
    def add_adversarial_callback(self, callback: Callable):
        """Add adversarial callback function"""
        self.adversarial_callbacks.append(callback)
    
    def get_account_summary(self) -> Dict:
        """Get account summary"""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'margin_used': self.margin_used,
            'free_margin': self.free_margin,
            'positions': len(self.positions),
            'orders': len(self.orders)
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.equity_curve:
            return {}
        
        equity_values = [point['equity'] for point in self.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        return {
            'total_return': (equity_values[-1] - equity_values[0]) / equity_values[0],
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(equity_values),
            'total_trades': len(self.trade_history),
            'win_rate': self._calculate_win_rate()
        }
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        if not self.trade_history:
            return 0.0
        
        # Simple win rate calculation
        # In a real implementation, you'd track individual trade PnL
        return 0.5  # Placeholder


class AdversarialEventType(Enum):
    SPOOFING = "spoofing"
    STOP_HUNT = "stop_hunt"
    NEWS_SHOCK = "news_shock"
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_CRUNCH = "liquidity_crunch"


@dataclass
class AdversarialEvent:
    """Represents an adversarial market event"""
    event_type: AdversarialEventType
    timestamp: datetime
    duration: timedelta
    intensity: float  # 0.0 to 1.0
    parameters: Dict[str, Any] = None
    active: bool = False


class AdversarialEngine:
    """Generates and manages adversarial market events"""
    
    def __init__(self, difficulty_level: float = 0.5, seed: Optional[int] = None):
        """
        Initialize adversarial engine
        
        Args:
            difficulty_level: Difficulty level (0.0 to 1.0)
            seed: Random seed
        """
        self.difficulty_level = max(0.0, min(1.0, difficulty_level))
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Event management
        self.active_events: List[AdversarialEvent] = []
        self.event_history: List[AdversarialEvent] = []
        
        # Configuration
        self.config = self._create_config(self.difficulty_level)
        
        logger.info(f"Initialized adversarial engine with difficulty: {difficulty_level}")
    
    def _create_config(self, difficulty: float) -> Dict:
        """Create configuration based on difficulty level"""
        return {
            'event_frequency': 0.01 + (difficulty * 0.09),  # 1-10% chance per tick
            'max_concurrent_events': int(1 + difficulty * 3),  # 1-4 concurrent events
            'event_duration_min': timedelta(minutes=5),
            'event_duration_max': timedelta(hours=2),
            'intensity_range': (0.3 + difficulty * 0.4, 0.7 + difficulty * 0.3)
        }
    
    def update(self, market_state: MarketState, simulator: MarketSimulator):
        """
        Update adversarial engine
        
        Args:
            market_state: Current market state
            simulator: Market simulator instance
        """
        # Update active events
        self._update_active_events(market_state.timestamp)
        
        # Check for new events
        if (len(self.active_events) < self.config['max_concurrent_events'] and
            random.random() < self.config['event_frequency']):
            self._trigger_random_event(market_state, simulator)
        
        # Apply active effects
        self._apply_active_effects(market_state, simulator)
    
    def _update_active_events(self, current_time: datetime):
        """Update active events"""
        for event in self.active_events[:]:
            if current_time >= event.timestamp + event.duration:
                event.active = False
                self.active_events.remove(event)
                self.event_history.append(event)
    
    def _trigger_random_event(self, market_state: MarketState, simulator: MarketSimulator):
        """Trigger a random adversarial event"""
        event_type = random.choice(list(AdversarialEventType))
        
        duration = random.uniform(
            self.config['event_duration_min'].total_seconds(),
            self.config['event_duration_max'].total_seconds()
        )
        
        intensity = random.uniform(*self.config['intensity_range'])
        
        event = AdversarialEvent(
            event_type=event_type,
            timestamp=market_state.timestamp,
            duration=timedelta(seconds=duration),
            intensity=intensity,
            parameters={},
            active=True
        )
        
        self.active_events.append(event)
        logger.info(f"Triggered {event_type.value} event with intensity {intensity:.2f}")
    
    def _apply_active_effects(self, market_state: MarketState, simulator: MarketSimulator):
        """Apply effects from active events"""
        for event in self.active_events:
            if event.event_type == AdversarialEventType.SPOOFING:
                self._apply_spoofing_effects(market_state, event)
            elif event.event_type == AdversarialEventType.STOP_HUNT:
                self._apply_stop_hunt_effects(market_state, event)
            elif event.event_type == AdversarialEventType.NEWS_SHOCK:
                self._apply_news_shock_effects(market_state, event)
            elif event.event_type == AdversarialEventType.FLASH_CRASH:
                self._apply_flash_crash_effects(market_state, event)
    
    def _apply_spoofing_effects(self, market_state: MarketState, event: AdversarialEvent):
        """Apply spoofing effects"""
        # Increase spread and reduce liquidity
        spread_multiplier = 1.0 + (event.intensity * 2.0)
        market_state.spread_bps *= spread_multiplier
        market_state.bid_volume *= (1.0 - event.intensity * 0.5)
        market_state.ask_volume *= (1.0 - event.intensity * 0.5)
    
    def _apply_stop_hunt_effects(self, market_state: MarketState, event: AdversarialEvent):
        """Apply stop hunt effects"""
        # Create false breakouts
        direction = random.choice([-1, 1])
        price_move = direction * event.intensity * market_state.atr * 0.5 if market_state.atr else 0.001
        
        market_state.bid += price_move
        market_state.ask += price_move
        market_state.mid_price += price_move
    
    def _apply_news_shock_effects(self, market_state: MarketState, event: AdversarialEvent):
        """Apply news shock effects"""
        # Sudden price movement
        direction = random.choice([-1, 1])
        price_move = direction * event.intensity * 0.01  # 1% move
        
        market_state.bid += price_move
        market_state.ask += price_move
        market_state.mid_price += price_move
    
    def _apply_flash_crash_effects(self, market_state: MarketState, event: AdversarialEvent):
        """Apply flash crash effects"""
        # Extreme price movement
        direction = random.choice([-1, 1])
        price_move = direction * event.intensity * 0.05  # 5% move
        
        market_state.bid += price_move
        market_state.ask += price_move
        market_state.mid_price += price_move
        
        # Increase volatility
        if market_state.volatility:
            market_state.volatility *= (1.0 + event.intensity * 2.0)
    
    def get_active_events(self) -> List[AdversarialEvent]:
        """Get currently active events"""
        return self.active_events.copy()
    
    def get_event_history(self) -> List[AdversarialEvent]:
        """Get event history"""
        return self.event_history.copy() 