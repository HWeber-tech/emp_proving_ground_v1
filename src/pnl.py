"""
PnL Engine for the EMP Proving Ground system.

This module provides precise profit and loss calculations:
- EnhancedPosition: Position tracking with v2.0 features
- TradeRecord: Immutable trade transaction records
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any

from .core import Instrument

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Immutable record of a trade transaction for audit trail"""
    timestamp: datetime
    trade_type: str  # 'OPEN', 'ADD', 'REDUCE', 'CLOSE', 'REVERSE', 'SWAP'
    quantity: int
    price: Decimal
    commission: Decimal
    slippage: Decimal
    swap_fee: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedPosition:
    """Enhanced position with v2.0 features for precise PnL tracking"""
    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_price: Decimal
    entry_timestamp: datetime
    last_swap_time: datetime
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    max_adverse_excursion: Decimal = Decimal('0')
    max_favorable_excursion: Decimal = Decimal('0')
    trade_history: List[TradeRecord] = field(default_factory=list)
    
    def update(self, trade_quantity: int, trade_price: Decimal, 
               commission: Decimal, slippage: Decimal, 
               current_time: datetime, trade_type: str = "UNKNOWN") -> None:
        """Update position with new trade"""
        
        # Create trade record
        trade_record = TradeRecord(
            timestamp=current_time,
            trade_type=trade_type,
            quantity=trade_quantity,
            price=trade_price,
            commission=commission,
            slippage=slippage
        )
        self.trade_history.append(trade_record)
        
        # Calculate new position
        old_quantity = self.quantity
        old_avg_price = self.avg_price
        
        if trade_type in ["OPEN", "ADD"]:
            # Opening or adding to position
            if old_quantity == 0:
                # Opening new position
                self.quantity = trade_quantity
                self.avg_price = trade_price
                self.entry_timestamp = current_time
            else:
                # Adding to existing position
                total_quantity = old_quantity + trade_quantity
                self.avg_price = ((old_quantity * old_avg_price) + (trade_quantity * trade_price)) / total_quantity
                self.quantity = total_quantity
                
        elif trade_type in ["REDUCE", "CLOSE"]:
            # Reducing or closing position
            if abs(trade_quantity) > abs(old_quantity):
                raise ValueError(f"Cannot close more than current position: {trade_quantity} vs {old_quantity}")
            
            # Calculate realized PnL
            if trade_type == "CLOSE":
                # Full close
                pnl = (trade_price - old_avg_price) * old_quantity
                if old_quantity < 0:  # Short position
                    pnl = -pnl
                self.realized_pnl += pnl
                self.quantity = 0
            else:
                # Partial close
                pnl = (trade_price - old_avg_price) * trade_quantity
                if old_quantity < 0:  # Short position
                    pnl = -pnl
                self.realized_pnl += pnl
                self.quantity = old_quantity - trade_quantity
                
        elif trade_type == "REVERSE":
            # Reverse position (close old and open new)
            # First close existing position
            if old_quantity != 0:
                pnl = (trade_price - old_avg_price) * old_quantity
                if old_quantity < 0:  # Short position
                    pnl = -pnl
                self.realized_pnl += pnl
            
            # Then open new position
            self.quantity = trade_quantity
            self.avg_price = trade_price
            self.entry_timestamp = current_time
    
    def update_unrealized_pnl(self, current_market_price: Decimal) -> None:
        """Update unrealized PnL and track MAE/MFE"""
        if self.quantity == 0:
            self.unrealized_pnl = Decimal('0')
            return
        
        # Calculate unrealized PnL
        pnl = (current_market_price - self.avg_price) * self.quantity
        if self.quantity < 0:  # Short position
            pnl = -pnl
        
        self.unrealized_pnl = pnl
        
        # Update MAE/MFE
        if pnl < self.max_adverse_excursion:
            self.max_adverse_excursion = pnl
        if pnl > self.max_favorable_excursion:
            self.max_favorable_excursion = pnl
    
    def apply_swap_fee(self, current_time: datetime, instrument: Instrument) -> None:
        """Apply swap fee if past swap time"""
        if self.quantity == 0:
            return
        
        # Parse swap time
        swap_hour, swap_minute = map(int, instrument.swap_time.split(':'))
        swap_time = current_time.replace(hour=swap_hour, minute=swap_minute, second=0, microsecond=0)
        
        # Check if we're past swap time and it's a new day
        if (current_time >= swap_time and 
            current_time.date() > self.last_swap_time.date()):
            
            # Apply appropriate swap rate
            if self.quantity > 0:  # Long position
                swap_fee = instrument.long_swap_rate * abs(self.quantity)
            else:  # Short position
                swap_fee = instrument.short_swap_rate * abs(self.quantity)
            
            # Add to trade history
            swap_record = TradeRecord(
                timestamp=current_time,
                trade_type="SWAP",
                quantity=0,
                price=Decimal('0'),
                commission=Decimal('0'),
                slippage=Decimal('0'),
                swap_fee=swap_fee
            )
            self.trade_history.append(swap_record)
            
            # Update realized PnL
            self.realized_pnl -= swap_fee
            
            # Update last swap time
            self.last_swap_time = current_time
    
    def get_total_pnl(self) -> Decimal:
        """Get total PnL (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    def get_total_commission(self) -> Decimal:
        """Get total commission paid"""
        total = sum(trade.commission for trade in self.trade_history)
        return total if total else Decimal('0')
    
    def get_total_slippage(self) -> Decimal:
        """Get total slippage incurred"""
        total = sum(trade.slippage for trade in self.trade_history)
        return total if total else Decimal('0')
    
    def get_total_swap_fees(self) -> Decimal:
        """Get total swap fees paid"""
        total = sum(trade.swap_fee for trade in self.trade_history)
        return total if total else Decimal('0')
    
    def get_trade_count(self) -> int:
        """Get number of trades in this position"""
        return len(self.trade_history)
    
    def get_position_value(self, current_price: Decimal) -> Decimal:
        """Get current position value"""
        return abs(self.quantity * current_price)
    
    def get_margin_requirement(self, leverage: Decimal = Decimal('1')) -> Decimal:
        """Get margin requirement for this position"""
        if leverage <= 0:
            raise ValueError("Leverage must be positive")
        return self.get_position_value(self.avg_price) / leverage
    
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0
    
    def is_flat(self) -> bool:
        """Check if position is flat (no position)"""
        return self.quantity == 0
    
    def get_duration(self, current_time: datetime) -> float:
        """Get position duration in days"""
        if self.is_flat():
            return 0.0
        return (current_time - self.entry_timestamp).total_seconds() / (24 * 3600) 