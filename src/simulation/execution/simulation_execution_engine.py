"""
Simulation Execution Engine v1.0 - Virtual Broker for Backtesting

Implements SIM-03 ticket requirements for virtual portfolio management.
Provides realistic trade execution simulation with slippage, fees, and portfolio tracking.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from ...core.events import TradeIntent, ExecutionReport, FitnessReport
from ...core.event_bus import EventBus
from ...domain.models import Portfolio, Position, Trade

logger = logging.getLogger(__name__)


class SimulationExecutionEngine:
    """
    Virtual broker that simulates trade execution for backtesting purposes.
    Provides realistic execution with slippage, fees, and portfolio tracking.
    """
    
    def __init__(self, event_bus: EventBus, initial_capital: Decimal = Decimal('100000')):
        """
        Initialize the simulation execution engine.
        
        Args:
            event_bus: The event bus for receiving trade intents and publishing reports
            initial_capital: Starting capital for the virtual portfolio
        """
        self.event_bus = event_bus
        self.portfolio = Portfolio(
            cash=initial_capital,
            positions={},
            total_value=initial_capital,
            unrealized_pnl=Decimal('0'),
            realized_pnl=Decimal('0')
        )
        self.trade_history: List[Trade] = []
        self.is_running = False
        
        # Configuration parameters
        self.slippage_rate = Decimal('0.0001')  # 0.01% slippage
        self.commission_rate = Decimal('0.0005')  # 0.05% commission
        self.minimum_commission = Decimal('1.00')
        
    async def start(self) -> None:
        """Start the simulation execution engine."""
        self.is_running = True
        logger.info("Simulation execution engine started")
        
    async def stop(self) -> None:
        """Stop the simulation execution engine."""
        self.is_running = False
        logger.info("Simulation execution engine stopped")
        
    async def process_trade_intent(self, trade_intent: TradeIntent) -> ExecutionReport:
        """
        Process a trade intent and generate an execution report.
        
        Args:
            trade_intent: The trade intent to process
            
        Returns:
            ExecutionReport: The execution report for the trade
        """
        if not self.is_running:
            return self._create_rejection_report(trade_intent, "Engine not running")
            
        try:
            # Validate trade intent
            validation_result = self._validate_trade_intent(trade_intent)
            if not validation_result['valid']:
                return self._create_rejection_report(trade_intent, validation_result['reason'])
                
            # Calculate execution price with slippage
            execution_price = self._calculate_execution_price(
                trade_intent.price or Decimal('0'),
                trade_intent.action
            )
            
            # Calculate fees
            fees = self._calculate_fees(trade_intent.quantity, execution_price)
            
            # Update portfolio
            trade = self._execute_trade(trade_intent, execution_price, fees)
            
            # Create execution report
            execution_report = ExecutionReport(
                event_id=f"exec_{datetime.now().isoformat()}",
                timestamp=datetime.now(),
                source="SimulationExecutionEngine",
                trade_intent_id=trade_intent.event_id,
                symbol=trade_intent.symbol,
                action=trade_intent.action,
                quantity=trade_intent.quantity,
                price=execution_price,
                fees=fees,
                status="FILLED",
                metadata={
                    'portfolio_value': self.portfolio.total_value,
                    'cash_balance': self.portfolio.cash,
                    'realized_pnl': self.portfolio.realized_pnl
                }
            )
            
            # Publish execution report
            await self.event_bus.publish(execution_report)
            
            # Check if simulation should end (e.g., portfolio depleted)
            if self.portfolio.total_value <= Decimal('0'):
                await self._publish_fitness_report()
                
            return execution_report
            
        except Exception as e:
            logger.error(f"Error processing trade intent: {e}")
            return self._create_rejection_report(trade_intent, str(e))
            
    def _validate_trade_intent(self, trade_intent: TradeIntent) -> Dict[str, Any]:
        """Validate a trade intent."""
        if trade_intent.action not in ["BUY", "SELL"]:
            return {'valid': False, 'reason': f"Invalid action: {trade_intent.action}"}
            
        if trade_intent.quantity <= Decimal('0'):
            return {'valid': False, 'reason': "Quantity must be positive"}
            
        if trade_intent.action == "BUY":
            required_cash = trade_intent.quantity * (trade_intent.price or Decimal('0'))
            if required_cash > self.portfolio.cash:
                return {'valid': False, 'reason': "Insufficient cash"}
                
        elif trade_intent.action == "SELL":
            position = self.portfolio.positions.get(trade_intent.symbol)
            if not position or position.quantity < trade_intent.quantity:
                return {'valid': False, 'reason': "Insufficient position"}
                
        return {'valid': True, 'reason': ""}
        
    def _calculate_execution_price(self, intended_price: Decimal, action: str) -> Decimal:
        """Calculate execution price with slippage."""
        if intended_price == Decimal('0'):
            intended_price = Decimal('100')  # Default price if not provided
            
        slippage = intended_price * self.slippage_rate
        
        if action == "BUY":
            return intended_price + slippage
        else:  # SELL
            return intended_price - slippage
            
    def _calculate_fees(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate trading fees."""
        fees = quantity * price * self.commission_rate
        return max(fees, self.minimum_commission)
        
    def _execute_trade(self, trade_intent: TradeIntent, execution_price: Decimal, fees: Decimal) -> Trade:
        """Execute the trade and update portfolio."""
        trade = Trade(
            id=f"trade_{datetime.now().isoformat()}",
            symbol=trade_intent.symbol,
            action=trade_intent.action,
            quantity=trade_intent.quantity,
            price=execution_price,
            fees=fees,
            timestamp=datetime.now()
        )
        
        self.trade_history.append(trade)
        
        # Update portfolio
        trade_value = trade_intent.quantity * execution_price
        
        if trade_intent.action == "BUY":
            self.portfolio.cash -= trade_value + fees
            if trade_intent.symbol in self.portfolio.positions:
                self.portfolio.positions[trade_intent.symbol].quantity += trade_intent.quantity
            else:
                self.portfolio.positions[trade_intent.symbol] = Position(
                    symbol=trade_intent.symbol,
                    quantity=trade_intent.quantity,
                    average_price=execution_price
                )
                
        elif trade_intent.action == "SELL":
            self.portfolio.cash += trade_value - fees
            if trade_intent.symbol in self.portfolio.positions:
                self.portfolio.positions[trade_intent.symbol].quantity -= trade_intent.quantity
                if self.portfolio.positions[trade_intent.symbol].quantity <= Decimal('0'):
                    del self.portfolio.positions[trade_intent.symbol]
                    
        # Update portfolio value
        self._update_portfolio_value()
        
        return trade
        
    def _update_portfolio_value(self) -> None:
        """Update the total portfolio value."""
        # In simulation, we'll use the last trade price for valuation
        total_position_value = Decimal('0')
        
        for symbol, position in self.portfolio.positions.items():
            # For simulation, use the last trade price
            last_trade_price = Decimal('100')  # Placeholder
            total_position_value += position.quantity * last_trade_price
            
        self.portfolio.total_value = self.portfolio.cash + total_position_value
        
    def _create_rejection_report(self, trade_intent: TradeIntent, reason: str) -> ExecutionReport:
        """Create a rejection execution report."""
        return ExecutionReport(
            event_id=f"exec_reject_{datetime.now().isoformat()}",
            timestamp=datetime.now(),
            source="SimulationExecutionEngine",
            trade_intent_id=trade_intent.event_id,
            symbol=trade_intent.symbol,
            action=trade_intent.action,
            quantity=trade_intent.quantity,
            price=Decimal('0'),
            fees=Decimal('0'),
            status="REJECTED",
            metadata={'reason': reason}
        )
        
    async def _publish_fitness_report(self) -> None:
        """Publish a fitness report when simulation ends."""
        fitness_report = FitnessReport(
            event_id=f"fitness_{datetime.now().isoformat()}",
            timestamp=datetime.now(),
            source="SimulationExecutionEngine",
            genome_id="simulation_genome",
            fitness_score=self.portfolio.total_value,
            metrics={
                'final_value': self.portfolio.total_value,
                'realized_pnl': self.portfolio.realized_pnl,
                'total_trades': len(self.trade_history)
            }
        )
        
        await self.event_bus.publish(fitness_report)
        
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of the current portfolio."""
        return {
            'cash': float(self.portfolio.cash),
            'total_value': float(self.portfolio.total_value),
            'realized_pnl': float(self.portfolio.realized_pnl),
            'positions': {
                symbol: {
                    'quantity': float(pos.quantity),
                    'average_price': float(pos.average_price)
                }
                for symbol, pos in self.portfolio.positions.items()
            },
            'total_trades': len(self.trade_history)
        }
        
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get the trade history as dictionaries."""
        return [
            {
                'id': trade.id,
                'symbol': trade.symbol,
                'action': trade.action,
                'quantity': float(trade.quantity),
                'price': float(trade.price),
                'fees': float(trade.fees),
                'timestamp': trade.timestamp.isoformat()
            }
            for trade in self.trade_history
        ]
