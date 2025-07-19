"""
Performance Analyzer

Specialized performance analysis for backtesting results.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    avg_trade_duration: float
    avg_holding_period: float


class PerformanceAnalyzer:
    """
    Advanced Performance Analysis System
    
    Implements comprehensive performance analysis with:
    - Risk-adjusted returns
    - Drawdown analysis
    - Trade statistics
    - Performance attribution
    """
    
    def __init__(self):
        logger.info("PerformanceAnalyzer initialized")
    
    def analyze_performance(self, equity_curve: List[float], 
                          trade_history: List[Dict[str, Any]],
                          risk_free_rate: float = 0.02) -> PerformanceMetrics:
        """Analyze comprehensive performance metrics"""
        
        if not equity_curve or len(equity_curve) < 2:
            return self._create_default_metrics()
        
        # Calculate basic returns
        returns = self._calculate_returns(equity_curve)
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = self._calculate_sortino_ratio(returns, risk_free_rate)
        
        # Calculate drawdown metrics
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
        
        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics(trade_history)
        
        # Calculate holding period metrics
        holding_metrics = self._calculate_holding_period_metrics(trade_history)
        
        return PerformanceMetrics(
            total_return=self._calculate_total_return(equity_curve),
            annualized_return=self._calculate_annualized_return(returns),
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            average_win=trade_stats['average_win'],
            average_loss=trade_stats['average_loss'],
            largest_win=trade_stats['largest_win'],
            largest_loss=trade_stats['largest_loss'],
            consecutive_wins=trade_stats['consecutive_wins'],
            consecutive_losses=trade_stats['consecutive_losses'],
            avg_trade_duration=holding_metrics['avg_trade_duration'],
            avg_holding_period=holding_metrics['avg_holding_period']
        )
    
    def _calculate_returns(self, equity_curve: List[float]) -> List[float]:
        """Calculate daily returns from equity curve"""
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                daily_return = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(daily_return)
            else:
                returns.append(0.0)
        return returns
    
    def _calculate_total_return(self, equity_curve: List[float]) -> float:
        """Calculate total return"""
        if len(equity_curve) < 2:
            return 0.0
        
        initial_value = equity_curve[0]
        final_value = equity_curve[-1]
        
        if initial_value == 0:
            return 0.0
        
        return (final_value - initial_value) / initial_value
    
    def _calculate_annualized_return(self, returns: List[float]) -> float:
        """Calculate annualized return"""
        if not returns:
            return 0.0
        
        total_return = (1 + sum(returns)) - 1
        num_periods = len(returns)
        
        if num_periods == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        annualized_return = (1 + total_return) ** (252 / num_periods) - 1
        
        return annualized_return
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        sharpe_ratio = (mean_return - risk_free_rate/252) / std_return * np.sqrt(252)
        
        return sharpe_ratio
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        # Annualize
        sortino_ratio = (mean_return - risk_free_rate/252) / downside_deviation * np.sqrt(252)
        
        return sortino_ratio
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if not returns or max_drawdown == 0:
            return 0.0
        
        annualized_return = self._calculate_annualized_return(returns)
        calmar_ratio = annualized_return / max_drawdown
        
        return calmar_ratio
    
    def _calculate_trade_statistics(self, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade statistics"""
        if not trade_history:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            }
        
        # Calculate P&L for each trade
        trade_pnls = []
        for trade in trade_history:
            # Simplified P&L calculation
            if 'pnl' in trade:
                trade_pnls.append(trade['pnl'])
            else:
                # Estimate P&L from trade data
                pnl = 0.0  # Placeholder
                trade_pnls.append(pnl)
        
        # Calculate statistics
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        total_trades = len(trade_pnls)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        win_rate = winning_count / total_trades if total_trades > 0 else 0.0
        
        total_profit = sum(winning_trades) if winning_trades else 0.0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_win = np.mean(winning_trades) if winning_trades else 0.0
        average_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        largest_win = max(winning_trades) if winning_trades else 0.0
        largest_loss = min(losing_trades) if losing_trades else 0.0
        
        # Calculate consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_wins(trade_pnls)
        consecutive_losses = self._calculate_consecutive_losses(trade_pnls)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses
        }
    
    def _calculate_consecutive_wins(self, trade_pnls: List[float]) -> int:
        """Calculate maximum consecutive wins"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in trade_pnls:
            if pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, trade_pnls: List[float]) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in trade_pnls:
            if pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_holding_period_metrics(self, trade_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate holding period metrics"""
        if not trade_history:
            return {
                'avg_trade_duration': 0.0,
                'avg_holding_period': 0.0
            }
        
        # Simplified calculation - in practice would use actual timestamps
        avg_trade_duration = 1.0  # Placeholder
        avg_holding_period = 1.0  # Placeholder
        
        return {
            'avg_trade_duration': avg_trade_duration,
            'avg_holding_period': avg_holding_period
        }
    
    def _create_default_metrics(self) -> PerformanceMetrics:
        """Create default performance metrics"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            avg_trade_duration=0.0,
            avg_holding_period=0.0
        ) 