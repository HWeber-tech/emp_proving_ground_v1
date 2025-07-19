"""
EMP Performance Analyzer v1.1

Comprehensive performance analysis module that processes trading data
to calculate performance metrics, returns analysis, and trade statistics.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from src.core.interfaces import SensorySignal, AnalysisResult
from src.core.exceptions import ThinkingException

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_time: float
    metadata: Dict[str, Any]


class PerformanceAnalyzer:
    """Analyzes performance from trading data and sensory signals."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.lookback_periods = self.config.get('lookback_periods', 252)
        self._performance_history: List[Dict[str, Any]] = []
        
    def analyze(self, trading_data: Dict[str, Any], 
                sensory_signals: Optional[List[SensorySignal]] = None) -> AnalysisResult:
        """Analyze performance from trading data."""
        try:
            # Extract performance data
            equity_curve = trading_data.get('equity_curve', [])
            trade_history = trading_data.get('trade_history', [])
            
            if not equity_curve:
                return self._create_neutral_result()
                
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                equity_curve, trade_history
            )
            
            # Create analysis result
            return AnalysisResult(
                timestamp=datetime.now(),
                analysis_type="performance_analysis",
                result={
                    'total_return': performance_metrics.total_return,
                    'annualized_return': performance_metrics.annualized_return,
                    'volatility': performance_metrics.volatility,
                    'sharpe_ratio': performance_metrics.sharpe_ratio,
                    'sortino_ratio': performance_metrics.sortino_ratio,
                    'calmar_ratio': performance_metrics.calmar_ratio,
                    'max_drawdown': performance_metrics.max_drawdown,
                    'win_rate': performance_metrics.win_rate,
                    'profit_factor': performance_metrics.profit_factor,
                    'average_win': performance_metrics.average_win,
                    'average_loss': performance_metrics.average_loss,
                    'consecutive_wins': performance_metrics.consecutive_wins,
                    'consecutive_losses': performance_metrics.consecutive_losses,
                    'recovery_time': performance_metrics.recovery_time,
                    'metadata': performance_metrics.metadata
                },
                confidence=self._calculate_confidence(trading_data),
                metadata={
                    'trade_count': len(trade_history),
                    'data_points': len(equity_curve),
                    'analysis_method': 'comprehensive_performance'
                }
            )
            
        except Exception as e:
            raise ThinkingException(f"Error in performance analysis: {e}")
            
    def _calculate_performance_metrics(self, equity_curve: List[float], 
                                     trade_history: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not equity_curve:
            return self._create_neutral_performance_metrics()
            
        # Calculate returns
        returns = self._calculate_returns(equity_curve)
        
        if not returns:
            return self._create_neutral_performance_metrics()
            
        returns_array = np.array(returns)
        
        # Basic return metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        annualized_return = self._calculate_annualized_return(returns_array)
        
        # Risk metrics
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
        
        # Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(returns_array)
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trade_history)
        
        # Recovery time
        recovery_time = self._calculate_recovery_time(equity_curve)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            average_win=trade_stats['average_win'],
            average_loss=trade_stats['average_loss'],
            consecutive_wins=trade_stats['consecutive_wins'],
            consecutive_losses=trade_stats['consecutive_losses'],
            recovery_time=recovery_time,
            metadata={
                'return_count': len(returns),
                'trade_count': len(trade_history),
                'analysis_periods': self.lookback_periods
            }
        )
        
    def _calculate_returns(self, equity_curve: List[float]) -> List[float]:
        """Calculate returns from equity curve."""
        if len(equity_curve) < 2:
            return []
            
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] != 0:
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(ret)
                
        return returns
        
    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
            
        mean_return = np.mean(returns)
        return mean_return * 252  # Annualize
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
            
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
            
        # Annualize
        annualized_return = mean_return * 252
        annualized_std = std_return * np.sqrt(252)
        
        return (annualized_return - self.risk_free_rate) / annualized_std
        
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
            
        mean_return = np.mean(returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
            
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
            
        # Annualize
        annualized_return = mean_return * 252
        annualized_downside = downside_deviation * np.sqrt(252)
        
        return (annualized_return - self.risk_free_rate) / annualized_downside
        
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0.0
            
        peak = equity_curve[0]
        max_dd = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    def _calculate_trade_statistics(self, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade statistics."""
        if not trade_history:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            }
            
        # Separate winning and losing trades
        winning_trades = [t for t in trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trade_history if t.get('pnl', 0) < 0]
        
        # Calculate basic statistics
        win_rate = len(winning_trades) / len(trade_history) if trade_history else 0.0
        
        total_wins = sum(t.get('pnl', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
        
        profit_factor = total_wins / total_losses if total_losses > 0 else total_wins
        
        average_win = total_wins / len(winning_trades) if winning_trades else 0.0
        average_loss = total_losses / len(losing_trades) if losing_trades else 0.0
        
        # Calculate consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_wins(trade_history)
        consecutive_losses = self._calculate_consecutive_losses(trade_history)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses
        }
        
    def _calculate_consecutive_wins(self, trade_history: List[Dict[str, Any]]) -> int:
        """Calculate maximum consecutive wins."""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trade_history:
            if trade.get('pnl', 0) > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
        
    def _calculate_consecutive_losses(self, trade_history: List[Dict[str, Any]]) -> int:
        """Calculate maximum consecutive losses."""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trade_history:
            if trade.get('pnl', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
        
    def _calculate_recovery_time(self, equity_curve: List[float]) -> float:
        """Calculate average recovery time from drawdowns."""
        if len(equity_curve) < 2:
            return 0.0
            
        recovery_times = []
        peak = equity_curve[0]
        underwater_start = None
        
        for i, equity in enumerate(equity_curve):
            if equity > peak:
                if underwater_start is not None:
                    recovery_times.append(i - underwater_start)
                    underwater_start = None
                peak = equity
            elif underwater_start is None and equity < peak:
                underwater_start = i
                
        return float(np.mean(recovery_times)) if recovery_times else 0.0
        
    def _calculate_confidence(self, trading_data: Dict[str, Any]) -> float:
        """Calculate confidence in performance analysis."""
        equity_curve = trading_data.get('equity_curve', [])
        trade_history = trading_data.get('trade_history', [])
        
        # Confidence based on data quality
        data_confidence = min(len(equity_curve) / 100, 1.0)  # More data = higher confidence
        trade_confidence = min(len(trade_history) / 50, 1.0)  # More trades = higher confidence
        
        return (data_confidence + trade_confidence) / 2
        
    def _create_neutral_performance_metrics(self) -> PerformanceMetrics:
        """Create neutral performance metrics when no data is available."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            recovery_time=0.0,
            metadata={
                'return_count': 0,
                'trade_count': 0,
                'analysis_periods': 0
            }
        )
        
    def _create_neutral_result(self) -> AnalysisResult:
        """Create a neutral analysis result when no data is available."""
        return AnalysisResult(
            timestamp=datetime.now(),
            analysis_type="performance_analysis",
            result={
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'recovery_time': 0.0,
                'metadata': {}
            },
            confidence=0.0,
            metadata={
                'trade_count': 0,
                'data_points': 0,
                'analysis_method': 'neutral_fallback'
            }
        ) 