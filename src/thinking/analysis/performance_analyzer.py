"""
EMP Performance Analyzer v1.1

Performance analysis and scoring for the thinking layer.
Migrated from evolution layer to thinking layer where cognitive functions belong.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...core.events import AnalysisResult, PerformanceMetrics, TradeIntent

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Performance analyzer for cognitive assessment of trading strategies."""
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.analysis_history: List[dict[str, object]] = []
        
        logger.info(f"Performance Analyzer initialized with {risk_free_rate:.2%} risk-free rate")
        
    def analyze_performance(self, trade_history: List[TradeIntent], 
                          initial_capital: float = 100000.0) -> AnalysisResult:
        """Analyze trading performance and generate metrics."""
        try:
            if not trade_history:
                logger.warning("Empty trade history provided for performance analysis")
                return self._create_default_analysis()
                
            # Convert trade history to performance data
            performance_data = self._convert_trades_to_performance(trade_history, initial_capital)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(performance_data)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_analysis_confidence(performance_data)
            
            # Create analysis result
            result = AnalysisResult(
                timestamp=datetime.now(),
                analysis_type="performance_analysis",
                result={
                    "performance_metrics": metrics.__dict__,
                    "trade_count": len(trade_history),
                    "analysis_period": self._calculate_analysis_period(trade_history),
                    "initial_capital": initial_capital,
                    "final_capital": performance_data['equity_curve'].iloc[-1] if len(performance_data['equity_curve']) > 0 else initial_capital
                },
                confidence=confidence,
                metadata={
                    "analyzer_version": "1.1.0",
                    "method": "comprehensive_performance_analysis",
                    "risk_free_rate": self.risk_free_rate,
                    "trading_days_per_year": self.trading_days_per_year
                }
            )
            
            # Store in history
            self.analysis_history.append({
                "timestamp": result.timestamp,
                "trade_count": len(trade_history),
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "confidence": confidence
            })
            
            logger.debug(f"Performance analyzed: {metrics.total_return:.2%} return, {metrics.sharpe_ratio:.2f} Sharpe")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return self._create_default_analysis()
            
    def analyze_backtest_results(self, backtest_results: dict[str, object]) -> AnalysisResult:
        """Analyze backtest results from genome evaluation."""
        try:
            # Extract data from backtest results
            trades = backtest_results.get("trades", [])
            equity_curve = backtest_results.get("equity_curve", [])
            
            if not equity_curve:
                return self._create_default_analysis()
                
            # Calculate returns from equity curve
            returns = []
            for i in range(1, len(equity_curve)):
                if equity_curve[i-1] > 0:
                    returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
                    
            # Create performance metrics
            metrics = self._calculate_metrics_from_returns(returns, equity_curve, trades)
            
            # Calculate confidence
            confidence = self._calculate_backtest_confidence(backtest_results)
            
            # Create analysis result
            result = AnalysisResult(
                timestamp=datetime.now(),
                analysis_type="backtest_performance_analysis",
                result={
                    "performance_metrics": metrics.__dict__,
                    "trade_count": len(trades),
                    "equity_curve_length": len(equity_curve),
                    "initial_capital": equity_curve[0] if equity_curve else 0,
                    "final_capital": equity_curve[-1] if equity_curve else 0
                },
                confidence=confidence,
                metadata={
                    "analyzer_version": "1.1.0",
                    "method": "backtest_performance_analysis",
                    "source": "genome_evaluation"
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing backtest results: {e}")
            return self._create_default_analysis()
            
    def _calculate_metrics_from_returns(self, returns: List[float], 
                                      equity_curve: List[float], 
                                      trades: List[dict[str, object]]) -> PerformanceMetrics:
        """Calculate performance metrics from returns, equity curve, and trades."""
        if not returns:
            return PerformanceMetrics()
            
        returns_array = np.array(returns)
        
        # Basic metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if equity_curve[0] > 0 else 0
        volatility = returns_array.std() * np.sqrt(self.trading_days_per_year)
        
        # Sharpe ratio
        if volatility > 0:
            excess_return = (returns_array.mean() * self.trading_days_per_year) - self.risk_free_rate
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0
            
        # Sortino ratio
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(self.trading_days_per_year)
            if downside_deviation > 0:
                sortino_ratio = ((returns_array.mean() * self.trading_days_per_year) - self.risk_free_rate) / downside_deviation
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0
            
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Win rate and profit factor from trades
        if trades:
            winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
            losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
            
            win_rate = len(winning_trades) / len(trades)
            
            total_wins = sum(t.get("pnl", 0) for t in winning_trades)
            total_losses = abs(sum(t.get("pnl", 0) for t in losing_trades))
            
            if total_losses > 0:
                profit_factor = min(total_wins / total_losses, 10.0)
            else:
                profit_factor = total_wins / 1e-6 if total_wins > 0 else 0.0
        else:
            win_rate = 0
            profit_factor = 0
            
        # Value at Risk
        var_95 = -np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        
        # Recovery time
        recovery_time = self._calculate_recovery_time(equity_curve)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=total_return * (self.trading_days_per_year / len(returns)) if returns else 0,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_duration=recovery_time,
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points": len(returns)
            }
        )
        
    def _convert_trades_to_performance(self, trade_history: List[TradeIntent], 
                                     initial_capital: float) -> dict[str, object]:
        """Convert trade history to performance data structure."""
        # Sort trades by timestamp
        sorted_trades = sorted(trade_history, key=lambda x: x.timestamp)
        
        # Initialize performance tracking
        current_capital = initial_capital
        equity_curve = [initial_capital]
        timestamps = [sorted_trades[0].timestamp if sorted_trades else datetime.now()]
        returns = []
        trade_returns = []
        
        for trade in sorted_trades:
            if trade.action == "HOLD":
                continue
                
            # Calculate trade P&L (simplified)
            if trade.price:
                if trade.action == "BUY":
                    # Assume we're buying and will sell later
                    trade_pnl = 0  # Will be calculated on sell
                elif trade.action == "SELL":
                    # Calculate P&L from previous buy
                    trade_pnl = trade.quantity * (trade.price - 0)  # Simplified
                    current_capital += trade_pnl
                    trade_returns.append(trade_pnl / current_capital)
                    
                equity_curve.append(current_capital)
                timestamps.append(trade.timestamp)
                
                if len(equity_curve) > 1:
                    period_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                    returns.append(period_return)
                    
        return {
            'equity_curve': pd.Series(equity_curve, index=timestamps),
            'returns': pd.Series(returns),
            'trade_returns': trade_returns,
            'initial_capital': initial_capital,
            'final_capital': current_capital
        }
        
    def _calculate_performance_metrics(self, performance_data: dict[str, object]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        equity_curve = performance_data['equity_curve']
        returns = performance_data['returns']
        trade_returns = performance_data['trade_returns']
        initial_capital = performance_data['initial_capital']
        final_capital = performance_data['final_capital']
        
        # Basic return metrics
        total_return = (final_capital - initial_capital) / initial_capital if initial_capital > 0 else 0
        
        # Annualized return
        if len(equity_curve) > 1:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if days > 0:
                annualized_return = ((final_capital / initial_capital) ** (self.trading_days_per_year / days)) - 1
            else:
                annualized_return = total_return
        else:
            annualized_return = total_return
            
        # Volatility
        volatility = returns.std() * np.sqrt(self.trading_days_per_year) if len(returns) > 1 else 0
        
        # Sharpe ratio
        if volatility > 0:
            excess_return = annualized_return - self.risk_free_rate
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0
            
        # Sortino ratio
        if len(returns) > 1:
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(self.trading_days_per_year) if len(negative_returns) > 1 else 0
            if downside_deviation > 0:
                sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0
            
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Win rate and profit factor
        if trade_returns:
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]
            
            win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
            
            total_profit = sum(winning_trades) if winning_trades else 0
            total_loss = abs(sum(losing_trades)) if losing_trades else 0
            
            profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0)
        else:
            win_rate = 0
            profit_factor = 0
            
        # Trade statistics
        total_trades = len(trade_returns)
        avg_trade_duration = self._calculate_avg_trade_duration(performance_data)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=avg_trade_duration,
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points": len(equity_curve)
            }
        )
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(equity_curve) < 2:
            return 0.0
            
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Return maximum drawdown
        return abs(drawdown.min())
        
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
        
    def _calculate_avg_trade_duration(self, performance_data: dict[str, object]) -> float:
        """Calculate average trade duration in days."""
        # This is a simplified calculation
        # In a real implementation, you'd track entry and exit times for each trade
        if len(performance_data['equity_curve']) < 2:
            return 0.0
            
        total_days = (performance_data['equity_curve'].index[-1] - performance_data['equity_curve'].index[0]).days
        trade_count = len(performance_data['trade_returns'])
        
        if trade_count > 0:
            return total_days / trade_count
        else:
            return 0.0
            
    def _calculate_analysis_confidence(self, performance_data: dict[str, object]) -> float:
        """Calculate confidence in performance analysis."""
        confidence_factors = []
        
        # Data sufficiency
        data_points = len(performance_data['equity_curve'])
        if data_points >= 100:
            confidence_factors.append(1.0)
        elif data_points >= 50:
            confidence_factors.append(0.8)
        elif data_points >= 20:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
            
        # Trade frequency
        trade_count = len(performance_data['trade_returns'])
        if trade_count >= 50:
            confidence_factors.append(1.0)
        elif trade_count >= 20:
            confidence_factors.append(0.8)
        elif trade_count >= 10:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
            
        # Time span
        if len(performance_data['equity_curve']) > 1:
            time_span = (performance_data['equity_curve'].index[-1] - performance_data['equity_curve'].index[0]).days
            if time_span >= 252:  # 1 year
                confidence_factors.append(1.0)
            elif time_span >= 126:  # 6 months
                confidence_factors.append(0.8)
            elif time_span >= 63:   # 3 months
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.1)
            
        return np.mean(confidence_factors)
        
    def _calculate_backtest_confidence(self, backtest_results: dict[str, object]) -> float:
        """Calculate confidence in backtest analysis."""
        confidence_factors = []
        
        # Data sufficiency
        equity_curve_length = len(backtest_results.get("equity_curve", []))
        if equity_curve_length >= 100:
            confidence_factors.append(1.0)
        elif equity_curve_length >= 50:
            confidence_factors.append(0.8)
        elif equity_curve_length >= 20:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
            
        # Trade count
        trade_count = len(backtest_results.get("trades", []))
        if trade_count >= 50:
            confidence_factors.append(1.0)
        elif trade_count >= 20:
            confidence_factors.append(0.8)
        elif trade_count >= 10:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
            
        return np.mean(confidence_factors) if confidence_factors else 0.5
        
    def _calculate_analysis_period(self, trade_history: List[TradeIntent]) -> dict[str, object]:
        """Calculate the analysis period from trade history."""
        if not trade_history:
            return {"start": None, "end": None, "duration_days": 0}
            
        start_time = min(trade.timestamp for trade in trade_history)
        end_time = max(trade.timestamp for trade in trade_history)
        duration_days = (end_time - start_time).days
        
        return {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_days": duration_days
        }
        
    def _create_default_analysis(self) -> AnalysisResult:
        """Create default analysis when performance analysis fails."""
        return AnalysisResult(
            timestamp=datetime.now(),
            analysis_type="performance_analysis",
            result={
                "performance_metrics": PerformanceMetrics().__dict__,
                "trade_count": 0,
                "analysis_period": {"start": None, "end": None, "duration_days": 0},
                "initial_capital": 0,
                "final_capital": 0
            },
            confidence=0.1,
            metadata={
                "analyzer_version": "1.1.0",
                "method": "default_fallback",
                "error": "Insufficient data for analysis"
            }
        )
        
    def get_performance_history(self, limit: Optional[int] = None) -> List[dict[str, object]]:
        """Get performance analysis history."""
        if limit:
            return self.analysis_history[-limit:]
        return self.analysis_history.copy()
        
    def get_performance_statistics(self) -> dict[str, object]:
        """Get statistics about performance analyses."""
        if not self.analysis_history:
            return {}
            
        returns = [h['total_return'] for h in self.analysis_history]
        sharpe_ratios = [h['sharpe_ratio'] for h in self.analysis_history]
        drawdowns = [h['max_drawdown'] for h in self.analysis_history]
        confidences = [h['confidence'] for h in self.analysis_history]
        
        return {
            'total_analyses': len(self.analysis_history),
            'average_return': np.mean(returns),
            'return_std': np.std(returns),
            'average_sharpe': np.mean(sharpe_ratios),
            'average_drawdown': np.mean(drawdowns),
            'average_confidence': np.mean(confidences),
            'best_return': max(returns) if returns else 0,
            'worst_return': min(returns) if returns else 0
        } 
