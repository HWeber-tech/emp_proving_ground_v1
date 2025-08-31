"""
Advanced Performance Tracking System
Tracks real-time performance metrics, strategy analysis, and generates detailed reports
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Performance metric types"""

    RETURNS = "returns"
    RISK = "risk"
    TRADING = "trading"
    STRATEGY = "strategy"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""

    # Returns metrics
    total_return: float
    annualized_return: float
    daily_returns: List[float]

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float

    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float

    # Strategy metrics
    strategy_performance: Dict[str, dict[str, object]]
    regime_performance: Dict[str, dict[str, object]]
    correlation_matrix: pd.DataFrame

    # Timestamps
    start_date: datetime
    end_date: datetime
    last_updated: datetime


class PerformanceTracker:
    """
    Advanced performance tracking system with real-time metrics
    """

    # Class-level annotations for attribute types (mypy-friendly)
    positions_history: List[dict[str, object]]
    trades_history: List[dict[str, object]]
    daily_equity: List[dict[str, object]]
    strategy_performance: Dict[str, dict[str, object]]
    regime_performance: Dict[str, dict[str, object]]
    metrics: Optional[PerformanceMetrics]
    last_calculation: Optional[datetime]

    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions_history = []
        self.trades_history = []
        self.daily_equity = []
        self.strategy_performance = {}
        self.regime_performance = {}

        # Performance tracking
        self.metrics = None
        self.last_calculation = None

        logger.info(f"Performance tracker initialized with balance: ${initial_balance:,.2f}")

    def update_position(self, position_data: dict[str, object]) -> None:
        """Update position tracking"""
        position_data["timestamp"] = datetime.now()
        self.positions_history.append(position_data)

        # Update current balance
        if "pnl" in position_data:
            pnl = float(cast(float, position_data.get("pnl", 0.0)))
            self.current_balance += pnl

        logger.debug(f"Position updated: {position_data}")

    def record_trade(self, trade_data: dict[str, object]) -> None:
        """Record a completed trade"""
        trade_data["timestamp"] = datetime.now()
        trade_data["trade_id"] = len(self.trades_history) + 1

        # Calculate trade metrics
        if "entry_price" in trade_data and "exit_price" in trade_data:
            ep = float(cast(float, trade_data["entry_price"]))
            xp = float(cast(float, trade_data["exit_price"]))
            sz = float(cast(float, trade_data["size"]))
            pnl = (xp - ep) * sz
            trade_data["pnl"] = pnl
            trade_data["return"] = pnl / (ep * sz)

        self.trades_history.append(trade_data)

        # Update strategy performance
        strategy = str(trade_data.get("strategy", "unknown"))
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "total_return": 0.0,
            }

        self.strategy_performance[strategy]["trades"] = (
            int(cast(int, self.strategy_performance[strategy]["trades"])) + 1
        )
        self.strategy_performance[strategy]["total_pnl"] = float(
            cast(float, self.strategy_performance[strategy]["total_pnl"])
        ) + float(cast(float, trade_data.get("pnl", 0.0)))

        if float(cast(float, trade_data.get("pnl", 0.0))) > 0.0:
            self.strategy_performance[strategy]["wins"] = (
                int(cast(int, self.strategy_performance[strategy]["wins"])) + 1
            )
        else:
            self.strategy_performance[strategy]["losses"] = (
                int(cast(int, self.strategy_performance[strategy]["losses"])) + 1
            )

        logger.info(f"Trade recorded: {trade_data}")

    def update_daily_equity(self, equity: float, date: Optional[datetime] = None) -> None:
        """Update daily equity tracking"""
        if date is None:
            date = datetime.now()

        self.daily_equity.append({"date": date, "equity": equity, "balance": self.current_balance})

        logger.debug(f"Daily equity updated: ${equity:,.2f} on {date.date()}")

    def update_regime_performance(self, regime: str, performance: float) -> None:
        """Update regime-specific performance"""
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {"trades": 0, "total_return": 0.0, "avg_return": 0.0}

        self.regime_performance[regime]["trades"] = (
            int(cast(int, self.regime_performance[regime]["trades"])) + 1
        )
        self.regime_performance[regime]["total_return"] = float(
            cast(float, self.regime_performance[regime]["total_return"])
        ) + float(performance)
        self.regime_performance[regime]["avg_return"] = float(
            cast(float, self.regime_performance[regime]["total_return"])
        ) / float(cast(int, self.regime_performance[regime]["trades"]))

        logger.debug(f"Regime performance updated: {regime} = {performance:.4f}")

    def calculate_metrics(self, force_recalculate: bool = False) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if (
            self.metrics is not None
            and self.last_calculation is not None
            and datetime.now() - self.last_calculation < timedelta(minutes=5)
            and not force_recalculate
        ):
            return self.metrics

        if not self.daily_equity:
            logger.warning("No equity data available for metrics calculation")
            return self._create_empty_metrics()

        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.daily_equity)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Calculate returns
        df["daily_return"] = df["equity"].pct_change()
        daily_returns = df["daily_return"].dropna().tolist()

        total_return = (df["equity"].iloc[-1] - self.initial_balance) / self.initial_balance
        annualized_return = self._calculate_annualized_return(df)

        # Risk metrics
        volatility = df["daily_return"].std() * np.sqrt(252)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        max_drawdown = self._calculate_max_drawdown(df["equity"])
        var_95, cvar_95 = self._calculate_var_cvar(daily_returns)

        # Trading metrics
        trading_metrics = self._calculate_trading_metrics()

        # Extract typed locals for metrics
        tm_total_trades: int = int(cast(int, trading_metrics["total_trades"]))
        tm_winning_trades: int = int(cast(int, trading_metrics["winning_trades"]))
        tm_losing_trades: int = int(cast(int, trading_metrics["losing_trades"]))
        tm_win_rate: float = float(cast(float, trading_metrics["win_rate"]))
        tm_avg_win: float = float(cast(float, trading_metrics["avg_win"]))
        tm_avg_loss: float = float(cast(float, trading_metrics["avg_loss"]))
        tm_profit_factor: float = float(cast(float, trading_metrics["profit_factor"]))
        tm_avg_trade_duration: float = float(cast(float, trading_metrics["avg_trade_duration"]))

        # Strategy performance
        strategy_perf = self._calculate_strategy_performance()

        # Regime performance
        regime_perf = self._calculate_regime_performance()

        # Correlation matrix
        correlation_matrix = self._calculate_correlation_matrix()

        self.metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns=daily_returns,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=tm_total_trades,
            winning_trades=tm_winning_trades,
            losing_trades=tm_losing_trades,
            win_rate=tm_win_rate,
            avg_win=tm_avg_win,
            avg_loss=tm_avg_loss,
            profit_factor=tm_profit_factor,
            avg_trade_duration=tm_avg_trade_duration,
            strategy_performance=strategy_perf,
            regime_performance=regime_perf,
            correlation_matrix=correlation_matrix,
            start_date=df["date"].min(),
            end_date=df["date"].max(),
            last_updated=datetime.now(),
        )

        self.last_calculation = datetime.now()

        logger.info(
            f"Performance metrics calculated: {total_return:.2%} total return, {sharpe_ratio:.2f} Sharpe"
        )
        return self.metrics

    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics when no data is available"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            daily_returns=[],
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            cvar_95=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            avg_trade_duration=0.0,
            strategy_performance={},
            regime_performance={},
            correlation_matrix=pd.DataFrame(),
            start_date=datetime.now(),
            end_date=datetime.now(),
            last_updated=datetime.now(),
        )

    def _calculate_annualized_return(self, df: pd.DataFrame) -> float:
        """Calculate annualized return"""
        if len(df) < 2:
            return 0.0

        total_days = (df["date"].max() - df["date"].min()).days
        if total_days == 0:
            return 0.0

        total_return = (df["equity"].iloc[-1] - self.initial_balance) / self.initial_balance
        return float((1 + total_return) ** (365 / total_days) - 1)

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        if returns_array.std() == 0:
            return 0.0

        return float(returns_array.mean() / returns_array.std() * np.sqrt(252))

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        negative_returns = returns_array[returns_array < 0]

        if len(negative_returns) == 0:
            return float("inf") if returns_array.mean() > 0 else 0.0

        downside_deviation = negative_returns.std()
        if downside_deviation == 0:
            return 0.0

        return float(returns_array.mean() / downside_deviation * np.sqrt(252))

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(equity) < 2:
            return 0.0

        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return float(abs(drawdown.min()))

    def _calculate_var_cvar(
        self, returns: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        if not returns:
            return 0.0, 0.0

        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence) * 100)
        cvar = returns_array[returns_array <= var].mean()

        return float(abs(var)), float(abs(cvar))

    def _calculate_trading_metrics(self) -> dict[str, object]:
        """Calculate trading-specific metrics"""
        if not self.trades_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "avg_trade_duration": 0.0,
            }

        trades_df = pd.DataFrame(self.trades_history)

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        losing_trades = len(trades_df[trades_df["pnl"] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0.0
        avg_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].mean()) if losing_trades > 0 else 0.0

        total_wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        total_losses = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # Calculate average trade duration
        if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
            trades_df["duration"] = pd.to_datetime(trades_df["exit_time"]) - pd.to_datetime(
                trades_df["entry_time"]
            )
            duration_mean = cast(pd.Timedelta, trades_df["duration"].mean())
            avg_trade_duration = float(duration_mean.total_seconds() / 3600)  # hours
        else:
            avg_trade_duration = 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_trade_duration": avg_trade_duration,
        }

    def _calculate_strategy_performance(self) -> Dict[str, dict[str, object]]:
        """Calculate strategy-specific performance metrics"""
        strategy_perf: Dict[str, dict[str, object]] = {}

        for strategy, data in self.strategy_performance.items():
            trades_n = int(cast(int, data.get("trades", 0)))
            if trades_n > 0:
                wins_n = float(cast(float, data.get("wins", 0.0)))
                total_ret = float(cast(float, data.get("total_return", 0.0)))
                win_rate = wins_n / trades_n if trades_n > 0 else 0.0
                avg_return = total_ret / trades_n if trades_n > 0 else 0.0

                strategy_perf[strategy] = {
                    "win_rate": win_rate,
                    "avg_return": avg_return,
                    "total_pnl": float(cast(float, data.get("total_pnl", 0.0))),
                    "trade_count": trades_n,
                }

        return strategy_perf

    def _calculate_regime_performance(self) -> Dict[str, dict[str, object]]:
        """Calculate regime-specific performance metrics"""
        regime_perf: Dict[str, dict[str, object]] = {}

        for regime, data in self.regime_performance.items():
            trades_n = int(cast(int, data.get("trades", 0)))
            if trades_n > 0:
                avg_ret = float(cast(float, data.get("avg_return", 0.0)))
                tot_ret = float(cast(float, data.get("total_return", 0.0)))
                regime_perf[regime] = {
                    "avg_return": avg_ret,
                    "total_return": tot_ret,
                    "trade_count": trades_n,
                }

        return regime_perf

    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between strategies"""
        if len(self.strategy_performance) < 2:
            return pd.DataFrame()

        # Create strategy returns series
        strategy_returns = {}
        for strategy, data in self.strategy_performance.items():
            trades_n = int(cast(int, data.get("trades", 0)))
            if trades_n > 0:
                # Use average return as proxy for strategy performance
                strategy_returns[strategy] = float(
                    cast(float, data.get("total_return", 0.0))
                ) / float(trades_n)

        if len(strategy_returns) < 2:
            return pd.DataFrame()

        # Create correlation matrix
        df = pd.DataFrame([strategy_returns])
        return df.corr()

    def generate_report(self, report_type: str = "comprehensive") -> dict[str, object]:
        """Generate performance report"""
        metrics = self.calculate_metrics()

        if report_type == "summary":
            return {
                "total_return": f"{metrics.total_return:.2%}",
                "annualized_return": f"{metrics.annualized_return:.2%}",
                "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                "max_drawdown": f"{metrics.max_drawdown:.2%}",
                "win_rate": f"{metrics.win_rate:.2%}",
                "total_trades": metrics.total_trades,
                "current_balance": f"${self.current_balance:,.2f}",
            }

        elif report_type == "detailed":
            return {
                "returns": {
                    "total_return": f"{metrics.total_return:.2%}",
                    "annualized_return": f"{metrics.annualized_return:.2%}",
                    "daily_returns_count": len(metrics.daily_returns),
                },
                "risk": {
                    "volatility": f"{metrics.volatility:.2%}",
                    "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                    "sortino_ratio": f"{metrics.sortino_ratio:.2f}",
                    "max_drawdown": f"{metrics.max_drawdown:.2%}",
                    "var_95": f"{metrics.var_95:.2%}",
                    "cvar_95": f"{metrics.cvar_95:.2%}",
                },
                "trading": {
                    "total_trades": metrics.total_trades,
                    "winning_trades": metrics.winning_trades,
                    "losing_trades": metrics.losing_trades,
                    "win_rate": f"{metrics.win_rate:.2%}",
                    "avg_win": f"${metrics.avg_win:,.2f}",
                    "avg_loss": f"${metrics.avg_loss:,.2f}",
                    "profit_factor": f"{metrics.profit_factor:.2f}",
                    "avg_trade_duration": f"{metrics.avg_trade_duration:.1f}h",
                },
                "strategy_performance": metrics.strategy_performance,
                "regime_performance": metrics.regime_performance,
            }

        else:  # comprehensive
            return {
                "summary": {
                    "total_return": f"{metrics.total_return:.2%}",
                    "annualized_return": f"{metrics.annualized_return:.2%}",
                    "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                    "max_drawdown": f"{metrics.max_drawdown:.2%}",
                    "win_rate": f"{metrics.win_rate:.2%}",
                    "total_trades": metrics.total_trades,
                    "current_balance": f"${self.current_balance:,.2f}",
                },
                "detailed_metrics": {
                    "returns": {
                        "total_return": f"{metrics.total_return:.2%}",
                        "annualized_return": f"{metrics.annualized_return:.2%}",
                        "daily_returns_count": len(metrics.daily_returns),
                    },
                    "risk": {
                        "volatility": f"{metrics.volatility:.2%}",
                        "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
                        "sortino_ratio": f"{metrics.sortino_ratio:.2f}",
                        "max_drawdown": f"{metrics.max_drawdown:.2%}",
                        "var_95": f"{metrics.var_95:.2%}",
                        "cvar_95": f"{metrics.cvar_95:.2%}",
                    },
                    "trading": {
                        "total_trades": metrics.total_trades,
                        "winning_trades": metrics.winning_trades,
                        "losing_trades": metrics.losing_trades,
                        "win_rate": f"{metrics.win_rate:.2%}",
                        "avg_win": f"${metrics.avg_win:,.2f}",
                        "avg_loss": f"${metrics.avg_loss:,.2f}",
                        "profit_factor": f"{metrics.profit_factor:.2f}",
                        "avg_trade_duration": f"{metrics.avg_trade_duration:.1f}h",
                    },
                },
                "strategy_performance": metrics.strategy_performance,
                "regime_performance": metrics.regime_performance,
                "correlation_matrix": (
                    metrics.correlation_matrix.to_dict()
                    if not metrics.correlation_matrix.empty
                    else {}
                ),
                "period": {
                    "start_date": metrics.start_date.isoformat(),
                    "end_date": metrics.end_date.isoformat(),
                    "last_updated": metrics.last_updated.isoformat(),
                },
            }

    def export_data(self, format: str = "json") -> str:
        """Export performance data"""
        metrics = self.calculate_metrics()

        if format == "json":
            data = {
                "metrics": asdict(metrics),
                "trades_history": self.trades_history,
                "positions_history": self.positions_history,
                "daily_equity": [
                    {
                        "date": cast(datetime, entry["date"]).isoformat(),
                        "equity": entry["equity"],
                        "balance": entry["balance"],
                    }
                    for entry in self.daily_equity
                ],
                "strategy_performance": self.strategy_performance,
                "regime_performance": self.regime_performance,
            }

            return json.dumps(data, indent=2, default=str)

        elif format == "csv":
            # Export trades to CSV
            if self.trades_history:
                trades_df = pd.DataFrame(self.trades_history)
                return cast(str, trades_df.to_csv(index=False))
            else:
                return ""

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_performance_alerts(self) -> List[dict[str, object]]:
        """Get performance alerts based on thresholds"""
        metrics = self.calculate_metrics()
        alerts = []

        # Risk alerts
        if metrics.max_drawdown > 0.20:  # 20% max drawdown
            alerts.append(
                {
                    "type": "risk",
                    "severity": "high",
                    "message": f"High drawdown detected: {metrics.max_drawdown:.2%}",
                    "metric": "max_drawdown",
                    "value": metrics.max_drawdown,
                }
            )

        if metrics.sharpe_ratio < 0.5:
            alerts.append(
                {
                    "type": "performance",
                    "severity": "medium",
                    "message": f"Low Sharpe ratio: {metrics.sharpe_ratio:.2f}",
                    "metric": "sharpe_ratio",
                    "value": metrics.sharpe_ratio,
                }
            )

        if metrics.win_rate < 0.4:
            alerts.append(
                {
                    "type": "trading",
                    "severity": "medium",
                    "message": f"Low win rate: {metrics.win_rate:.2%}",
                    "metric": "win_rate",
                    "value": metrics.win_rate,
                }
            )

        # Strategy-specific alerts
        for strategy, perf in metrics.strategy_performance.items():
            if isinstance(perf, dict) and "win_rate" in perf:
                if float(cast(float, perf["win_rate"])) < 0.3:
                    alerts.append(
                        {
                            "type": "strategy",
                            "severity": "medium",
                            "message": f"Poor performance for strategy {strategy}: {perf['win_rate']:.2%} win rate",
                            "strategy": strategy,
                            "metric": "win_rate",
                            "value": float(cast(float, perf["win_rate"])),
                        }
                    )

        return alerts
