"""Advanced performance tracking system with reporting helpers."""

import json
import logging
import math
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, cast

import pandas as pd

from .performance_metrics import (
    PerformanceMetrics,
    calculate_annualized_return,
    calculate_correlation_matrix,
    calculate_max_drawdown,
    calculate_regime_performance,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_strategy_performance,
    calculate_trading_metrics,
    calculate_var_cvar,
    create_empty_metrics,
)

logger = logging.getLogger(__name__)


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
        """Calculate comprehensive performance metrics."""

        if (
            self.metrics is not None
            and self.last_calculation is not None
            and datetime.now() - self.last_calculation < timedelta(minutes=5)
            and not force_recalculate
        ):
            return self.metrics

        if not self.daily_equity:
            logger.warning("No equity data available for metrics calculation")
            empty_metrics = create_empty_metrics(datetime.now())
            self.metrics = empty_metrics
            self.last_calculation = datetime.now()
            return empty_metrics

        frame = pd.DataFrame(self.daily_equity)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date")
        frame["daily_return"] = frame["equity"].pct_change()

        daily_return_series = frame["daily_return"].dropna()
        daily_returns = daily_return_series.tolist()

        total_return = (frame["equity"].iloc[-1] - self.initial_balance) / self.initial_balance
        annualized_return = calculate_annualized_return(frame, self.initial_balance)

        if daily_return_series.empty:
            volatility = 0.0
        else:
            std = float(daily_return_series.std(ddof=0))
            volatility = float(std * math.sqrt(252))

        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        sortino_ratio = calculate_sortino_ratio(daily_returns)
        max_drawdown = calculate_max_drawdown(frame["equity"])
        var_95, cvar_95 = calculate_var_cvar(daily_returns)

        trading_metrics = calculate_trading_metrics(self.trades_history)
        tm_total_trades = int(cast(int, trading_metrics["total_trades"]))
        tm_winning_trades = int(cast(int, trading_metrics["winning_trades"]))
        tm_losing_trades = int(cast(int, trading_metrics["losing_trades"]))
        tm_win_rate = float(cast(float, trading_metrics["win_rate"]))
        tm_avg_win = float(cast(float, trading_metrics["avg_win"]))
        tm_avg_loss = float(cast(float, trading_metrics["avg_loss"]))
        tm_profit_factor = float(cast(float, trading_metrics["profit_factor"]))
        tm_avg_trade_duration = float(cast(float, trading_metrics["avg_trade_duration"]))

        strategy_perf = calculate_strategy_performance(self.strategy_performance)
        regime_perf = calculate_regime_performance(self.regime_performance)
        correlation_matrix = calculate_correlation_matrix(self.strategy_performance)

        metrics = PerformanceMetrics(
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
            start_date=frame["date"].min(),
            end_date=frame["date"].max(),
            last_updated=datetime.now(),
        )

        self.metrics = metrics
        self.last_calculation = datetime.now()

        logger.info(
            f"Performance metrics calculated: {total_return:.2%} total return, {sharpe_ratio:.2f} Sharpe"
        )
        return metrics

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
