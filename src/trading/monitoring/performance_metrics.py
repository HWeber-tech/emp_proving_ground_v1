"""Reusable performance analytics helpers for trading monitors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence, Tuple, cast

import numpy as np
import pandas as pd


def _safe_int(value: object, default: int = 0) -> int:
    """Best-effort integer coercion that mirrors legacy tracker behavior."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _safe_float(value: object, default: float = 0.0) -> float:
    """Best-effort float coercion that mirrors legacy tracker behavior."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


@dataclass
class PerformanceMetrics:
    """Container for aggregated trading performance metrics."""

    # Returns metrics
    total_return: float
    annualized_return: float
    daily_returns: list[float]

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
    strategy_performance: dict[str, dict[str, object]]
    regime_performance: dict[str, dict[str, object]]
    correlation_matrix: pd.DataFrame

    # Timestamps
    start_date: datetime
    end_date: datetime
    last_updated: datetime


def create_empty_metrics(timestamp: datetime) -> PerformanceMetrics:
    """Return a zeroed metrics container used when no data is available."""

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
        start_date=timestamp,
        end_date=timestamp,
        last_updated=timestamp,
    )


def calculate_annualized_return(df: pd.DataFrame, initial_balance: float) -> float:
    """Calculate annualized return given an equity curve data frame."""

    if len(df) < 2:
        return 0.0

    total_days = (df["date"].max() - df["date"].min()).days
    if total_days <= 0:
        return 0.0

    total_return = (df["equity"].iloc[-1] - initial_balance) / initial_balance
    return float((1 + total_return) ** (365 / total_days) - 1)


def calculate_sharpe_ratio(returns: Sequence[float]) -> float:
    """Calculate the annualized Sharpe ratio for a series of returns."""

    if not returns:
        return 0.0

    returns_array = np.asarray(list(returns))
    std = returns_array.std()
    if std == 0:
        return 0.0

    return float(returns_array.mean() / std * np.sqrt(252))


def calculate_sortino_ratio(returns: Sequence[float]) -> float:
    """Calculate the annualized Sortino ratio for a series of returns."""

    if not returns:
        return 0.0

    returns_array = np.asarray(list(returns))
    negative_returns = returns_array[returns_array < 0]

    if len(negative_returns) == 0:
        return float("inf") if returns_array.mean() > 0 else 0.0

    downside_deviation = negative_returns.std()
    if downside_deviation == 0:
        return 0.0

    return float(returns_array.mean() / downside_deviation * np.sqrt(252))


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown for an equity curve."""

    if len(equity) < 2:
        return 0.0

    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    return float(abs(drawdown.min()))


def calculate_var_cvar(returns: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Value at Risk (VaR) and Conditional VaR at the provided confidence."""

    if not returns:
        return 0.0, 0.0

    returns_array = np.asarray(list(returns))
    var = np.percentile(returns_array, (1 - confidence) * 100)
    tail = returns_array[returns_array <= var]
    cvar = tail.mean() if len(tail) else var
    return float(abs(var)), float(abs(cvar))


def calculate_trading_metrics(trades_history: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate trade-level metrics into summary statistics."""

    if not trades_history:
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

    trades_df = pd.DataFrame(trades_history)

    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    avg_win = float(wins["pnl"].mean()) if winning_trades > 0 else 0.0
    avg_loss = float(abs(losses["pnl"].mean())) if losing_trades > 0 else 0.0

    total_wins = float(wins["pnl"].sum()) if winning_trades > 0 else 0.0
    total_losses = float(abs(losses["pnl"].sum())) if losing_trades > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    if "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
        trades_df = trades_df.assign(
            duration=pd.to_datetime(trades_df["exit_time"])
            - pd.to_datetime(trades_df["entry_time"])
        )
        duration_mean = cast(pd.Timedelta, trades_df["duration"].mean())
        avg_trade_duration = float(duration_mean.total_seconds() / 3600)
    else:
        avg_trade_duration = 0.0

    return {
        "total_trades": int(total_trades),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "avg_trade_duration": float(avg_trade_duration),
    }


def calculate_strategy_performance(
    raw_strategy_performance: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize per-strategy metrics in a stable dictionary."""

    result: dict[str, dict[str, object]] = {}

    for strategy, data in raw_strategy_performance.items():
        mapping = cast(Mapping[str, object], data)
        trades_n = _safe_int(mapping.get("trades", 0))
        if trades_n <= 0:
            continue

        wins_n = _safe_float(mapping.get("wins", 0.0))
        total_ret = _safe_float(mapping.get("total_return", 0.0))
        win_rate = wins_n / trades_n if trades_n > 0 else 0.0
        avg_return = total_ret / trades_n if trades_n > 0 else 0.0

        result[strategy] = {
            "win_rate": float(win_rate),
            "avg_return": float(avg_return),
            "total_pnl": _safe_float(mapping.get("total_pnl", 0.0)),
            "trade_count": trades_n,
        }

    return result


def calculate_regime_performance(
    raw_regime_performance: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize performance by market regime."""

    result: dict[str, dict[str, object]] = {}

    for regime, data in raw_regime_performance.items():
        mapping = cast(Mapping[str, object], data)
        trades_n = _safe_int(mapping.get("trades", 0))
        if trades_n <= 0:
            continue

        avg_ret = _safe_float(mapping.get("avg_return", 0.0))
        tot_ret = _safe_float(mapping.get("total_return", 0.0))

        result[regime] = {
            "avg_return": float(avg_ret),
            "total_return": float(tot_ret),
            "trade_count": trades_n,
        }

    return result


def calculate_correlation_matrix(
    strategy_performance: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Return a correlation matrix between strategies based on average returns."""

    if len(strategy_performance) < 2:
        return pd.DataFrame()

    strategy_returns: dict[str, float] = {}
    for strategy, data in strategy_performance.items():
        mapping = cast(Mapping[str, object], data)
        trades_n = _safe_int(mapping.get("trades", 0))
        if trades_n <= 0:
            continue
        total_return = _safe_float(mapping.get("total_return", 0.0))
        strategy_returns[strategy] = total_return / trades_n if trades_n else 0.0

    if len(strategy_returns) < 2:
        return pd.DataFrame()

    df = pd.DataFrame([strategy_returns])
    return df.corr()


__all__ = [
    "PerformanceMetrics",
    "calculate_annualized_return",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_var_cvar",
    "calculate_trading_metrics",
    "calculate_strategy_performance",
    "calculate_regime_performance",
    "calculate_correlation_matrix",
    "create_empty_metrics",
]
