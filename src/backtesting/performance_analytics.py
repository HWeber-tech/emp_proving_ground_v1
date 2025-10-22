"""Performance analytics utilities for backtesting results.

This module focuses on metrics frequently used when evaluating systematic
strategies.  The intent is to provide reliable, dependency-light helpers that
can be composed by orchestration layers after a backtest run completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence

__all__ = [
    "DrawdownEvent",
    "PerformanceMetrics",
    "TradeAttribution",
    "compute_drawdown_events",
    "compute_performance_metrics",
    "compute_trade_attribution",
]


@dataclass(slots=True)
class DrawdownEvent:
    """Represents a single drawdown episode within an equity curve."""

    start_index: int
    trough_index: int
    recovery_index: int | None
    depth: float
    duration: int


@dataclass(slots=True)
class TradeAttribution:
    """Aggregated trade-level statistics."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    expectancy: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    average_duration: float | None


@dataclass(slots=True)
class PerformanceMetrics:
    """Summary metrics for a backtest equity curve."""

    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    average_drawdown: float
    drawdowns: tuple[DrawdownEvent, ...]
    trade_attribution: TradeAttribution


def _ensure_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _normalise_equity_curve(equity_curve: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(equity_curve, Sequence):
        raise TypeError("equity_curve must be a sequence of numbers")
    if len(equity_curve) < 2:
        raise ValueError("equity_curve must contain at least two observations")
    tuple_curve = tuple(float(x) for x in equity_curve)
    if any(value <= 0 for value in tuple_curve):
        raise ValueError("equity_curve values must be positive to compute returns")
    return tuple_curve


def _normalise_returns(returns: Sequence[float] | None, equity_curve: Sequence[float]) -> tuple[float, ...]:
    if returns is not None:
        if not isinstance(returns, Sequence):
            raise TypeError("returns must be a sequence of numbers")
        if len(returns) != len(equity_curve) - 1:
            raise ValueError("returns must have length len(equity_curve) - 1")
        return tuple(float(x) for x in returns)

    # Compute simple returns from the equity curve.
    computed_returns = []
    for previous, current in zip(equity_curve, equity_curve[1:]):
        if previous == 0:
            raise ValueError("equity curve contains zero value, cannot compute returns")
        computed_returns.append((current - previous) / previous)
    return tuple(computed_returns)


def _annualise_return(total_return: float, periods: int, periods_per_year: int) -> float:
    if periods == 0:
        return 0.0
    compounded = 1.0 + total_return
    if compounded <= 0:
        return -1.0
    years = periods / periods_per_year
    return compounded ** (1.0 / years) - 1.0 if years else total_return


def _annualised_volatility(returns: Sequence[float], periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    std_dev = pstdev(returns)
    return std_dev * sqrt(periods_per_year)


def _sharpe_ratio(returns: Sequence[float], risk_free_rate: float, periods_per_year: int) -> float:
    if not returns:
        return 0.0
    excess_rate = risk_free_rate / periods_per_year
    excess_returns = [r - excess_rate for r in returns]
    volatility = pstdev(returns)
    if volatility == 0:
        return 0.0
    mean_excess = fmean(excess_returns)
    return (mean_excess / volatility) * sqrt(periods_per_year)


def _sortino_ratio(
    returns: Sequence[float],
    risk_free_rate: float,
    downside_threshold: float,
    periods_per_year: int,
) -> float:
    if not returns:
        return 0.0
    target_rate = max(risk_free_rate / periods_per_year, downside_threshold)
    downside_diffs = [target_rate - r for r in returns if r < target_rate]
    if not downside_diffs:
        return float("inf") if fmean(returns) > target_rate else 0.0
    downside_deviation = sqrt(sum(diff * diff for diff in downside_diffs) / len(returns))
    if downside_deviation == 0:
        return 0.0
    mean_return = fmean(returns)
    return (mean_return - target_rate) / downside_deviation * sqrt(periods_per_year)


def compute_drawdown_events(equity_curve: Sequence[float]) -> tuple[DrawdownEvent, ...]:
    """Return drawdown events for the provided equity curve."""

    curve = _normalise_equity_curve(equity_curve)
    running_max = curve[0]
    peak_index = 0
    in_drawdown = False
    current_min = 0.0
    trough_index = 0
    current_duration = 0
    events: list[DrawdownEvent] = []

    for index, value in enumerate(curve[1:], start=1):
        if value >= running_max:
            if in_drawdown:
                events.append(
                    DrawdownEvent(
                        start_index=peak_index,
                        trough_index=trough_index,
                        recovery_index=index,
                        depth=abs(current_min),
                        duration=current_duration,
                    )
                )
                in_drawdown = False
                current_duration = 0
            running_max = value
            peak_index = index
            current_min = 0.0
            continue

        drawdown = (value / running_max) - 1.0
        if not in_drawdown:
            in_drawdown = True
            trough_index = index
            current_min = drawdown
            current_duration = 1
        else:
            current_duration += 1
            if drawdown < current_min:
                current_min = drawdown
                trough_index = index

    if in_drawdown:
        events.append(
            DrawdownEvent(
                start_index=peak_index,
                trough_index=trough_index,
                recovery_index=None,
                depth=abs(current_min),
                duration=current_duration,
            )
        )

    return tuple(events)


def compute_trade_attribution(trades: Iterable[Mapping[str, float]] | None) -> TradeAttribution:
    """Aggregate trade-level attribution metrics.

    Trades should expose at least a ``pnl`` field and may optionally provide a
    ``duration`` field measured in the same units for all trades.
    """

    if not trades:
        return TradeAttribution(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
            average_duration=None,
        )

    pnl_values: list[float] = []
    durations: list[float] = []
    for trade in trades:
        if "pnl" not in trade:
            raise KeyError("trade mappings must include a 'pnl' field")
        pnl = float(trade["pnl"])
        pnl_values.append(pnl)
        if "duration" in trade:
            durations.append(float(trade["duration"]))

    total_trades = len(pnl_values)
    winning = [pnl for pnl in pnl_values if pnl > 0]
    losing = [pnl for pnl in pnl_values if pnl < 0]
    breakeven = total_trades - len(winning) - len(losing)

    gross_profit = sum(winning) if winning else 0.0
    gross_loss = sum(losing) if losing else 0.0
    net_profit = gross_profit + gross_loss

    average_win = fmean(winning) if winning else 0.0
    average_loss = fmean(losing) if losing else 0.0
    win_rate = len(winning) / total_trades if total_trades else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss else float("inf") if gross_profit > 0 else 0.0
    expectancy = net_profit / total_trades if total_trades else 0.0
    average_duration = fmean(durations) if durations else None

    return TradeAttribution(
        total_trades=total_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        breakeven_trades=breakeven,
        win_rate=win_rate,
        average_win=average_win,
        average_loss=average_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        average_duration=average_duration,
    )


def compute_performance_metrics(
    equity_curve: Sequence[float],
    *,
    returns: Sequence[float] | None = None,
    trades: Iterable[Mapping[str, float]] | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    downside_threshold: float = 0.0,
) -> PerformanceMetrics:
    """Compute standard performance metrics for a backtest equity curve."""

    _ensure_positive(periods_per_year, "periods_per_year")
    curve = _normalise_equity_curve(equity_curve)
    period_returns = _normalise_returns(returns, curve)

    total_return = curve[-1] / curve[0] - 1.0
    annualised_return = _annualise_return(total_return, len(period_returns), periods_per_year)
    annualised_volatility = _annualised_volatility(period_returns, periods_per_year)
    sharpe = _sharpe_ratio(period_returns, risk_free_rate, periods_per_year)
    sortino = _sortino_ratio(period_returns, risk_free_rate, downside_threshold, periods_per_year)

    drawdowns = compute_drawdown_events(curve)
    if drawdowns:
        max_drawdown = max(event.depth for event in drawdowns)
        max_drawdown_duration = max(event.duration for event in drawdowns)
        average_drawdown = fmean(event.depth for event in drawdowns)
    else:
        max_drawdown = 0.0
        max_drawdown_duration = 0
        average_drawdown = 0.0

    calmar = annualised_return / max_drawdown if max_drawdown else float("inf") if annualised_return > 0 else 0.0

    trade_attribution = compute_trade_attribution(trades)

    return PerformanceMetrics(
        total_return=total_return,
        annualised_return=annualised_return,
        annualised_volatility=annualised_volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        average_drawdown=average_drawdown,
        drawdowns=drawdowns,
        trade_attribution=trade_attribution,
    )
