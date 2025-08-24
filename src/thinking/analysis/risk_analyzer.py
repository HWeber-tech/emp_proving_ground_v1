"""
EMP Risk Analyzer v1.1

Risk analysis and assessment for the thinking layer.
Provides comprehensive risk metrics and analysis for trading strategies.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from pandas import Series

from ...core.events import AnalysisResult, RiskMetrics, TradeIntent

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Risk analyzer for cognitive assessment of trading risks."""

    def __init__(self, confidence_level: float = 0.95, lookback_period: int = 252):
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.risk_history: list[dict[str, object]] = []

        logger.info(f"Risk Analyzer initialized with {confidence_level:.0%} confidence level")

    def analyze_risk(
        self, trade_history: list[TradeIntent], market_data: Optional[list[object]] = None
    ) -> AnalysisResult:
        """Analyze trading risk and generate risk metrics."""
        try:
            if not trade_history:
                logger.warning("Empty trade history provided for risk analysis")
                return self._create_default_analysis()

            # Convert trade history to risk data
            risk_data = self._convert_trades_to_risk_data(trade_history)

            # Calculate risk metrics
            metrics = self._calculate_risk_metrics(risk_data, market_data)

            # Calculate confidence based on data quality
            confidence = self._calculate_risk_confidence(risk_data)

            # Create analysis result
            result = AnalysisResult(
                timestamp=datetime.now(),
                analysis_type="risk_analysis",
                result={
                    "risk_metrics": metrics.__dict__,
                    "trade_count": len(trade_history),
                    "analysis_period": self._calculate_analysis_period(trade_history),
                    "confidence_level": self.confidence_level,
                    "risk_assessment": self._assess_risk_level(metrics),
                },
                confidence=confidence,
                metadata={
                    "analyzer_version": "1.1.0",
                    "method": "comprehensive_risk_analysis",
                    "confidence_level": self.confidence_level,
                    "lookback_period": self.lookback_period,
                },
            )

            # Store in history
            self.risk_history.append(
                {
                    "timestamp": result.timestamp,
                    "trade_count": len(trade_history),
                    "var_95": metrics.var_95,
                    "cvar_95": metrics.cvar_95,
                    "max_drawdown": metrics.max_drawdown,
                    "risk_score": metrics.risk_score,
                    "confidence": confidence,
                }
            )

            logger.debug(f"Risk analyzed: VaR95={metrics.var_95:.2%}, CVaR95={metrics.cvar_95:.2%}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing risk: {e}")
            return self._create_default_analysis()

    def _convert_trades_to_risk_data(self, trade_history: list[TradeIntent]) -> dict[str, object]:
        """Convert trade history to risk data structure."""
        # Sort trades by timestamp
        sorted_trades = sorted(trade_history, key=lambda x: x.timestamp)

        # Extract trade returns and positions
        trade_returns = []
        position_sizes = []
        timestamps = []

        for trade in sorted_trades:
            if trade.action == "HOLD":
                continue

            # Calculate trade return (simplified)
            if trade.price and trade.quantity:
                # Simplified P&L calculation
                if trade.action == "SELL":
                    # Assume we're closing a position
                    trade_return = 0.02  # Simplified 2% return per trade
                    trade_returns.append(trade_return)
                    position_sizes.append(trade.quantity * trade.price)
                    timestamps.append(trade.timestamp)

        return {
            "trade_returns": pd.Series(trade_returns),
            "position_sizes": position_sizes,
            "timestamps": timestamps,
            "total_trades": len(trade_returns),
        }

    def _calculate_risk_metrics(
        self, risk_data: dict[str, object], market_data: Optional[list[object]] = None
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        trade_returns_obj = risk_data["trade_returns"]
        trade_returns: Series[Any] = cast(Series[Any], trade_returns_obj)
        position_sizes = risk_data["position_sizes"]

        # Value at Risk (VaR)
        var_95 = self._calculate_var(trade_returns, 0.95)
        var_99 = self._calculate_var(trade_returns, 0.99)

        # Conditional Value at Risk (CVaR)
        cvar_95 = self._calculate_cvar(trade_returns, 0.95)
        cvar_99 = self._calculate_cvar(trade_returns, 0.99)

        # Beta calculation (simplified)
        beta = self._calculate_beta(trade_returns, market_data)

        # Correlation calculation
        correlation = self._calculate_correlation(trade_returns, market_data)

        # Current drawdown
        current_drawdown = self._calculate_current_drawdown(trade_returns)

        # Risk score
        risk_score = self._calculate_risk_score(var_95, cvar_95, current_drawdown, beta)

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            correlation=correlation,
            current_drawdown=current_drawdown,
            risk_score=risk_score,
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points": len(trade_returns),
                "confidence_level": self.confidence_level,
            },
        )

    def _calculate_var(self, returns: Series[Any], confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 2:
            return 0.0

        # Historical VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)

        return float(abs(var))

    def _calculate_cvar(self, returns: Series[Any], confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) < 2:
            return 0.0

        # Historical CVaR
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)

        # Calculate expected value of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]

        cvar: float
        if len(tail_returns) > 0:
            mean_val = tail_returns.mean()
            cvar = float(mean_val)
        else:
            cvar = float(var_threshold)

        return float(abs(cvar))

    def _calculate_beta(self, returns: Series[Any], market_data: Optional[list[object]]) -> float:
        """Calculate beta relative to market."""
        if not market_data or len(returns) < 2:
            return 1.0  # Default to market beta

        try:
            # Simplified market return calculation
            # In a real implementation, you'd use actual market data
            market_returns: Series[Any] = pd.Series(
                [0.001] * len(returns)
            )  # Simplified 0.1% daily return

            # Calculate covariance and variance
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)

            if market_variance > 0:
                beta = covariance / market_variance
            else:
                beta = 1.0

            return float(beta)

        except Exception as e:
            logger.warning(f"Error calculating beta: {e}")
            return 1.0

    def _calculate_correlation(
        self, returns: Series[Any], market_data: Optional[list[object]]
    ) -> float:
        """Calculate correlation with market."""
        if not market_data or len(returns) < 2:
            return 0.0

        try:
            # Simplified market return calculation
            market_returns: Series[Any] = pd.Series([0.001] * len(returns))  # Simplified

            correlation = np.corrcoef(returns, market_returns)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.warning(f"Error calculating correlation: {e}")
            return 0.0

    def _calculate_current_drawdown(self, returns: Series[Any]) -> float:
        """Calculate current drawdown from returns."""
        if len(returns) < 2:
            return 0.0

        # Calculate cumulative returns
        cumulative_returns: Series[Any] = (1 + returns).cumprod()

        # Calculate running maximum
        running_max: Series[Any] = cumulative_returns.expanding().max()

        # Calculate current drawdown
        current_drawdown_val = (
            cumulative_returns.iloc[-1] - running_max.iloc[-1]
        ) / running_max.iloc[-1]

        return float(abs(current_drawdown_val))

    def _calculate_risk_score(
        self, var_95: float, cvar_95: float, current_drawdown: float, beta: float
    ) -> float:
        """Calculate composite risk score."""
        # Normalize risk metrics to [0, 1] scale
        var_score = min(var_95 / 0.1, 1.0)  # Normalize to 10% VaR
        cvar_score = min(cvar_95 / 0.15, 1.0)  # Normalize to 15% CVaR
        drawdown_score = min(current_drawdown / 0.2, 1.0)  # Normalize to 20% drawdown
        beta_score = min(abs(beta - 1.0) / 0.5, 1.0)  # Normalize beta deviation

        # Weighted risk score
        weights = [0.3, 0.3, 0.25, 0.15]  # VaR, CVaR, Drawdown, Beta
        risk_score = (
            var_score * weights[0]
            + cvar_score * weights[1]
            + drawdown_score * weights[2]
            + beta_score * weights[3]
        )

        return min(risk_score, 1.0)

    def _assess_risk_level(self, metrics: RiskMetrics) -> str:
        """Assess overall risk level."""
        risk_score = metrics.risk_score

        if risk_score < 0.2:
            return "LOW"
        elif risk_score < 0.4:
            return "LOW_MEDIUM"
        elif risk_score < 0.6:
            return "MEDIUM"
        elif risk_score < 0.8:
            return "MEDIUM_HIGH"
        else:
            return "HIGH"

    def _calculate_risk_confidence(self, risk_data: dict[str, object]) -> float:
        """Calculate confidence in risk analysis."""
        confidence_factors = []

        # Data sufficiency
        trade_count = cast(int, risk_data["total_trades"])
        if trade_count >= 100:
            confidence_factors.append(1.0)
        elif trade_count >= 50:
            confidence_factors.append(0.8)
        elif trade_count >= 20:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)

        # Data quality
        returns_obj = risk_data["trade_returns"]
        returns_ser: Series[Any] = cast(Series[Any], returns_obj)
        if len(returns_ser) > 0:
            # Check for extreme outliers
            q1 = returns_ser.quantile(0.25)
            q3 = returns_ser.quantile(0.75)
            iqr = q3 - q1
            outliers = returns_ser[(returns_ser < q1 - 1.5 * iqr) | (returns_ser > q3 + 1.5 * iqr)]

            outlier_ratio = len(outliers) / len(returns_ser)
            if outlier_ratio < 0.05:
                confidence_factors.append(1.0)
            elif outlier_ratio < 0.1:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)

        # Time span
        timestamps = cast(list[datetime], risk_data["timestamps"])
        if len(timestamps) > 1:
            time_span = (timestamps[-1] - timestamps[0]).days
            if time_span >= 252:  # 1 year
                confidence_factors.append(1.0)
            elif time_span >= 126:  # 6 months
                confidence_factors.append(0.8)
            elif time_span >= 63:  # 3 months
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.1)

        return float(np.mean(confidence_factors))

    def _calculate_analysis_period(self, trade_history: list[TradeIntent]) -> dict[str, object]:
        """Calculate the analysis period from trade history."""
        if not trade_history:
            return {"start": None, "end": None, "duration_days": 0}

        start_time = min(trade.timestamp for trade in trade_history)
        end_time = max(trade.timestamp for trade in trade_history)
        duration_days = (end_time - start_time).days

        return {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_days": duration_days,
        }

    def _create_default_analysis(self) -> AnalysisResult:
        """Create default analysis when risk analysis fails."""
        return AnalysisResult(
            timestamp=datetime.now(),
            analysis_type="risk_analysis",
            result={
                "risk_metrics": RiskMetrics().__dict__,
                "trade_count": 0,
                "analysis_period": {"start": None, "end": None, "duration_days": 0},
                "confidence_level": self.confidence_level,
                "risk_assessment": "UNKNOWN",
            },
            confidence=0.1,
            metadata={
                "analyzer_version": "1.1.0",
                "method": "default_fallback",
                "error": "Insufficient data for analysis",
            },
        )

    def get_risk_history(self, limit: Optional[int] = None) -> list[dict[str, object]]:
        """Get risk analysis history."""
        if limit:
            return self.risk_history[-limit:]
        return self.risk_history.copy()

    def get_risk_statistics(self) -> dict[str, object]:
        """Get statistics about risk analyses."""
        if not self.risk_history:
            return {}

        var_95s = [cast(float, h["var_95"]) for h in self.risk_history]
        cvar_95s = [cast(float, h["cvar_95"]) for h in self.risk_history]
        drawdowns = [cast(float, h["max_drawdown"]) for h in self.risk_history]
        risk_scores = [cast(float, h["risk_score"]) for h in self.risk_history]
        confidences = [cast(float, h["confidence"]) for h in self.risk_history]

        return {
            "total_analyses": len(self.risk_history),
            "average_var_95": np.mean(var_95s),
            "average_cvar_95": np.mean(cvar_95s),
            "average_drawdown": np.mean(drawdowns),
            "average_risk_score": np.mean(risk_scores),
            "average_confidence": np.mean(confidences),
            "max_var_95": max(var_95s) if var_95s else 0,
            "max_cvar_95": max(cvar_95s) if cvar_95s else 0,
            "max_risk_score": max(risk_scores) if risk_scores else 0,
        }
