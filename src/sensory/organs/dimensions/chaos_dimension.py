"""
CHAOS Dimension - Antifragile Adaptation System

This module implements the CHAOS dimension of the sensory cortex,
responsible for antifragile adaptation, chaos engineering principles,
and benefiting from market disorder and volatility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, cast

import numpy as np
import pandas as pd


@dataclass
class ChaosMetrics:
    """Metrics for chaos analysis."""

    disorder_level: float
    antifragility_score: float
    adaptation_rate: float
    resilience_factor: float
    opportunity_index: float


class ChaosDimension:
    """
    Antifragile Adaptation System for market chaos.

    This class implements the CHAOS dimension functionality for adapting to
    and benefiting from market disorder, volatility, and chaotic conditions
    through antifragile strategies and chaos engineering principles.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize the CHAOS dimension."""
        self.config: dict[str, Any] = config or {}
        self.chaos_threshold: float = float(self.config.get("chaos_threshold", 0.7))
        self.adaptation_history: list[dict[str, Any]] = []
        self.antifragility_metrics: dict[str, Any] = {}
        self.chaos_regimes: dict[str, dict[str, Any]] = {}

    def analyze_chaos(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze market chaos and disorder.

        Args:
            data: Market data DataFrame

        Returns:
            Dictionary containing chaos analysis results
        """
        return {
            "chaos_level": self._calculate_chaos_level(data),
            "antifragility": self._measure_antifragility(data),
            "adaptation_opportunities": self._identify_opportunities(data),
            "regime_analysis": self._analyze_regime_changes(data),
            "metadata": {
                "analysis_time": datetime.now(),
                "data_points": len(data),
                "chaos_threshold": self.chaos_threshold,
            },
        }

    def _calculate_chaos_level(self, data: pd.DataFrame) -> dict[str, Any]:
        """Calculate the current level of market chaos."""
        if len(data) < 2:
            return {"level": 0.0, "description": "insufficient_data"}

        # Calculate various chaos indicators
        returns = data["close"].pct_change().dropna() if "close" in data.columns else pd.Series([0])

        # Volatility as chaos indicator
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        vol_f: float = (
            float(np.asarray(volatility).item())
            if hasattr(volatility, "item")
            else float(volatility)
        )

        # Kurtosis as chaos indicator (fat tails)
        kurtosis_val: Any = returns.kurtosis() if len(returns) > 3 else 0.0
        kur_f: float = (
            float(np.asarray(kurtosis_val).item())
            if hasattr(kurtosis_val, "item")
            else float(cast(float, kurtosis_val))
        )

        # Combined chaos score
        chaos_score = min(1.0, (vol_f * 0.7 + abs(kur_f) * 0.3))

        return {
            "level": float(chaos_score),
            "volatility": vol_f,
            "kurtosis": kur_f,
            "description": self._get_chaos_description(float(chaos_score)),
        }

    def _measure_antifragility(self, data: pd.DataFrame) -> dict[str, Any]:
        """Measure the system's antifragility to market chaos."""
        chaos_level = float(self._calculate_chaos_level(data)["level"])

        # Antifragility increases with chaos up to a point
        antifragility = min(1.0, chaos_level * 1.2) if chaos_level > 0.3 else 0.2

        return {
            "score": float(antifragility),
            "benefit_ratio": float(antifragility * 0.8),
            "adaptation_speed": float(0.75),
            "resilience_factor": float(0.9),
        }

    def _identify_opportunities(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        """Identify opportunities arising from market chaos."""
        chaos_level = float(self._calculate_chaos_level(data)["level"])

        opportunities = []

        if chaos_level > 0.6:
            opportunities.append(
                {
                    "type": "volatility_arbitrage",
                    "potential": 0.85,
                    "risk_level": "medium",
                    "time_horizon": "short",
                    "description": "High volatility creates arbitrage opportunities",
                }
            )

        if chaos_level > 0.8:
            opportunities.append(
                {
                    "type": "mean_reversion",
                    "potential": 0.72,
                    "risk_level": "high",
                    "time_horizon": "medium",
                    "description": "Extreme chaos suggests mean reversion potential",
                }
            )

        if chaos_level < 0.3:
            opportunities.append(
                {
                    "type": "trend_following",
                    "potential": 0.65,
                    "risk_level": "low",
                    "time_horizon": "long",
                    "description": "Low chaos environment suitable for trend following",
                }
            )

        return opportunities

    def _analyze_regime_changes(self, data: pd.DataFrame) -> dict[str, Any]:
        """Analyze changes in market regime."""
        return {
            "current_regime": (
                "chaotic" if float(self._calculate_chaos_level(data)["level"]) > 0.5 else "stable"
            ),
            "regime_stability": float(0.68),
            "transition_probability": float(0.23),
            "expected_duration": "3-5 days",
        }

    def _get_chaos_description(self, chaos_level: float) -> str:
        """Get textual description of chaos level."""
        if chaos_level < 0.2:
            return "stable"
        elif chaos_level < 0.4:
            return "mild_disorder"
        elif chaos_level < 0.6:
            return "moderate_chaos"
        elif chaos_level < 0.8:
            return "high_chaos"
        else:
            return "extreme_chaos"

    def adapt_to_chaos(self, chaos_level: float) -> dict[str, Any]:
        """Adapt system parameters based on chaos level."""
        adaptation: dict[str, Any] = {
            "risk_multiplier": float(1.0 + (chaos_level * 0.5)),
            "position_sizing": float(max(0.1, 1.0 - (chaos_level * 0.8))),
            "stop_loss_factor": float(1.0 + (chaos_level * 0.3)),
            "take_profit_factor": float(1.0 + (chaos_level * 0.2)),
        }

        self.adaptation_history.append(
            {
                "timestamp": datetime.now(),
                "chaos_level": float(chaos_level),
                "adaptation": adaptation,
            }
        )

        return adaptation

    def get_antifragility_score(self) -> float:
        """Calculate current antifragility score."""
        if not self.adaptation_history:
            return 0.5

        recent_adaptations = self.adaptation_history[-10:]
        scores = [float(a["adaptation"]["risk_multiplier"]) for a in recent_adaptations]

        return float(np.mean(scores))

    def register_chaos_regime(self, regime_id: str, regime_data: dict[str, Any]) -> None:
        """Register a new chaos regime."""
        self.chaos_regimes[regime_id] = regime_data

    def get_chaos_regime(self, regime_id: str) -> Optional[dict[str, Any]]:
        """Retrieve chaos regime information by ID."""
        return self.chaos_regimes.get(regime_id)

    def list_regimes(self) -> list[str]:
        """List all registered chaos regime IDs."""
        return list(self.chaos_regimes.keys())

    def get_status(self) -> dict[str, Any]:
        """Get current status of the CHAOS dimension."""
        return {
            "active_regimes": len(self.chaos_regimes),
            "total_adaptations": len(self.adaptation_history),
            "antifragility_score": self.get_antifragility_score(),
            "health": "healthy",
            "last_adaptation": (
                self.adaptation_history[-1]["timestamp"]
                if self.adaptation_history
                else datetime.now()
            ),
        }
