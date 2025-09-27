"""Risk management module for EMP system."""

from __future__ import annotations

from .manager import (
    CircuitBreakerEvent,
    CircuitBreakerState,
    DrawdownCircuitBreaker,
)
from .real_risk_manager import RealRiskConfig, RealRiskManager
from .reporting import (
    ExposureBreakdown,
    PortfolioRiskLimits,
    RiskReport,
    generate_risk_report,
    load_portfolio_limits,
    render_risk_report_json,
    render_risk_report_markdown,
)
from .telemetry import (
    RiskLimitCheck,
    RiskLimitStatus,
    RiskTelemetrySnapshot,
    RiskThresholdType,
    evaluate_risk_posture,
    format_risk_markdown,
    publish_risk_snapshot,
)
from .sizing import (
    check_classification_limits,
    compute_classified_exposure,
    kelly_fraction,
    kelly_position_size,
    volatility_target_position_size,
)

__all__ = [
    "RealRiskManager",
    "RealRiskConfig",
    "DrawdownCircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerEvent",
    "ExposureBreakdown",
    "PortfolioRiskLimits",
    "RiskReport",
    "generate_risk_report",
    "load_portfolio_limits",
    "render_risk_report_json",
    "render_risk_report_markdown",
    "RiskLimitCheck",
    "RiskLimitStatus",
    "RiskTelemetrySnapshot",
    "RiskThresholdType",
    "evaluate_risk_posture",
    "format_risk_markdown",
    "publish_risk_snapshot",
    "kelly_fraction",
    "kelly_position_size",
    "volatility_target_position_size",
    "compute_classified_exposure",
    "check_classification_limits",
]
