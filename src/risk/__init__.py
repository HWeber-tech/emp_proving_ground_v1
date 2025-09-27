"""Risk management module for EMP system."""

from __future__ import annotations

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

__all__ = [
    "RealRiskManager",
    "RealRiskConfig",
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
]
