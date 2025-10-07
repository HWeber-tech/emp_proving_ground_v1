"""Risk management module for EMP system."""

from __future__ import annotations

from .manager import RiskManager, create_risk_manager, get_risk_manager
from .real_risk_manager import RealRiskConfig, RealRiskManager
from .analytics import (
    VolatilityTargetAllocation,
    VolatilityRegime,
    VolatilityRegimeAssessment,
    VolatilityRegimeThresholds,
    calculate_realised_volatility,
    classify_volatility_regime,
    determine_target_allocation,
)
from .reporting import (
    BudgetUtilisation,
    ExposureBreakdown,
    PortfolioRiskLimits,
    RiskReport,
    generate_capital_efficiency_memo,
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
    "RiskManager",
    "create_risk_manager",
    "get_risk_manager",
    "BudgetUtilisation",
    "ExposureBreakdown",
    "PortfolioRiskLimits",
    "RiskReport",
    "generate_capital_efficiency_memo",
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
    "VolatilityTargetAllocation",
    "calculate_realised_volatility",
    "determine_target_allocation",
    "VolatilityRegime",
    "VolatilityRegimeAssessment",
    "VolatilityRegimeThresholds",
    "classify_volatility_regime",
]
