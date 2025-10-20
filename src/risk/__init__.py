"""Risk management module for EMP system."""

from __future__ import annotations

from importlib import import_module

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
from .position_sizing import (
    kelly_fraction,
    normalise_quantile_triplet,
    position_size,
    quantile_edge_ratio,
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

_RISK_MANAGER_EXPORTS = {
    "RiskManager": ("src.risk.manager", "RiskManager"),
    "create_risk_manager": ("src.risk.manager", "create_risk_manager"),
}

__all__ = [
    "RealRiskManager",
    "RealRiskConfig",
    "RiskManager",
    "create_risk_manager",
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
    "kelly_fraction",
    "position_size",
    "normalise_quantile_triplet",
    "quantile_edge_ratio",
]


def __getattr__(name: str):
    target = _RISK_MANAGER_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
