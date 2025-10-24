"""Risk analytics helpers aligned with the high-impact roadmap."""

from .var import (
    compute_historical_var,
    compute_parametric_var,
    compute_monte_carlo_var,
)
from .monte_carlo import (
    MonteCarloSimulationResult,
    simulate_geometric_brownian_motion,
)
from .expected_shortfall import (
    compute_historical_expected_shortfall,
    compute_parametric_expected_shortfall,
)
from .volatility_target import (
    VolatilityTargetAllocation,
    calculate_realised_volatility,
    determine_target_allocation,
)
from .rolling_cvar import (
    RollingCVaRMeasurement,
    RollingCVaRMonitor,
)
from .volatility_regime import (
    VolatilityRegime,
    VolatilityRegimeAssessment,
    VolatilityRegimeThresholds,
    classify_volatility_regime,
)

__all__ = [
    "compute_historical_var",
    "compute_parametric_var",
    "compute_monte_carlo_var",
    "MonteCarloSimulationResult",
    "simulate_geometric_brownian_motion",
    "compute_historical_expected_shortfall",
    "compute_parametric_expected_shortfall",
    "VolatilityTargetAllocation",
    "calculate_realised_volatility",
    "determine_target_allocation",
    "RollingCVaRMeasurement",
    "RollingCVaRMonitor",
    "VolatilityRegime",
    "VolatilityRegimeAssessment",
    "VolatilityRegimeThresholds",
    "classify_volatility_regime",
]
