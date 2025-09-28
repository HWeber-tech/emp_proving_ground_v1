from __future__ import annotations

import numpy as np
import pytest

from src.trading.strategies.analytics import (
    compute_performance_attribution,
    result_to_dataframe,
)


def test_performance_attribution_basic() -> None:
    returns = [0.01, 0.015, -0.005, 0.02, 0.0]
    features = {
        "momentum": [1.2, 0.9, -0.5, 1.1, 0.0],
        "volatility": [0.5, 0.4, 0.6, 0.7, 0.2],
    }

    result = compute_performance_attribution(returns, features)

    assert result.total_return == pytest.approx(sum(returns))
    assert len(result.contributions) == 2
    assert {contrib.name for contrib in result.contributions} == {"momentum", "volatility"}

    predicted_mean = result.intercept + sum(c.contribution for c in result.contributions)
    assert predicted_mean == pytest.approx(result.average_return, rel=1e-6, abs=1e-9)

    table = result_to_dataframe(result)
    assert set(table["name"]) == {"momentum", "volatility"}
    assert table["coefficient"].dtype == np.float64


def test_performance_attribution_handles_nans() -> None:
    returns = [0.01, np.nan, 0.02, -0.01]
    features = {
        "signal": [0.5, 0.6, np.nan, 0.4],
    }

    result = compute_performance_attribution(returns, features)
    assert len(result.contributions) == 1
    # Only two valid observations should remain after dropping NaNs
    assert result.total_return == pytest.approx(0.01 - 0.01)


def test_performance_attribution_regularisation_validation() -> None:
    with pytest.raises(ValueError):
        compute_performance_attribution([0.01, 0.02], {"x": [1.0, 2.0]}, regularisation=-1.0)
