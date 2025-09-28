import numpy as np
import pytest

from src.config.risk.risk_config import RiskConfig
from src.data_foundation.config.sizing_config import SizingConfig
from src.risk.risk_manager_impl import RiskManagerImpl
from src.sensory.what.volatility_engine import VolConfig
from src.trading.risk.market_regime_detector import MarketRegimeDetector


def _generate_prices(volatility: float, *, seed: int, length: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, volatility, size=length)
    return 100.0 * np.exp(np.cumsum(returns))


def test_risk_budget_scales_with_detected_regime() -> None:
    sizing = SizingConfig(regime_multipliers={"calm": 1.2, "normal": 1.0, "storm": 0.4})
    vol_cfg = VolConfig(calm_thr=0.08, storm_thr=0.22)
    detector = MarketRegimeDetector(vol_config=vol_cfg, sizing_config=sizing)

    manager = RiskManagerImpl(
        initial_balance=100_000.0,
        risk_config=RiskConfig(),
        market_regime_detector=detector,
        sizing_config=sizing,
    )

    base_budget = manager._compute_risk_budget()

    calm_prices = _generate_prices(0.002, seed=101)
    calm_result = manager.update_market_regime({"close": calm_prices, "periods_per_year": 252})

    assert calm_result.regime.value == "calm"
    assert manager._regime_risk_multiplier == pytest.approx(sizing.regime_multipliers["calm"])
    assert manager._compute_risk_budget() == pytest.approx(
        base_budget * sizing.regime_multipliers["calm"]
    )

    storm_prices = _generate_prices(0.05, seed=202)
    storm_result = manager.update_market_regime({"close": storm_prices, "periods_per_year": 252})

    assert storm_result.regime.value == "storm"
    assert manager._regime_risk_multiplier == pytest.approx(sizing.regime_multipliers["storm"])
    assert manager._compute_risk_budget() == pytest.approx(
        base_budget * sizing.regime_multipliers["storm"]
    )


def test_risk_budget_blocks_when_gate_triggers() -> None:
    sizing = SizingConfig(regime_multipliers={"calm": 1.0, "normal": 0.9, "storm": 0.6})
    vol_cfg = VolConfig(
        calm_thr=0.05,
        storm_thr=0.18,
        use_regime_gate=True,
        block_regime="storm",
        gate_mode="block",
    )
    detector = MarketRegimeDetector(vol_config=vol_cfg, sizing_config=sizing)

    manager = RiskManagerImpl(
        initial_balance=50_000.0,
        risk_config=RiskConfig(),
        market_regime_detector=detector,
        sizing_config=sizing,
    )

    base_budget = manager._compute_risk_budget()
    assert base_budget > 0

    storm_prices = _generate_prices(0.045, seed=303)
    result = manager.update_market_regime({"close": storm_prices, "periods_per_year": 252})

    assert result.regime.value == "storm"
    assert result.blocked is True
    assert manager._regime_risk_multiplier == 0.0
    assert manager._compute_risk_budget() == pytest.approx(0.0)
