import numpy as np
import pytest

from src.data_foundation.config.sizing_config import SizingConfig
from src.sensory.what.volatility_engine import VolConfig
from src.trading.risk.market_regime_detector import MarketRegimeDetector


def _generate_prices(volatility: float, *, seed: int = 42, length: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, volatility, size=length)
    return 100.0 * np.exp(np.cumsum(returns))


def test_detector_identifies_calm_regime() -> None:
    sizing = SizingConfig(regime_multipliers={"calm": 1.2, "normal": 1.0, "storm": 0.5})
    vol_cfg = VolConfig(calm_thr=0.1, storm_thr=0.25)
    detector = MarketRegimeDetector(vol_config=vol_cfg, sizing_config=sizing)

    prices = _generate_prices(0.002, seed=7)
    result = detector.detect_regime({"close": prices, "periods_per_year": 252})

    assert result.regime.value == "calm"
    assert result.risk_multiplier == pytest.approx(sizing.regime_multipliers["calm"])
    assert result.confidence > 0.3


def test_detector_identifies_storm_regime_with_gate_attenuation() -> None:
    sizing = SizingConfig(regime_multipliers={"calm": 1.2, "normal": 1.0, "storm": 0.5})
    vol_cfg = VolConfig(
        calm_thr=0.1,
        storm_thr=0.2,
        use_regime_gate=True,
        block_regime="storm",
        gate_mode="attenuate",
        attenuation_factor=0.4,
    )
    detector = MarketRegimeDetector(vol_config=vol_cfg, sizing_config=sizing)

    prices = _generate_prices(0.04, seed=11)
    result = detector.detect_regime({"close": prices, "periods_per_year": 252})

    assert result.regime.value == "storm"
    expected = sizing.regime_multipliers["storm"] * vol_cfg.attenuation_factor
    assert result.risk_multiplier == pytest.approx(expected)
    assert not result.blocked


def test_detector_blocks_when_gate_mode_block() -> None:
    sizing = SizingConfig(regime_multipliers={"calm": 1.1, "normal": 0.9, "storm": 0.6})
    vol_cfg = VolConfig(
        calm_thr=0.05,
        storm_thr=0.18,
        use_regime_gate=True,
        block_regime="storm",
        gate_mode="block",
    )
    detector = MarketRegimeDetector(vol_config=vol_cfg, sizing_config=sizing)

    prices = _generate_prices(0.05, seed=23)
    result = detector.detect_regime({"close": prices, "periods_per_year": 252})

    assert result.regime.value == "storm"
    assert result.risk_multiplier == pytest.approx(0.0)
    assert result.blocked is True
