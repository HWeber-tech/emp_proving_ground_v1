from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_foundation.config import (
    execution_config,
    risk_portfolio_config,
    sizing_config,
    vol_config,
    why_config,
)
from src.sensory.what.volatility_engine import VolConfig


class _JSONLoader:
    """Simple stand-in for `yaml.safe_load` that reads JSON/YAML payloads."""

    def safe_load(self, stream: object) -> object:  # pragma: no cover - exercised via tests
        text = stream.read()
        if isinstance(text, bytes):  # defensive – mirrors PyYAML behaviour
            text = text.decode("utf-8")
        return json.loads(text) if text.strip() else {}


@pytest.fixture()
def json_loader() -> _JSONLoader:
    return _JSONLoader()


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload))


def test_load_execution_config_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_loader: _JSONLoader
) -> None:
    config_path = tmp_path / "execution.yaml"
    _write_yaml(
        config_path,
        {
            "execution": {
                "slippage": {
                    "base_bps": 0.35,
                    "spread_coef": 42.0,
                    "imbalance_coef": 1.5,
                    "sigma_coef": 44.0,
                    "size_coef": 7.5,
                },
                "fees": {"commission_bps": 0.22},
            }
        },
    )

    monkeypatch.setenv("EXECUTION_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(execution_config, "yaml", json_loader, raising=True)

    cfg = execution_config.load_execution_config()

    assert cfg.slippage.base_bps == pytest.approx(0.35)
    assert cfg.slippage.spread_coef == pytest.approx(42.0)
    assert cfg.slippage.imbalance_coef == pytest.approx(1.5)
    assert cfg.slippage.sigma_coef == pytest.approx(44.0)
    assert cfg.slippage.size_coef == pytest.approx(7.5)
    assert cfg.fees.commission_bps == pytest.approx(0.22)


def test_load_execution_config_missing_file_returns_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_loader: _JSONLoader
) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"
    monkeypatch.setenv("EXECUTION_CONFIG_PATH", str(missing_path))
    monkeypatch.setattr(execution_config, "yaml", json_loader, raising=True)

    cfg = execution_config.load_execution_config()

    assert cfg == execution_config.ExecutionConfig()


def test_load_portfolio_risk_config_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_loader: _JSONLoader
) -> None:
    config_path = tmp_path / "portfolio.yaml"
    _write_yaml(
        config_path,
        {
            "portfolio_risk": {
                "per_asset_cap": 0.75,
                "aggregate_cap": 1.8,
                "usd_beta_cap": 1.1,
                "var95_cap": 0.015,
            }
        },
    )

    monkeypatch.setenv("RISK_PORTFOLIO_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(risk_portfolio_config, "_yaml", json_loader, raising=True)

    cfg = risk_portfolio_config.load_portfolio_risk_config()

    assert cfg.per_asset_cap == pytest.approx(0.75)
    assert cfg.aggregate_cap == pytest.approx(1.8)
    assert cfg.usd_beta_cap == pytest.approx(1.1)
    assert cfg.var95_cap == pytest.approx(0.015)


def test_load_portfolio_risk_config_missing_file_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_loader: _JSONLoader
) -> None:
    missing_path = tmp_path / "missing.yaml"
    monkeypatch.setenv("RISK_PORTFOLIO_CONFIG_PATH", str(missing_path))
    monkeypatch.setattr(risk_portfolio_config, "_yaml", json_loader, raising=True)

    cfg = risk_portfolio_config.load_portfolio_risk_config()

    assert cfg == risk_portfolio_config.PortfolioRiskConfig()


def test_load_sizing_config_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_loader: _JSONLoader
) -> None:
    config_path = tmp_path / "sizing.yaml"
    _write_yaml(
        config_path,
        {
            "sizing": {
                "k_exposure": 0.9,
                "sigma_floor": 0.1,
                "sigma_ceiling": 0.4,
                "regime_multipliers": {"calm": 1.2, "normal": 0.9, "storm": 0.4},
            }
        },
    )

    monkeypatch.setenv("SIZING_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(sizing_config, "_yaml_mod", json_loader, raising=True)

    cfg = sizing_config.load_sizing_config()

    assert cfg.k_exposure == pytest.approx(0.9)
    assert cfg.sigma_floor == pytest.approx(0.1)
    assert cfg.sigma_ceiling == pytest.approx(0.4)
    assert cfg.regime_multipliers == {"calm": 1.2, "normal": 0.9, "storm": 0.4}


def test_load_sizing_config_defaults_when_yaml_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_loader: _JSONLoader
) -> None:
    missing_path = tmp_path / "nope.yaml"
    monkeypatch.setenv("SIZING_CONFIG_PATH", str(missing_path))
    monkeypatch.setattr(sizing_config, "_yaml_mod", json_loader, raising=True)

    cfg = sizing_config.load_sizing_config()

    assert cfg == sizing_config.SizingConfig(
        regime_multipliers={"calm": 1.0, "normal": 0.8, "storm": 0.5}
    )


def test_load_vol_config_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_loader: _JSONLoader
) -> None:
    config_path = tmp_path / "vol.yaml"
    _write_yaml(
        config_path,
        {
            "vol_engine": {
                "bar_interval": "15m",
                "daily_fit_lookback": "750d",
                "rv_window": "2h",
                "blend_weight": 0.55,
                "regime_thresholds": {"calm": 0.1, "storm": 0.25},
                "sizing": {"risk_budget_per_trade": 0.004, "k_stop": 1.6},
                "var": {"confidence": 0.99},
                "fallbacks": {"ewma_lambda": 0.88},
                "regime_gate": {
                    "enabled": True,
                    "block": "calm",
                    "mode": "attenuate",
                    "attenuation_factor": 0.45,
                },
                "risk_controls": {"brake_scale": 0.5},
            }
        },
    )

    monkeypatch.setenv("VOL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(vol_config, "yaml", json_loader, raising=True)

    cfg = vol_config.load_vol_config()

    assert isinstance(cfg, VolConfig)
    assert cfg.bar_interval_minutes == 15
    assert cfg.daily_fit_lookback_days == 750
    assert cfg.rv_window_minutes == 120
    assert cfg.blend_weight == pytest.approx(0.55)
    assert cfg.calm_thr == pytest.approx(0.1)
    assert cfg.storm_thr == pytest.approx(0.25)
    assert cfg.risk_budget_per_trade == pytest.approx(0.004)
    assert cfg.k_stop == pytest.approx(1.6)
    assert cfg.var_confidence == pytest.approx(0.99)
    assert cfg.ewma_lambda == pytest.approx(0.88)
    assert cfg.use_regime_gate is True
    assert cfg.block_regime == "calm"
    assert cfg.gate_mode == "attenuate"
    assert cfg.attenuation_factor == pytest.approx(0.45)
    assert cfg.brake_scale == pytest.approx(0.5)


def test_load_why_config_from_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_loader: _JSONLoader
) -> None:
    config_path = tmp_path / "why.yaml"
    _write_yaml(
        config_path,
        {
            "why_engine": {
                "enable_macro_proximity": False,
                "enable_yields": True,
                "weights": {"macro": 0.4, "yields": 0.6},
                "yield_features": {
                    "slope_2s10s": False,
                    "slope_5s30s": True,
                    "curvature_2_10_30": False,
                    "parallel_shift": True,
                },
            }
        },
    )

    monkeypatch.setenv("WHY_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(why_config, "_yaml", json_loader, raising=True)

    cfg = why_config.load_why_config()

    assert cfg.enable_macro_proximity is False
    assert cfg.enable_yields is True
    assert cfg.weight_macro == pytest.approx(0.4)
    assert cfg.weight_yields == pytest.approx(0.6)
    assert cfg.use_slope_2s10s is False
    assert cfg.use_slope_5s30s is True
    assert cfg.use_curvature_2_10_30 is False
    assert cfg.use_parallel_shift is True


def test_load_why_config_defaults_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_loader: _JSONLoader
) -> None:
    missing_path = tmp_path / "absent.yaml"
    monkeypatch.setenv("WHY_CONFIG_PATH", str(missing_path))
    monkeypatch.setattr(why_config, "_yaml", json_loader, raising=True)

    cfg = why_config.load_why_config()

    assert cfg == why_config.WhyConfig()
