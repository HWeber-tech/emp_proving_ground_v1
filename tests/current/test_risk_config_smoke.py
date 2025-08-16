import importlib
import io


import pytest


def test_config_defaults_without_fs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force "no file" path for all loaders
    monkeypatch.setattr("os.path.exists", lambda _p: False)

    ec = importlib.import_module("src.data_foundation.config.execution_config")
    rc = importlib.import_module("src.data_foundation.config.risk_portfolio_config")
    sc = importlib.import_module("src.data_foundation.config.sizing_config")
    vc = importlib.import_module("src.data_foundation.config.vol_config")
    wc = importlib.import_module("src.data_foundation.config.why_config")

    # Execution defaults
    e = ec.load_execution_config("dummy.yaml")
    assert e.slippage.base_bps == pytest.approx(0.2)
    assert e.fees.commission_bps == pytest.approx(0.1)

    # Portfolio risk defaults
    r = rc.load_portfolio_risk_config("dummy.yaml")
    assert r.per_asset_cap == pytest.approx(1.0)
    assert r.aggregate_cap == pytest.approx(2.0)
    assert r.usd_beta_cap == pytest.approx(1.5)
    assert r.var95_cap == pytest.approx(0.02)

    # Sizing defaults include regime multipliers
    s = sc.load_sizing_config("dummy.yaml")
    assert isinstance(s.regime_multipliers, dict)
    for k in ("calm", "normal", "storm"):
        assert k in s.regime_multipliers

    # Vol defaults (fallback dataclass if legacy import not available)
    v = vc.load_vol_config("dummy.yaml")
    assert getattr(v, "bar_interval_minutes", 5) >= 1

    # WHY defaults
    w = wc.load_why_config("dummy.yaml")
    assert w.enable_macro_proximity is True
    assert w.enable_yields is True
    assert w.weight_macro == pytest.approx(0.5)
    assert w.weight_yields == pytest.approx(0.5)


def test_execution_and_vol_config_inmemory_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare modules
    ec = importlib.import_module("src.data_foundation.config.execution_config")
    vc = importlib.import_module("src.data_foundation.config.vol_config")

    # Make the paths appear present
    monkeypatch.setattr("os.path.exists", lambda p: p in {"EXEC_PATH.yaml", "VOL_PATH.yaml"})

    # Stub open to return a harmless file-like (content unused by our yaml stubs)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO("ignored"))

    # Stub yaml.safe_load on each module to return structured dicts
    class DummyYamlExec:
        @staticmethod
        def safe_load(_fh):
            return {
                "execution": {
                    "slippage": {
                        "base_bps": 0.5,
                        "spread_coef": 12,
                        "imbalance_coef": 3.0,
                        "sigma_coef": 77.0,
                        "size_coef": 6.0,
                    },
                    "fees": {"commission_bps": 0.2},
                }
            }

    class DummyYamlVol:
        @staticmethod
        def safe_load(_fh):
            return {
                "vol_engine": {
                    "bar_interval": "5m",
                    "daily_fit_lookback": "250d",
                    "rv_window": "60m",
                    "blend_weight": 0.65,
                    "regime_thresholds": {"calm": 0.07, "storm": 0.2},
                    "sizing": {"risk_budget_per_trade": 0.004, "k_stop": 1.1},
                    "var": {"confidence": 0.97},
                    "fallbacks": {"ewma_lambda": 0.93},
                    "regime_gate": {"enabled": True, "block": "storm", "mode": "block", "attenuation_factor": 0.25},
                    "risk_controls": {"brake_scale": 0.6},
                }
            }

    monkeypatch.setattr(ec, "yaml", DummyYamlExec)  # type: ignore[attr-defined]
    monkeypatch.setattr(vc, "yaml", DummyYamlVol)  # type: ignore[attr-defined]

    # Execute loaders with our dummy paths
    e = ec.load_execution_config("EXEC_PATH.yaml")
    assert e.slippage.base_bps == pytest.approx(0.5)
    assert e.slippage.spread_coef == pytest.approx(12.0)
    assert e.fees.commission_bps == pytest.approx(0.2)

    v = vc.load_vol_config("VOL_PATH.yaml")
    # Confirm parsing helpers applied
    assert v.bar_interval_minutes == 5
    assert v.daily_fit_lookback_days == 250
    assert v.rv_window_minutes == 60
    assert v.blend_weight == pytest.approx(0.65)
    assert v.calm_thr == pytest.approx(0.07)
    assert v.storm_thr == pytest.approx(0.2)
    assert v.risk_budget_per_trade == pytest.approx(0.004)
    assert v.k_stop == pytest.approx(1.1)
    assert v.var_confidence == pytest.approx(0.97)
    assert v.ewma_lambda == pytest.approx(0.93)
    assert v.use_regime_gate is True
    assert v.block_regime == "storm"
    assert v.gate_mode == "block"
    assert v.attenuation_factor == pytest.approx(0.25)
    assert v.brake_scale == pytest.approx(0.6)