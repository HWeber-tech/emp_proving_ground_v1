import copy
import math

import pytest

from src.reflection.trm.model import TRMModel
from src.reflection.trm.types import StrategyEncoding, StrategyStats


_BASE_SPEC = {
    "feature_names": ("a",),
    "heads": {
        "weight_adjust": {
            "weights": {"a": 1.0},
            "bias": 0.0,
            "clip": 1.0,
        },
        "flag": {
            "weights": {"a": 1.0},
            "bias": 0.0,
        },
        "experiment": {
            "weights": {"a": 1.0},
            "bias": 0.0,
        },
        "confidence": {
            "weights": {"a": 1.0},
            "bias": 0.0,
        },
    },
}


def _make_stats() -> StrategyStats:
    return StrategyStats(
        entry_count=1,
        mean_pnl=0.0,
        pnl_std=0.0,
        risk_rate=0.0,
        win_rate=0.0,
        loss_rate=0.0,
        volatility_mean=0.0,
        spread_mean=0.0,
        belief_confidence_mean=0.0,
        pnl_trend=0.0,
        drawdown_ratio=0.0,
    )


def _make_encoding(strategy_id: str, value: float) -> StrategyEncoding:
    features = {"a": value}
    return StrategyEncoding(
        strategy_id=strategy_id,
        features=features,
        stats=_make_stats(),
        audit_entry_hashes=tuple(),
    )


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def test_per_domain_affine_and_temperature() -> None:
    spec = copy.deepcopy(_BASE_SPEC)
    spec["shared_heads"] = {
        "shared_flag": {
            "weights": {"a": 2.0},
            "bias": -1.0,
        }
    }
    spec["strategy_domains"] = {"s1": "fx"}
    spec["domain_heads"] = {
        "fx": {
            "heads": {
                "weight_adjust": {
                    "shared": "weight_adjust",
                    "affine": {"scale": 0.5, "bias": 0.1},
                    "clip": 0.25,
                },
                "flag": {
                    "shared": "shared_flag",
                    "affine": {"scale": 1.0, "bias": 0.5},
                    "temperature": 0.5,
                },
                "confidence": {
                    "shared": "confidence",
                    "temperature": 0.5,
                },
            }
        }
    }

    model = TRMModel(spec, temperature=2.0)
    inference = model.infer(_make_encoding("s1", 1.0))

    assert inference.weight_delta == pytest.approx(0.25, rel=1e-6)
    assert inference.flag_probability == pytest.approx(_sigmoid(3.0), rel=1e-6)
    assert inference.experiment_probability == pytest.approx(_sigmoid(1.0), rel=1e-6)
    assert inference.confidence == pytest.approx(0.99, rel=1e-6)


def test_domain_strategies_map_and_default_fallback() -> None:
    spec = copy.deepcopy(_BASE_SPEC)
    spec["domain_heads"] = {
        "fx": {
            "strategies": ["s2"],
            "flag": {
                "affine": {"scale": 2.0},
            },
        }
    }

    model = TRMModel(spec)
    fx_inference = model.infer(_make_encoding("s2", 1.0))
    baseline = model.infer(_make_encoding("s3", 1.0))

    assert fx_inference.flag_probability == pytest.approx(_sigmoid(2.0), rel=1e-6)
    assert baseline.flag_probability == pytest.approx(_sigmoid(1.0), rel=1e-6)
