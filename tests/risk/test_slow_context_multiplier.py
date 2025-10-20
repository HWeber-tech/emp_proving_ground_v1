from __future__ import annotations

import pytest

from src.risk.slow_context import resolve_size_multiplier


def test_defaults_to_full_size_multiplier() -> None:
    decision = resolve_size_multiplier()

    assert decision.multiplier == pytest.approx(1.0)
    assert decision.drivers == {"macro": False, "volatility": False, "earnings": False}


def test_macro_block_forces_zero_multiplier() -> None:
    decision = resolve_size_multiplier({"macro_block": True})

    assert decision.multiplier == pytest.approx(0.0)
    assert decision.drivers["macro"] is True


def test_volatility_or_earnings_trigger_throttle() -> None:
    decision = resolve_size_multiplier({"volatility_throttle": 1})

    assert decision.multiplier == pytest.approx(0.3)
    assert decision.drivers["volatility"] is True

    earnings_decision = resolve_size_multiplier({"earnings_blackout": "yes"})

    assert earnings_decision.multiplier == pytest.approx(0.3)
    assert earnings_decision.drivers["earnings"] is True


def test_override_respected_when_allowed_value() -> None:
    decision = resolve_size_multiplier({"size_multiplier": "0.3", "macro_block": True})

    assert decision.multiplier == pytest.approx(0.3)
    assert decision.drivers["macro"] is True


def test_invalid_override_falls_back_to_resolution() -> None:
    decision = resolve_size_multiplier({"size_multiplier": 0.42, "volatility_throttle": True})

    assert decision.multiplier == pytest.approx(0.3)
    assert decision.drivers["volatility"] is True
