from __future__ import annotations

import math

import pytest

from src.thinking.learning.lora_freeze import (
    LoRAFreezePlan,
    LoRALayerConfig,
    plan_lora_freeze,
)


def _make_layers(count: int) -> list[str]:
    return [f"layer_{index}" for index in range(count)]


def test_plan_respects_ranges_and_ordering() -> None:
    layers = _make_layers(20)
    plan = plan_lora_freeze(layers)

    assert isinstance(plan, LoRAFreezePlan)
    freeze_count = len(plan.frozen_layers)
    lora_count = len(plan.lora_layers)

    assert math.isclose(plan.freeze_fraction, freeze_count / len(layers))
    assert math.isclose(plan.lora_fraction, lora_count / len(layers))

    assert 12 <= freeze_count <= 16  # 60-80% of 20
    assert 6 <= lora_count <= 8  # 30-40% of 20

    assert plan.frozen_layers == tuple(layers[:freeze_count])
    assert {config.layer for config in plan.lora_layers}.isdisjoint(plan.frozen_layers)

    top_layers = layers[-lora_count:] if lora_count else []
    assert [config.layer for config in plan.lora_layers] == top_layers
    for config in plan.lora_layers:
        assert isinstance(config, LoRALayerConfig)
        assert 8 <= config.rank <= 16
        assert config.alpha == pytest.approx(config.rank * 2.0)
        assert config.dropout == pytest.approx(0.05)


def test_plan_handles_small_models() -> None:
    layers = ["embedding", "encoder", "decoder", "head"]
    plan = plan_lora_freeze(layers)

    assert len(plan.frozen_layers) >= 2  # best-effort towards 60%
    assert len(plan.lora_layers) >= 1
    assert plan.lora_layers[-1].layer == "head"
    assert {config.layer for config in plan.lora_layers}.isdisjoint(plan.frozen_layers)


def test_plan_supports_custom_targets() -> None:
    layers = _make_layers(10)
    plan = plan_lora_freeze(
        layers,
        freeze_fraction=0.6,
        lora_fraction=0.4,
        rank_range=(10, 12),
        lora_alpha_multiplier=1.5,
        lora_dropout=0.1,
    )

    freeze_count = len(plan.frozen_layers)
    lora_count = len(plan.lora_layers)

    assert 6 <= freeze_count <= 8
    assert 3 <= lora_count <= 4
    for config in plan.lora_layers:
        assert 10 <= config.rank <= 12
        assert config.alpha == pytest.approx(config.rank * 1.5)
        assert config.dropout == pytest.approx(0.1)


def test_plan_as_dict_roundtrip() -> None:
    layers = _make_layers(6)
    plan = plan_lora_freeze(layers)
    payload = plan.as_dict()

    assert payload["total_layers"] == 6
    assert isinstance(payload["frozen_layers"], list)
    assert isinstance(payload["lora_layers"], list)
    for config in payload["lora_layers"]:
        assert set(config) == {"layer", "rank", "alpha", "dropout"}


def test_plan_validates_input() -> None:
    with pytest.raises(ValueError):
        plan_lora_freeze([])

    with pytest.raises(ValueError):
        plan_lora_freeze(["a"], freeze_range=(0.9, 0.2))

    with pytest.raises(ValueError):
        plan_lora_freeze(["a"], lora_range=(-0.1, 0.5))

    with pytest.raises(ValueError):
        plan_lora_freeze(["a"], rank_range=(0, 4))

    with pytest.raises(ValueError):
        plan_lora_freeze(["a"], lora_alpha_multiplier=0)

    with pytest.raises(ValueError):
        plan_lora_freeze(["a"], lora_dropout=1.0)
