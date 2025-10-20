"""Tests for runtime determinism helper utilities."""

import os
import random

from src.runtime.determinism import resolve_seed, seed_runtime


def test_resolve_seed_prefers_primary_key() -> None:
    seed, invalid = resolve_seed({"RNG_SEED": "123", "BOOTSTRAP_RANDOM_SEED": "456"})

    assert seed == 123
    assert invalid == []


def test_resolve_seed_falls_back_to_secondary_key() -> None:
    seed, invalid = resolve_seed({"RNG_SEED": "oops", "BOOTSTRAP_RANDOM_SEED": "789"})

    assert seed == 789
    assert invalid == [("RNG_SEED", "oops")]


def test_resolve_seed_flags_blank_strings() -> None:
    seed, invalid = resolve_seed({"RNG_SEED": "  ", "BOOTSTRAP_RANDOM_SEED": "11"})

    assert seed == 11
    assert invalid == [("RNG_SEED", "  ")]


def test_resolve_seed_accepts_integral_float_values() -> None:
    seed, invalid = resolve_seed({"RNG_SEED": 17.0})

    assert seed == 17
    assert invalid == []


def test_resolve_seed_handles_missing_values() -> None:
    seed, invalid = resolve_seed({})

    assert seed is None
    assert invalid == []


def test_resolve_seed_accepts_integral_float_strings() -> None:
    seed, invalid = resolve_seed({"RNG_SEED": "17.0"})

    assert seed == 17
    assert invalid == []


def test_seed_runtime_overrides_pythonhashseed(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "999")
    original_state = random.getstate()

    try:
        seed_runtime(7)
        assert os.environ["PYTHONHASHSEED"] == "7"
        assert random.getstate() == random.Random(7).getstate()
    finally:
        random.setstate(original_state)
