"""Tests for runtime determinism helper utilities."""

from src.runtime.determinism import resolve_seed


def test_resolve_seed_prefers_primary_key() -> None:
    seed, invalid = resolve_seed({"RNG_SEED": "123", "BOOTSTRAP_RANDOM_SEED": "456"})

    assert seed == 123
    assert invalid == []


def test_resolve_seed_falls_back_to_secondary_key() -> None:
    seed, invalid = resolve_seed({"RNG_SEED": "oops", "BOOTSTRAP_RANDOM_SEED": "789"})

    assert seed == 789
    assert invalid == [("RNG_SEED", "oops")]


def test_resolve_seed_handles_missing_values() -> None:
    seed, invalid = resolve_seed({})

    assert seed is None
    assert invalid == []
