import pytest

from src.genome.models.adapters import from_legacy, to_legacy_view
from src.genome.models.genome import new_genome


def test_ecosystem_optimizer_smoke_import_and_adapt():
    # Import-or-skip the optimizer module
    mod = pytest.importorskip("src.ecosystem.optimization.ecosystem_optimizer")

    # Create a canonical genome and adapt to a legacy view, then back again
    g = new_genome("E1", {"alpha": 0.5}, generation=0, species_type="stalker")
    legacy = to_legacy_view(g)
    g2 = from_legacy(legacy)

    # Round-trip semantics preserved for core fields
    assert g2.id == g.id
    assert g2.parameters == g.parameters
    assert g2.species_type == g.species_type
    assert g2.generation == g.generation

    # Smoke: ensure optimizer can be instantiated (no deep coupling here)
    optimizer = mod.EcosystemOptimizer()
    assert hasattr(optimizer, "optimize_ecosystem")
