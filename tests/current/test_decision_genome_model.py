import time
import types

import pytest

from src.genome.models.adapters import from_legacy
from src.genome.models.genome import DecisionGenome, mutate, new_genome


def test_dataclass_roundtrip():
    data = {
        "id": "g1",
        "parameters": {"a": "1.2", "b": "x", "c": 3},
        "fitness": "4.5",
        "generation": "-2",
        "species_type": 123,
        "parent_ids": ("p1", 2),
        "mutation_history": [
            {"generation": 2, "parameter": "a", "old_value": 1, "new_value": 2},
            "g3:x:1->2",
        ],
        "performance_metrics": {"ret": "0.1", "bad": "x"},
        "created_at": "1700000000.5",
    }

    g = DecisionGenome.from_dict(data)
    assert g.id == "g1"
    assert g.generation == 0
    assert g.parameters == {"a": 1.2, "c": 3.0}
    assert "b" not in g.parameters
    assert g.performance_metrics == {"ret": 0.1}
    assert g.parent_ids == ["p1", "2"]
    assert "g2:a:1->2" in g.mutation_history
    assert "g3:x:1->2" in g.mutation_history
    assert isinstance(g.created_at, float)

    d = g.to_dict()
    g2 = DecisionGenome.from_dict(d)
    assert g2.to_dict() == g.to_dict()


def test_with_updated_returns_new_instance():
    g1 = new_genome("id-1", {"a": 1.0, "b": 2.0}, generation=1, species_type="s")
    g2 = g1.with_updated(
        parameters={"a": 5.0, "b": 2.0},
        parent_ids=["X"],
        performance_metrics={"score": 7.0},
    )

    assert g2 is not g1
    assert g1.parameters["a"] == 1.0
    assert g2.parameters["a"] == 5.0

    # Deep copy of mutables
    g2.parameters["a"] = 9.0
    g2.parent_ids.append("Y")
    g2.performance_metrics["score"] = 8.0

    assert g1.parameters["a"] == 1.0
    assert g1.parent_ids == []
    assert g1.performance_metrics == {}


def test_builder_mutate():
    g0 = new_genome("id-2", {"a": 1.0, "b": 2.0}, generation=0, species_type="s")
    g1 = mutate(g0, "delta", {"a": 1.1, "b": 2.0, "c": 3})

    # Original unchanged
    assert g0.parameters["a"] == 1.0
    assert "c" not in g0.parameters

    # Updated
    assert g1.parameters["a"] == pytest.approx(1.1)
    assert g1.parameters["c"] == pytest.approx(3.0)

    # Tags appended
    assert any(tag == "g0:mutation:delta" for tag in g1.mutation_history)
    assert any(tag.startswith("g0:a:1.0->") for tag in g1.mutation_history)
    assert any("g0:c:None->3.0" == tag for tag in g1.mutation_history)


def test_adapters_from_legacy_variants():
    # dict variant
    legacy_dict = {
        "id": "L1",
        "parameters": {"x": "2.5", "y": "u"},
        "generation": -1,
        "mutation_history": [{"generation": 1, "parameter": "x", "old_value": 1, "new_value": 2}],
        "performance_metrics": {"score": "0.3"},
    }
    g1 = from_legacy(legacy_dict)
    assert g1.id == "L1"
    assert g1.generation == 0
    assert g1.parameters == {"x": 2.5}
    assert "g1:x:1->2" in g1.mutation_history
    assert g1.performance_metrics == {"score": 0.3}

    # object-like variant using SimpleNamespace
    o = types.SimpleNamespace(
        id="L2",
        parameters={"a": 1, "b": "bad"},
        generation=3,
        mutation_history=["g2:a:3->4"],
        performance_metrics={"r": "0.2"},
        created_at=time.time(),
    )

    g2 = from_legacy(o)
    assert g2.id == "L2"
    assert g2.generation == 3
    assert g2.parameters == {"a": 1.0}
    assert g2.mutation_history == ["g2:a:3->4"]
    assert g2.performance_metrics == {"r": 0.2}
