from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src.thinking.memory.faiss_memory",
        "src.thinking.learning.real_time_learner",
        "src.thinking.sentient_adaptation_engine",
        "src.sensory.organs.yahoo_finance_organ",
        "src.orchestration.enhanced_intelligence_engine",
    ],
)
def test_removed_thinking_and_sensory_shims_are_absent(module_name: str) -> None:
    """Removed thinking-layer shims should no longer be importable."""

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module(module_name)

    assert module_name in str(excinfo.value)
