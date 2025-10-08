from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "expected_fragment"),
    [
        (
            "src.thinking.memory.faiss_memory",
            "src.sentient.memory.faiss_pattern_memory",
        ),
        (
            "src.thinking.learning.real_time_learner",
            "src.sentient.learning.real_time_learning_engine",
        ),
        (
            "src.thinking.sentient_adaptation_engine",
            "src.intelligence.sentient_adaptation",
        ),
    ],
)
def test_thinking_shims_raise_module_not_found(
    module_name: str, expected_fragment: str
) -> None:
    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module(module_name)

    message = str(excinfo.value)
    assert module_name in message
    assert expected_fragment in message

