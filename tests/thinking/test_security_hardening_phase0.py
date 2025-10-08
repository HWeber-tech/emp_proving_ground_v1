import logging

import pytest

from src.thinking.adversarial.red_team_ai import _to_mapping as red_to_mapping
from src.thinking.competitive.competitive_intelligence_system import _to_mapping as ci_to_mapping


class _DictFailure:
    def __init__(self) -> None:
        self.strategy_id = "strat-1"
        self.timestamp = "now"

    def dict(self) -> dict[str, object]:
        raise ValueError("dict explosion")


class _AttributeFailure:
    def __init__(self) -> None:
        self.algorithm_type = "alpha"

    def dict(self) -> object:
        return ["not", "mapping"]

    @property
    def frequency(self) -> str:
        raise RuntimeError("frequency unavailable")


def test_red_team_to_mapping_logs_on_dict_failure(caplog: pytest.LogCaptureFixture) -> None:
    payload = _DictFailure()

    with caplog.at_level(logging.WARNING):
        result = red_to_mapping(payload)

    assert result["strategy_id"] == "strat-1"
    assert "dict explosion" in caplog.text


def test_competitive_to_mapping_handles_attribute_failures(caplog: pytest.LogCaptureFixture) -> None:
    payload = _AttributeFailure()

    with caplog.at_level(logging.DEBUG):
        result = ci_to_mapping(payload, keys=["algorithm_type", "frequency"])

    assert result["algorithm_type"] == "alpha"
    assert "frequency" not in result
    assert "not mapping" in caplog.text or "frequency unavailable" in caplog.text
