"""Security-oriented regression tests for mock FIX coercion helpers."""

from __future__ import annotations

import pytest

from src.operational.mock_fix import (
    MockOrderBookLevel,
    _coerce_order_book_level,
    _coerce_positive_int,
)


class _ExplodingLevel:
    """Order book stub that raises while exposing price/size attributes."""

    def __init__(self) -> None:
        self._exc = ValueError("boom")

    @property
    def price(self) -> float:
        raise self._exc

    @property
    def size(self) -> float:
        return 1.0


def test_coerce_positive_int_rejects_non_ascii_bytes() -> None:
    """Non-ASCII payloads are rejected instead of propagating decode errors."""

    assert _coerce_positive_int(b"\xff\xfe") is None


def test_coerce_positive_int_accepts_ascii_bytes() -> None:
    """ASCII byte payloads still coerce to integers."""

    assert _coerce_positive_int(b"42") == 42


def test_coerce_order_book_level_handles_value_errors(caplog: pytest.LogCaptureFixture) -> None:
    """Order book coercion should log and fall back when attributes explode."""

    caplog.set_level("DEBUG")
    level = _coerce_order_book_level(_ExplodingLevel())
    assert level is None
    assert any("Failed to coerce order book" in message for message in caplog.messages)


def test_coerce_order_book_level_sequence_passthrough() -> None:
    """Valid sequence payloads still produce mock order book levels."""

    level = _coerce_order_book_level(("1", "2"))
    assert isinstance(level, MockOrderBookLevel)
    assert level.price == pytest.approx(1.0)
    assert level.size == pytest.approx(2.0)

