"""Tests for coercion helpers."""

from src.core.coercion import coerce_float, coerce_int


def test_coerce_float_accepts_bytes() -> None:
    assert coerce_float(b" 3.14 ") == 3.14


def test_coerce_int_accepts_bytearray() -> None:
    assert coerce_int(bytearray(b"42\n")) == 42


def test_coerce_helpers_accept_numeric_separators() -> None:
    assert coerce_float(" 1_234.5 ") == 1234.5
    assert coerce_int(" -2_000 ") == -2000
