"""Tests for coercion helpers."""

from src.core.coercion import coerce_float, coerce_int


def test_coerce_float_accepts_bytes() -> None:
    assert coerce_float(b" 3.14 ") == 3.14


def test_coerce_int_accepts_bytearray() -> None:
    assert coerce_int(bytearray(b"42\n")) == 42


def test_coerce_helpers_accept_numeric_separators() -> None:
    assert coerce_float(" 1_234.5 ") == 1234.5
    assert coerce_int(" -2_000 ") == -2000


def test_coerce_helpers_accept_thousand_separators() -> None:
    assert coerce_float("1,234.5") == 1234.5
    assert coerce_int("1,234") == 1234


def test_coerce_float_rejects_non_finite_strings() -> None:
    assert coerce_float("nan") is None
    assert coerce_float("inf", default=7.0) == 7.0


def test_coerce_int_handles_prefixed_bases() -> None:
    assert coerce_int("0x10") == 16


def test_coerce_helpers_accept_grouped_numeric_strings() -> None:
    assert coerce_float("1,234.5") == 1234.5
    assert coerce_int("-2,500") == -2500


def test_coerce_helpers_accept_space_grouped_numbers() -> None:
    assert coerce_float("1 234.5") == 1234.5
    assert coerce_int("7 654") == 7654
