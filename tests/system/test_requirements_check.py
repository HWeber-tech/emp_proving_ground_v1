from src.system.requirements_check import _parse


def test_parse_handles_prerelease_tags() -> None:
    assert _parse("1.26.0rc1") == (1, 26, 0)


def test_parse_handles_non_string_inputs() -> None:
    assert _parse(None) == (0, 0, 0)
    assert _parse(b"2.1.3") == (2, 1, 3)
