from src.system.requirements_check import _parse


def test_parse_handles_prerelease_tags() -> None:
    assert _parse("1.26.0rc1") == (1, 26, 0)
