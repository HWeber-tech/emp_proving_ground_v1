from __future__ import annotations

from pathlib import Path


from src.testing.flake_telemetry import (
    DEFAULT_RELATIVE_PATH,
    MAX_LONGREPR_LENGTH,
    clip_longrepr,
    resolve_output_path,
)


def test_resolve_output_path_expands_environment_variables(tmp_path, monkeypatch):
    target_dir = tmp_path / "telemetry"
    monkeypatch.setenv("FLAKE_TELEMETRY_DIR", str(target_dir))

    path = resolve_output_path(
        tmp_path,
        explicit="${FLAKE_TELEMETRY_DIR}/session.json",
        ini_value=None,
        env_value=None,
    )

    assert path == target_dir / "session.json"
    assert path.parent == target_dir
    assert target_dir.is_dir()


def test_resolve_output_path_ignores_blank_candidates(tmp_path):
    path = resolve_output_path(
        tmp_path,
        explicit="  ",
        ini_value="",
        env_value=" \t ",
    )

    expected = tmp_path / DEFAULT_RELATIVE_PATH
    assert path == expected
    assert expected.parent.is_dir()


def test_resolve_output_path_accepts_pathlike_values(tmp_path):
    explicit_path = Path("relative") / "session.json"
    resolved = resolve_output_path(
        tmp_path,
        explicit=explicit_path,
        ini_value=None,
        env_value=None,
    )

    assert resolved == tmp_path / explicit_path
    assert resolved.parent.is_dir()


def test_clip_longrepr_without_limit_returns_full_text() -> None:
    text = "x" * (MAX_LONGREPR_LENGTH + 5)
    assert clip_longrepr(text, limit=None) == text


def test_clip_longrepr_respects_limit_with_suffix() -> None:
    text = "x" * (MAX_LONGREPR_LENGTH + 50)
    limit = 120

    clipped = clip_longrepr(text, limit=limit)

    assert len(clipped) <= limit
    assert clipped.endswith("chars]")
    assert clipped.startswith("x")
    assert "… [truncated " in clipped


def test_clip_longrepr_retains_prefix_when_suffix_exceeds_limit() -> None:
    clipped = clip_longrepr("abcdef", limit=2)

    assert clipped.startswith("ab")
    assert clipped.endswith("… [truncated 4 chars]")


def test_resolve_output_path_supports_pathlike(tmp_path: Path) -> None:
    explicit_path = tmp_path / "telemetry" / "session.json"

    path = resolve_output_path(
        tmp_path,
        explicit=explicit_path,
        ini_value=None,
        env_value=None,
    )

    assert path == explicit_path
    assert path.parent.is_dir()
