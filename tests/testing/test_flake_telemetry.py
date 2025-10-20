from __future__ import annotations

from pathlib import Path


from src.testing.flake_telemetry import resolve_output_path


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
