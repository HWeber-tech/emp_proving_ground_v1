from __future__ import annotations

from datetime import datetime, timezone
try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    UTC = timezone.utc  # type: ignore[assignment]
from pathlib import Path

from src.artifacts import archive_artifact


def test_archive_artifact_copies_into_dated_directory(tmp_path: Path) -> None:
    source = tmp_path / "diary.json"
    source.write_text("{}\n", encoding="utf-8")

    root = tmp_path / "artifacts-root"
    timestamp = datetime(2025, 9, 28, 19, 1, 38, tzinfo=UTC)

    destination = archive_artifact(
        "Diaries",
        source,
        root=root,
        timestamp=timestamp,
        run_id="nightly:replay",
        target_name="decision_diary.json",
    )

    assert destination is not None
    expected_dir = root / "diaries" / "2025" / "09" / "28" / "nightly-replay"
    assert destination.parent == expected_dir
    assert destination.name == "decision_diary.json"
    assert destination.read_text(encoding="utf-8") == "{}\n"


def test_archive_artifact_returns_none_for_missing_source(tmp_path: Path) -> None:
    root = tmp_path / "artifacts-root"
    result = archive_artifact(
        "drift_reports",
        tmp_path / "missing.json",
        root=root,
        timestamp=datetime.now(tz=UTC),
    )
    assert result is None
    assert not (root / "drift_reports").exists()
