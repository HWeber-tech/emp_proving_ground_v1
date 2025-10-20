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


def test_archive_artifact_normalises_kind_segment(tmp_path: Path) -> None:
    source = tmp_path / "metrics.log"
    source.write_text("log", encoding="utf-8")

    root = tmp_path / "artifacts-root"
    timestamp = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)

    destination = archive_artifact("  ::ALERT!!  ", source, root=root, timestamp=timestamp)

    assert destination is not None
    assert destination.relative_to(root).parts[0] == "alert"


def test_archive_artifact_sanitises_target_name(tmp_path: Path) -> None:
    source = tmp_path / "report.txt"
    source.write_text("ok", encoding="utf-8")

    root = tmp_path / "artifacts-root"
    timestamp = datetime(2023, 7, 14, 6, 30, tzinfo=UTC)

    destination = archive_artifact(
        "reports",
        source,
        root=root,
        timestamp=timestamp,
        target_name="../../escape/override.log",
    )

    assert destination is not None
    assert destination.parent == root / "reports" / "2023" / "07" / "14" / "run-063000"
    assert destination.name == "override.log"
    assert destination.read_text(encoding="utf-8") == "ok"
