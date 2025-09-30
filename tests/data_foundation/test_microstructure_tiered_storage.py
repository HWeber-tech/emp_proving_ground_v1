from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.data_foundation.storage import MicrostructureTieredArchive, RetentionPolicy


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    path = tmp_path / "microstructure.parquet"
    path.write_bytes(b"test payload")
    return path


def test_archive_file_writes_metadata_and_copies(sample_file: Path, tmp_path: Path) -> None:
    hot_dir = tmp_path / "hot"
    cold_dir = tmp_path / "cold"
    archive = MicrostructureTieredArchive(
        hot_dir,
        cold_dir,
        retention_policy=RetentionPolicy.from_days(hot_days=1, cold_days=30),
    )

    result = archive.archive_file(sample_file, "ict_microstructure")

    assert result.destination.exists()
    assert result.destination.read_bytes() == sample_file.read_bytes()
    assert result.metadata_path.exists()
    payload = result.metadata_path.read_text(encoding="utf-8")
    assert "ict_microstructure" in payload
    assert "hot" in payload


def test_enforce_retention_moves_hot_to_cold(sample_file: Path, tmp_path: Path) -> None:
    hot_dir = tmp_path / "hot"
    cold_dir = tmp_path / "cold"
    archive = MicrostructureTieredArchive(
        hot_dir,
        cold_dir,
        retention_policy=RetentionPolicy.from_days(hot_days=1, cold_days=30),
    )

    as_of = datetime.utcnow() - timedelta(days=2)
    archive.archive_file(sample_file, "ict_microstructure", as_of=as_of)

    report = archive.enforce_retention(now=datetime.utcnow())

    assert report.moved_to_cold == 1
    assert not any(hot_dir.rglob("*.meta.json"))
    cold_files = list(cold_dir.rglob("*.meta.json"))
    assert len(cold_files) == 1
    cold_payload = cold_files[0].read_text(encoding="utf-8")
    assert '"tier": "cold"' in cold_payload


def test_enforce_retention_deletes_expired_cold(sample_file: Path, tmp_path: Path) -> None:
    hot_dir = tmp_path / "hot"
    cold_dir = tmp_path / "cold"
    archive = MicrostructureTieredArchive(
        hot_dir,
        cold_dir,
        retention_policy=RetentionPolicy.from_days(hot_days=1, cold_days=3),
    )

    as_of = datetime.utcnow() - timedelta(days=2)
    archive.archive_file(sample_file, "ict_microstructure", as_of=as_of)

    initial = archive.enforce_retention(now=datetime.utcnow())
    assert initial.moved_to_cold == 1
    assert initial.deleted_from_cold == 0

    future = datetime.utcnow() + timedelta(days=5)
    report = archive.enforce_retention(now=future)

    assert report.deleted_from_cold == 1
    assert not any(cold_dir.rglob("*.parquet"))

