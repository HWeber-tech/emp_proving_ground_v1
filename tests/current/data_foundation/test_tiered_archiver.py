from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.data_foundation.persist import DatasetPolicy, TieredDatasetArchiver


def _write_file(path: Path, *, contents: str, modified_days_ago: int, now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    timestamp = (now - timedelta(days=modified_days_ago)).timestamp()
    os.utime(path, (timestamp, timestamp))


def test_tiered_archiver_sorts_files_into_hot_cold_and_expired(tmp_path: Path) -> None:
    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    source = tmp_path / "source"
    hot = tmp_path / "hot"
    cold = tmp_path / "cold"
    metadata = tmp_path / "metadata"

    _write_file(source / "fresh.parquet", contents="fresh", modified_days_ago=1, now=now)
    _write_file(source / "stale/older.parquet", contents="stale", modified_days_ago=10, now=now)
    _write_file(source / "ancient.jsonl", contents="ancient", modified_days_ago=40, now=now)

    archiver = TieredDatasetArchiver(metadata_dir=metadata, clock=lambda: now)
    policy = DatasetPolicy(hot_retention_days=7, cold_retention_days=30)

    result = archiver.archive_dataset(
        dataset="microstructure",
        source_dir=source,
        hot_dir=hot,
        cold_dir=cold,
        policy=policy,
    )

    assert sorted(result.hot) == ["fresh.parquet"]
    assert sorted(result.cold) == ["stale/older.parquet"]
    assert sorted(result.expired) == ["ancient.jsonl"]
    assert not result.missing_source

    assert (hot / "fresh.parquet").exists()
    assert (cold / "stale/older.parquet").exists()
    assert not (hot / "ancient.jsonl").exists()
    assert not (cold / "ancient.jsonl").exists()

    metadata_file = metadata / "microstructure.json"
    assert metadata_file.exists()
    payload = json.loads(metadata_file.read_text())
    assert payload["hot"] == ["fresh.parquet"]
    assert payload["cold"] == ["stale/older.parquet"]
    assert payload["expired"] == ["ancient.jsonl"]


def test_tiered_archiver_handles_missing_source_directory(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    archiver = TieredDatasetArchiver(metadata_dir=metadata)
    policy = DatasetPolicy(hot_retention_days=3)

    result = archiver.archive_dataset(
        dataset="missing",
        source_dir=tmp_path / "missing",  # directory does not exist
        hot_dir=tmp_path / "hot",
        policy=policy,
    )

    assert result.missing_source
    assert result.hot == []
    assert result.cold == []
    assert (metadata / "missing.json").exists()
