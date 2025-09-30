from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import json

from src.data_foundation.microstructure import (
    MicrostructureArchive,
    MicrostructureArchiveConfig,
    build_retention_guidance,
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_archive_promotes_and_enforces_retention(tmp_path: Path) -> None:
    config = MicrostructureArchiveConfig(
        hot_path=tmp_path / "hot",
        cold_path=tmp_path / "cold",
        hot_retention_days=2,
        cold_retention_days=7,
        promote_to_cold_after_days=1,
    )
    archive = MicrostructureArchive(config)

    as_of = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    payload = [{"timestamp": "2025-01-01T12:00:00Z", "imbalance": 0.42}]
    snapshot_path = archive.archive("EUR/USD", payload, as_of=as_of)

    # Snapshot is written to hot storage with tier metadata.
    assert snapshot_path.exists()
    data = _read_json(snapshot_path)
    assert data["tier"] == "hot"
    assert data["record_count"] == 1
    assert data["records"][0]["imbalance"] == 0.42

    # After the promote window the snapshot moves to cold storage.
    report = archive.enforce_retention(reference=as_of + timedelta(days=2))
    assert report.promoted == 1
    assert report.hot_files == 0
    cold_path = (tmp_path / "cold" / "eur_usd")
    archived = list(cold_path.glob("*.json"))
    assert len(archived) == 1

    # Cold retention eventually purges the file.
    archive.enforce_retention(reference=as_of + timedelta(days=10))
    assert not any(cold_path.glob("*.json"))


def test_build_retention_guidance_exposes_cost_matrix_defaults(tmp_path: Path) -> None:
    config = MicrostructureArchiveConfig(
        hot_path=tmp_path / "hot",
        cold_path=tmp_path / "cold",
    )
    hot_guidance, cold_guidance = build_retention_guidance(config)

    assert hot_guidance.tier == "hot"
    assert "Tier-0" in hot_guidance.storage_class
    assert hot_guidance.retention_days == config.hot_retention_days
    assert "cost" in hot_guidance.notes.lower()

    assert cold_guidance.tier == "cold"
    assert "object storage" in cold_guidance.storage_class.lower()
    assert cold_guidance.retention_days == config.cold_retention_days
