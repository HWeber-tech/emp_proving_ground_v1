from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.data_foundation.cache.pricing_cache import PricingCache, PricingCacheEntry


def test_pricing_cache_prune_logs_when_delete_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = PricingCache(tmp_path)
    entry = PricingCacheEntry(
        dataset_path=tmp_path / "dataset.parquet",
        metadata_path=tmp_path / "entry_metadata.json",
        issues_path=tmp_path / "entry_issues.json",
        created_at=datetime.now(tz=UTC) - timedelta(days=10),
        metadata={},
        issues_payload=[],
    )

    for path in (entry.dataset_path, entry.metadata_path, entry.issues_path):
        path.touch()

    original_unlink = Path.unlink

    def fake_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == entry.metadata_path:
            raise PermissionError("denied")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fake_unlink)
    monkeypatch.setattr(cache, "list_entries", lambda: [entry])

    caplog.set_level(logging.WARNING)
    cache.prune(retention_days=1)

    assert any("Failed to remove pricing cache artefact" in record.message for record in caplog.records)
