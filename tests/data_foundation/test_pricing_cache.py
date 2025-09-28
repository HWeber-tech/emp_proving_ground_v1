from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_foundation.cache.pricing_cache import PricingCache
from src.data_foundation.pipelines.pricing_pipeline import (
    PricingPipelineConfig,
    PricingPipelineResult,
    PricingQualityIssue,
)


def _make_result() -> PricingPipelineResult:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
            "symbol": ["EURUSD=X", "EURUSD=X", "EURUSD=X"],
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "adj_close": [1.05, 1.15, 1.25],
            "volume": [100, 110, 120],
            "source": ["test"] * 3,
        }
    )
    issues = (
        PricingQualityIssue(
            code="duplicate_rows",
            severity="warning",
            message="Test warning",
            symbol="EURUSD=X",
        ),
    )
    return PricingPipelineResult(data=frame, issues=issues, metadata={"test": True})


def test_pricing_cache_store_creates_expected_files(tmp_path: Path) -> None:
    cache = PricingCache(tmp_path)
    config = PricingPipelineConfig(symbols=["EURUSD=X"], vendor="demo", interval="1d")
    result = _make_result()

    entry = cache.store(config, result)

    assert entry.dataset_path.exists()
    assert entry.metadata_path.exists()
    assert entry.issues_path.exists()

    metadata = json.loads(entry.metadata_path.read_text(encoding="utf-8"))
    assert metadata["symbols"] == ["EURUSD=X"]
    assert metadata["rows"] == len(result.data)

    issues_payload = json.loads(entry.issues_path.read_text(encoding="utf-8"))
    assert issues_payload["issues"][0]["code"] == "duplicate_rows"


def test_pricing_cache_prune_respects_retention(tmp_path: Path) -> None:
    cache = PricingCache(tmp_path)
    config = PricingPipelineConfig(symbols=["EURUSD=X"], vendor="demo", interval="1d")
    result = _make_result()

    old_entry = cache.store(config, result)
    metadata = json.loads(old_entry.metadata_path.read_text(encoding="utf-8"))
    metadata["created_at"] = "2000-01-01T00:00:00+00:00"
    old_entry.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    new_entry = cache.store(config, result)
    cache.prune(retention_days=1, max_entries=1)

    assert not old_entry.metadata_path.exists()
    assert not old_entry.dataset_path.exists()
    assert new_entry.metadata_path.exists()

