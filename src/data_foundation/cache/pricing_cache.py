"""Utilities for persisting normalised pricing datasets in a local cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.data_foundation.pipelines.pricing_pipeline import (
    PricingPipelineConfig,
    PricingPipelineResult,
    PricingQualityIssue,
)

__all__ = ["PricingCache", "PricingCacheEntry"]


@dataclass(frozen=True, slots=True)
class PricingCacheEntry:
    """Describes a cached pricing dataset and supporting artefacts."""

    dataset_path: Path
    metadata_path: Path
    issues_path: Path
    created_at: datetime
    metadata: Mapping[str, object]
    issues_payload: Sequence[Mapping[str, object]]


class PricingCache:
    """Persist pricing pipeline results with retention-aware helpers."""

    def __init__(self, root: Path | str = Path("data_foundation/cache/pricing")) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def store(
        self,
        config: PricingPipelineConfig,
        result: PricingPipelineResult,
        *,
        issues: Sequence[PricingQualityIssue] | None = None,
        retention_days: int | None = None,
        max_entries: int | None = None,
    ) -> PricingCacheEntry:
        """Persist the result and return the created cache entry."""

        created_at = datetime.now(tz=UTC)
        key = json.dumps(
            {
                "symbols": config.normalised_symbols(),
                "vendor": config.vendor,
                "interval": config.interval,
                "start": config.window_start().isoformat(),
                "end": config.window_end().isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha1(key).hexdigest()[:12]

        timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
        base_name = f"{timestamp}_{config.vendor}_{config.interval}_{digest}"
        dataset_path = self._write_dataset(result.data, base_name)

        metadata_payload = {
            "created_at": created_at.isoformat(),
            "dataset_path": str(dataset_path),
            "symbols": config.normalised_symbols(),
            "vendor": config.vendor,
            "interval": config.interval,
            "window_start": config.window_start().isoformat(),
            "window_end": config.window_end().isoformat(),
            "rows": int(len(result.data)),
            "metadata": dict(result.metadata),
        }

        issues_payload = [
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "symbol": issue.symbol,
                "context": dict(issue.context),
            }
            for issue in (issues or result.issues)
        ]

        metadata_path = self._write_json(base_name + "_metadata.json", metadata_payload)
        issues_path = self._write_json(base_name + "_issues.json", {"issues": issues_payload})

        entry = PricingCacheEntry(
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            issues_path=issues_path,
            created_at=created_at,
            metadata=metadata_payload,
            issues_payload=issues_payload,
        )

        if retention_days is not None or max_entries is not None:
            self.prune(retention_days=retention_days, max_entries=max_entries)

        return entry

    # ------------------------------------------------------------------
    def list_entries(self) -> list[PricingCacheEntry]:
        """Return cache entries discovered on disk sorted by creation time."""

        entries: list[PricingCacheEntry] = []
        for metadata_file in sorted(self._root.glob("*_metadata.json")):
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(payload["created_at"])
            dataset_path = Path(payload["dataset_path"])
            issues_path = metadata_file.with_name(metadata_file.name.replace("_metadata", "_issues"))
            issues_payload = json.loads(issues_path.read_text(encoding="utf-8"))["issues"]
            entries.append(
                PricingCacheEntry(
                    dataset_path=dataset_path,
                    metadata_path=metadata_file,
                    issues_path=issues_path,
                    created_at=created_at,
                    metadata=payload,
                    issues_payload=issues_payload,
                )
            )

        entries.sort(key=lambda item: item.created_at, reverse=True)
        return entries

    # ------------------------------------------------------------------
    def prune(
        self,
        *,
        retention_days: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        """Apply retention and max-entry policies to the cache directory."""

        entries = self.list_entries()
        now = datetime.now(tz=UTC)

        to_delete: list[PricingCacheEntry] = []
        if retention_days is not None:
            cutoff = now - timedelta(days=retention_days)
            to_delete.extend(entry for entry in entries if entry.created_at < cutoff)

        if max_entries is not None and len(entries) > max_entries:
            to_delete.extend(entries[max_entries:])

        seen = set()
        for entry in to_delete:
            if entry.metadata_path in seen:
                continue
            seen.add(entry.metadata_path)
            for path in (
                entry.metadata_path,
                entry.issues_path,
                entry.dataset_path,
            ):
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    continue

    # ------------------------------------------------------------------
    def _write_dataset(self, frame: pd.DataFrame, base_name: str) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        parquet_path = self._root / f"{base_name}.parquet"
        try:
            frame.to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception:
            csv_path = parquet_path.with_suffix(".csv")
            frame.to_csv(csv_path, index=False)
            return csv_path

    def _write_json(self, file_name: str, payload: Mapping[str, object]) -> Path:
        path = self._root / file_name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

