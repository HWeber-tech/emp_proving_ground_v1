"""Utilities for archiving microstructure datasets into tiered storage."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


@dataclass(slots=True)
class DatasetPolicy:
    """Retention policy for a dataset managed by :class:`TieredDatasetArchiver`."""

    hot_retention_days: int
    cold_retention_days: Optional[int] = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.hot_retention_days <= 0:
            msg = "hot_retention_days must be positive"
            raise ValueError(msg)
        if (
            self.cold_retention_days is not None
            and self.cold_retention_days < self.hot_retention_days
        ):
            msg = "cold_retention_days must be greater than or equal to hot_retention_days"
            raise ValueError(msg)


@dataclass(slots=True)
class ArchiveResult:
    """Summary of an archive run."""

    dataset: str
    hot: List[str]
    cold: List[str]
    expired: List[str]
    missing_source: bool = False
    dry_run: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "hot": list(self.hot),
            "cold": list(self.cold),
            "expired": list(self.expired),
            "missing_source": self.missing_source,
            "dry_run": self.dry_run,
        }


class TieredDatasetArchiver:
    """Manage copying datasets into hot/cold tiered storage directories."""

    def __init__(
        self,
        *,
        metadata_dir: Path,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._metadata_dir = metadata_dir
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def archive_dataset(
        self,
        *,
        dataset: str,
        source_dir: Path,
        hot_dir: Path,
        policy: DatasetPolicy,
        cold_dir: Path | None = None,
        dry_run: bool = False,
    ) -> ArchiveResult:
        """Archive a dataset into hot/cold tiers according to ``policy``.

        Parameters
        ----------
        dataset:
            Logical name of the dataset (used for metadata file naming).
        source_dir:
            Directory containing the raw dataset files to archive.
        hot_dir:
            Destination directory for hot-tier storage.
        policy:
            Retention policy describing how long files stay in the hot tier and,
            optionally, the cold tier.
        cold_dir:
            Destination directory for cold-tier storage. Required when the policy
            defines ``cold_retention_days``.
        dry_run:
            When ``True`` the file system is not modified; the method only
            computes the intended actions.
        """

        if policy.cold_retention_days is not None and cold_dir is None:
            msg = "cold_dir must be provided when cold_retention_days is configured"
            raise ValueError(msg)

        if not source_dir.exists():
            result = ArchiveResult(dataset, [], [], [], missing_source=True, dry_run=dry_run)
            if not dry_run:
                self._write_metadata(result)
            return result

        now = self._clock()
        hot_files: List[str] = []
        cold_files: List[str] = []
        expired_files: List[str] = []

        if not dry_run:
            hot_dir.mkdir(parents=True, exist_ok=True)
            if cold_dir is not None:
                cold_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(p for p in source_dir.rglob("*") if p.is_file()):
            relative_path = path.relative_to(source_dir)
            file_age = now - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

            tier = self._determine_tier(file_age, policy)

            if tier == "hot":
                hot_files.append(str(relative_path))
                if not dry_run:
                    self._sync_file(path, hot_dir / relative_path)
                    if cold_dir is not None:
                        self._delete_if_exists(cold_dir / relative_path)
            elif tier == "cold":
                cold_files.append(str(relative_path))
                if not dry_run and cold_dir is not None:
                    self._sync_file(path, cold_dir / relative_path)
                    self._delete_if_exists(hot_dir / relative_path)
            else:
                expired_files.append(str(relative_path))
                if not dry_run:
                    self._delete_if_exists(hot_dir / relative_path)
                    if cold_dir is not None:
                        self._delete_if_exists(cold_dir / relative_path)

        result = ArchiveResult(
            dataset=dataset,
            hot=hot_files,
            cold=cold_files,
            expired=expired_files,
            missing_source=False,
            dry_run=dry_run,
        )

        if not dry_run:
            self._write_metadata(result)
        return result

    def _determine_tier(self, age: timedelta, policy: DatasetPolicy) -> str:
        hot_threshold = timedelta(days=policy.hot_retention_days)
        if age <= hot_threshold:
            return "hot"
        if policy.cold_retention_days is None:
            return "expired"
        cold_threshold = timedelta(days=policy.cold_retention_days)
        if age <= cold_threshold:
            return "cold"
        return "expired"

    def _sync_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def _delete_if_exists(self, path: Path) -> None:
        if path.exists():
            path.unlink()

    def _write_metadata(self, result: ArchiveResult) -> None:
        metadata_path = self._metadata_dir / f"{result.dataset}.json"
        payload = {
            **result.to_dict(),
            "generated_at": self._clock().isoformat(),
        }
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

