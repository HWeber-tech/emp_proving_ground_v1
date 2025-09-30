"""Tiered storage helpers for microstructure datasets."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def _as_utc(value: datetime | None) -> datetime:
    """Normalise a datetime to UTC without timezone information."""

    if value is None:
        value = datetime.utcnow().replace(tzinfo=timezone.utc)
    elif value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value


@dataclass(slots=True, frozen=True)
class RetentionPolicy:
    """Retention windows for tiered storage."""

    hot_retention: timedelta
    cold_retention: timedelta

    @classmethod
    def from_days(cls, *, hot_days: int, cold_days: int) -> "RetentionPolicy":
        if hot_days <= 0:
            raise ValueError("hot_days must be positive")
        if cold_days <= 0:
            raise ValueError("cold_days must be positive")
        return cls(timedelta(days=hot_days), timedelta(days=cold_days))


@dataclass(slots=True)
class ArchiveMetadata:
    """Metadata written alongside archived datasets."""

    dataset: str
    tier: str
    as_of: datetime
    relative_path: str
    original_source: str
    size_bytes: int

    def to_dict(self, policy: RetentionPolicy) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "tier": self.tier,
            "as_of": self.as_of.isoformat(),
            "relative_path": self.relative_path,
            "original_source": self.original_source,
            "size_bytes": self.size_bytes,
            "hot_retention_days": policy.hot_retention.days,
            "cold_retention_days": policy.cold_retention.days,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ArchiveMetadata":
        as_of_raw = payload.get("as_of")
        if not isinstance(as_of_raw, str):
            raise ValueError("metadata missing as_of timestamp")
        return cls(
            dataset=str(payload.get("dataset")),
            tier=str(payload.get("tier", "hot")),
            as_of=datetime.fromisoformat(as_of_raw),
            relative_path=str(payload.get("relative_path")),
            original_source=str(payload.get("original_source", "")),
            size_bytes=int(payload.get("size_bytes", 0)),
        )


@dataclass(slots=True)
class ArchiveResult:
    """Result of archiving a dataset."""

    dataset: str
    destination: Path
    metadata_path: Path
    metadata: ArchiveMetadata


@dataclass(slots=True)
class RetentionEnforcementReport:
    """Summary of retention enforcement operations."""

    moved_to_cold: int
    deleted_from_cold: int
    missing_files: int


class MicrostructureTieredArchive:
    """Manage tiered storage for microstructure datasets."""

    def __init__(
        self,
        hot_directory: Path,
        cold_directory: Path,
        *,
        retention_policy: RetentionPolicy,
    ) -> None:
        self._hot_directory = Path(hot_directory).expanduser().resolve()
        self._cold_directory = Path(cold_directory).expanduser().resolve()
        self._retention_policy = retention_policy
        self._metadata_suffix = ".meta.json"
        self._hot_directory.mkdir(parents=True, exist_ok=True)
        self._cold_directory.mkdir(parents=True, exist_ok=True)

    @property
    def retention_policy(self) -> RetentionPolicy:
        return self._retention_policy

    def archive_file(
        self,
        source_path: Path,
        dataset: str,
        *,
        as_of: datetime | None = None,
    ) -> ArchiveResult:
        if not dataset or "/" in dataset or ".." in dataset:
            raise ValueError("dataset must be a simple name without path separators")

        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        as_of = _as_utc(as_of)
        relative_dir = Path(dataset) / as_of.strftime("%Y/%m/%d")
        destination_directory = self._hot_directory / relative_dir
        destination_directory.mkdir(parents=True, exist_ok=True)

        destination = destination_directory / source_path.name
        shutil.copy2(source_path, destination)
        size_bytes = destination.stat().st_size

        metadata = ArchiveMetadata(
            dataset=dataset,
            tier="hot",
            as_of=as_of.replace(tzinfo=None),
            relative_path=str(relative_dir / source_path.name),
            original_source=str(source_path),
            size_bytes=size_bytes,
        )
        metadata_path = destination.with_suffix(destination.suffix + self._metadata_suffix)
        metadata_path.write_text(
            json.dumps(metadata.to_dict(self._retention_policy), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return ArchiveResult(dataset, destination, metadata_path, metadata)

    def enforce_retention(self, *, now: datetime | None = None) -> RetentionEnforcementReport:
        now = _as_utc(now).replace(tzinfo=None)
        hot_cutoff = now - self._retention_policy.hot_retention
        cold_cutoff = now - self._retention_policy.cold_retention

        moved = 0
        deleted = 0
        missing = 0

        for metadata_path in self._metadata_files(self._hot_directory):
            metadata = self._load_metadata(metadata_path)
            if metadata is None:
                continue
            if metadata.as_of <= hot_cutoff:
                if self._move_to_cold(metadata, metadata_path):
                    moved += 1
                else:
                    missing += 1

        for metadata_path in self._metadata_files(self._cold_directory):
            metadata = self._load_metadata(metadata_path)
            if metadata is None:
                continue
            if metadata.as_of <= cold_cutoff:
                if self._delete_from_cold(metadata, metadata_path):
                    deleted += 1
                else:
                    missing += 1

        return RetentionEnforcementReport(moved_to_cold=moved, deleted_from_cold=deleted, missing_files=missing)

    def _metadata_files(self, root: Path) -> Iterable[Path]:
        suffix = self._metadata_suffix
        for path in root.rglob(f"*{suffix}"):
            if path.is_file():
                yield path

    def _load_metadata(self, path: Path) -> ArchiveMetadata | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = ArchiveMetadata.from_dict(payload)
            # Maintain awareness of storage tier transitions.
            if path.is_relative_to(self._cold_directory):
                metadata.tier = "cold"
            else:
                metadata.tier = "hot"
            return metadata
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("microstructure_archive_metadata_invalid", path=str(path))
            return None

    def _resolve_target(self, metadata: ArchiveMetadata, *, hot: bool) -> Path:
        base = self._hot_directory if hot else self._cold_directory
        return base / Path(metadata.relative_path)

    def _move_to_cold(self, metadata: ArchiveMetadata, metadata_path: Path) -> bool:
        source_file = self._resolve_target(metadata, hot=True)
        if not source_file.exists():
            logger.warning("microstructure_archive_missing_hot_file", path=str(source_file))
            metadata_path.unlink(missing_ok=True)
            return False

        destination = self._resolve_target(metadata, hot=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_file), str(destination))

        updated_metadata = metadata_path.read_text(encoding="utf-8")
        payload = json.loads(updated_metadata)
        payload["tier"] = "cold"
        payload["relative_path"] = metadata.relative_path
        payload["cold_storage_path"] = str(destination)
        metadata_path_cold = destination.with_suffix(destination.suffix + self._metadata_suffix)
        metadata_path_cold.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        metadata_path.unlink(missing_ok=True)
        return True

    def _delete_from_cold(self, metadata: ArchiveMetadata, metadata_path: Path) -> bool:
        target = self._resolve_target(metadata, hot=False)
        target.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return True

