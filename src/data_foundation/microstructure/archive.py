"""Microstructure dataset archiving utilities with tiered retention.

The high-impact roadmap calls for tiered hot/cold storage of market
microstructure datasets so investigators can keep the freshest order-flow
analytics close to the runtime while archiving longer histories for
compliance and research.

This module implements a light-weight archive manager that writes JSON
snapshots into hot storage, promotes aging snapshots into a cold tier, and
cleans up files according to retention windows aligned with the cost matrix
in the EMP Encyclopedia.  The implementation intentionally avoids heavier
dependencies so it can run inside CI and developer environments without
additional services.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "MicrostructureArchiveConfig",
    "MicrostructureArchive",
    "MicrostructureRetentionReport",
    "MicrostructureRetentionGuidance",
    "build_retention_guidance",
]


_SYMBOL_SANITISE_RE = re.compile(r"[^0-9A-Za-z]+")
_TIMESTAMP_RE = re.compile(r"_(\d{8}T\d{6}Z)\.json$")


@dataclass(frozen=True, slots=True)
class MicrostructureArchiveConfig:
    """Configuration describing the hot/cold archive layout."""

    hot_path: Path
    cold_path: Path
    hot_retention_days: int = 14
    cold_retention_days: int = 365
    promote_to_cold_after_days: int = 3

    def ensure_directories(self) -> None:
        """Create the archive directories if they do not exist."""

        self.hot_path.mkdir(parents=True, exist_ok=True)
        self.cold_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class MicrostructureRetentionReport:
    """Summary of retention maintenance activity."""

    hot_files: int
    cold_files: int
    promoted: int
    deleted: int


@dataclass(frozen=True, slots=True)
class MicrostructureRetentionGuidance:
    """Documentation-friendly representation of retention expectations."""

    tier: str
    storage_class: str
    retention_days: int
    notes: str


class MicrostructureArchive:
    """Manage tiered hot/cold microstructure storage."""

    def __init__(
        self,
        config: MicrostructureArchiveConfig,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        config.ensure_directories()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def archive(
        self,
        symbol: str,
        records: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
        *,
        as_of: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Persist a microstructure snapshot into the hot tier.

        Parameters
        ----------
        symbol:
            Instrument identifier used to segregate archives.
        records:
            Sequence of microstructure datapoints (already serialisable to
            JSON).  The iterator is materialised to guarantee durability.
        as_of:
            Optional explicit timestamp.  Defaults to the archive clock.
        metadata:
            Additional metadata to persist alongside the snapshot.
        """

        if not symbol:
            raise ValueError("symbol must be a non-empty string")

        slug = _slugify_symbol(symbol)
        if not slug:
            raise ValueError("symbol slug resolved to an empty value")

        as_of_ts = (as_of or self._clock()).astimezone(UTC)
        records_list = list(records)

        snapshot_dir = self._config.hot_path / slug
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{slug}_{as_of_ts.strftime('%Y%m%dT%H%M%SZ')}.json"
        path = snapshot_dir / filename

        payload: MutableMapping[str, object] = {
            "symbol": symbol,
            "slug": slug,
            "as_of": as_of_ts.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z"),
            "tier": "hot",
            "retention": {
                "hot_days": self._config.hot_retention_days,
                "cold_days": self._config.cold_retention_days,
                "promote_after_days": self._config.promote_to_cold_after_days,
            },
            "record_count": len(records_list),
            "records": records_list,
        }
        if metadata:
            payload["metadata"] = dict(metadata)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

        return path

    def enforce_retention(
        self,
        *,
        reference: datetime | None = None,
    ) -> MicrostructureRetentionReport:
        """Promote hot snapshots to cold storage and purge expired files."""

        reference_ts = (reference or self._clock()).astimezone(UTC)
        promoted = self._promote_hot(reference_ts)
        deleted_hot = self._purge_expired(self._config.hot_path, self._config.hot_retention_days, reference_ts)
        deleted_cold = self._purge_expired(self._config.cold_path, self._config.cold_retention_days, reference_ts)
        hot_files = _count_snapshot_files(self._config.hot_path)
        cold_files = _count_snapshot_files(self._config.cold_path)
        return MicrostructureRetentionReport(
            hot_files=hot_files,
            cold_files=cold_files,
            promoted=promoted,
            deleted=deleted_hot + deleted_cold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _promote_hot(self, reference: datetime) -> int:
        promote_after = max(self._config.promote_to_cold_after_days, 0)
        if promote_after == 0 or self._config.cold_retention_days <= 0:
            return 0

        promotions = 0
        for path in _iter_snapshot_files(self._config.hot_path):
            timestamp = _extract_timestamp(path)
            if timestamp is None:
                continue
            age = reference - timestamp
            if age >= timedelta(days=promote_after):
                destination = self._config.cold_path / path.parent.name
                destination.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), destination / path.name)
                promotions += 1
        return promotions

    def _purge_expired(self, root: Path, retention_days: int, reference: datetime) -> int:
        if retention_days < 0:
            return 0

        deletions = 0
        for path in _iter_snapshot_files(root):
            timestamp = _extract_timestamp(path)
            if timestamp is None:
                continue
            if (reference - timestamp) >= timedelta(days=retention_days):
                path.unlink(missing_ok=True)
                deletions += 1
        return deletions


def _slugify_symbol(symbol: str) -> str:
    slug = _SYMBOL_SANITISE_RE.sub("_", symbol.strip().lower())
    slug = slug.strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug


def _iter_snapshot_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for entry in root.rglob("*.json"):
        if entry.is_file():
            yield entry


def _extract_timestamp(path: Path) -> datetime | None:
    match = _TIMESTAMP_RE.search(path.name)
    if not match:
        return None
    try:
        parsed = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC)


def _count_snapshot_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.json"))


def build_retention_guidance(
    config: MicrostructureArchiveConfig,
) -> tuple[MicrostructureRetentionGuidance, MicrostructureRetentionGuidance]:
    """Return human-readable retention guidance for docs and runbooks."""

    hot_guidance = MicrostructureRetentionGuidance(
        tier="hot",
        storage_class="Tier-0 SSD / DuckDB cache",
        retention_days=config.hot_retention_days,
        notes=(
            "Retain recent intraday order-flow snapshots for active analysis. "
            "Sized for cost-neutral local storage per the encyclopedia cost matrix."
        ),
    )
    cold_guidance = MicrostructureRetentionGuidance(
        tier="cold",
        storage_class="Tier-1 object storage (OCI/B2/S3)",
        retention_days=config.cold_retention_days,
        notes=(
            "Archive aggregated microstructure history for compliance and research. "
            "Leverages low-cost object storage with monthly roll-ups."
        ),
    )
    return hot_guidance, cold_guidance
