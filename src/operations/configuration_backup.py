"""Configuration backup packaging utilities for operational runbooks.

This module fulfils the "Build configuration backup" task enumerated in the
README by providing a concrete implementation that packages configuration
assets into a deterministic archive alongside a machine-readable manifest. The
resulting artifact complements the existing backup readiness telemetry by
capturing the actual files operators would restore from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = (
    "*.yaml",
    "*.yml",
    "*.json",
    "*.toml",
    "*.ini",
    "*.cfg",
    "*.conf",
    "*.env",
    "*.txt",
)
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "*.bak",
    "*.swp",
    "__pycache__",
)
_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class ConfigurationBackupSource:
    """A source tree or file that should be captured in a configuration backup."""

    name: str
    path: Path | str
    include_patterns: tuple[str, ...] = DEFAULT_INCLUDE_PATTERNS
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS
    include_dotfiles: bool = True

    def __post_init__(self) -> None:  # pragma: no cover - trivial dataclass plumbing
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "include_patterns", tuple(self.include_patterns))
        object.__setattr__(self, "exclude_patterns", tuple(self.exclude_patterns))


@dataclass(frozen=True)
class ConfigurationBackupEntry:
    """Metadata describing a single file captured in a configuration backup."""

    source: str
    relative_path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ConfigurationBackupManifest:
    """Structured manifest produced alongside a configuration backup archive."""

    generated_at: datetime
    archive_path: Path
    entries: tuple[ConfigurationBackupEntry, ...] = field(default_factory=tuple)
    missing_sources: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "archive_path": str(self.archive_path),
            "entries": [entry.as_dict() for entry in self.entries],
            "missing_sources": list(self.missing_sources),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConfigurationBackupResult:
    """Outcome of building a configuration backup."""

    archive_path: Path
    manifest_path: Path
    manifest: ConfigurationBackupManifest


@dataclass(frozen=True)
class ConfigurationBackupPlan:
    """Declarative configuration backup plan."""

    sources: Sequence[ConfigurationBackupSource]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - trivial dataclass plumbing
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "metadata", dict(self.metadata))


def default_configuration_backup_plan(root: Path | str | None = None) -> ConfigurationBackupPlan:
    """Build a default configuration backup plan rooted at ``root``.

    The default plan captures the ``config`` directory, ``env_templates`` and the
    root ``config.yaml`` file when they are present.  Consumers can extend or
    override this plan to suit their environment.
    """

    base = Path(root) if root is not None else Path.cwd()
    sources: list[ConfigurationBackupSource] = []

    config_dir = base / "config"
    if config_dir.exists():
        sources.append(ConfigurationBackupSource(name="config", path=config_dir))

    env_templates_dir = base / "env_templates"
    if env_templates_dir.exists():
        sources.append(
            ConfigurationBackupSource(name="env_templates", path=env_templates_dir)
        )

    for filename in ("config.yaml", "config.yml", "config.json"):
        config_file = base / filename
        if config_file.exists():
            sources.append(
                ConfigurationBackupSource(
                    name=filename,
                    path=config_file,
                    include_patterns=("*",),
                    include_dotfiles=True,
                )
            )
            break

    return ConfigurationBackupPlan(sources=sources, metadata={"root": str(base)})


def build_configuration_backup(
    plan: ConfigurationBackupPlan,
    destination: Path | str,
    *,
    now: datetime | None = None,
    compression: int = ZIP_DEFLATED,
) -> ConfigurationBackupResult:
    """Create an archive containing configuration sources defined in ``plan``."""

    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    moment = now or datetime.now(tz=UTC)
    timestamp = moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_path = dest_path / f"configuration-backup-{timestamp}.zip"

    entries: list[ConfigurationBackupEntry] = []
    files_to_archive: list[tuple[str, Path]] = []
    missing_sources: list[str] = []

    for source in plan.sources:
        base_path = Path(source.path)
        if not base_path.exists():
            missing_sources.append(source.name)
            continue

        for relative_path, actual_path in _iter_source_files(source, base_path):
            checksum = _compute_sha256(actual_path)
            size_bytes = actual_path.stat().st_size
            entry = ConfigurationBackupEntry(
                source=source.name,
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=checksum,
            )
            entries.append(entry)
            files_to_archive.append((f"{source.name}/{relative_path}", actual_path))

    entries.sort(key=lambda item: (item.source, item.relative_path))
    files_to_archive.sort(key=lambda item: item[0])

    manifest = ConfigurationBackupManifest(
        generated_at=moment,
        archive_path=archive_path,
        entries=tuple(entries),
        missing_sources=tuple(missing_sources),
        metadata=dict(plan.metadata),
    )

    manifest_payload = json.dumps(manifest.as_dict(), indent=2, sort_keys=True)

    with ZipFile(archive_path, mode="w", compression=compression) as zip_file:
        for arcname, actual_path in files_to_archive:
            zip_file.write(actual_path, arcname=arcname)
        zip_file.writestr(_MANIFEST_FILENAME, manifest_payload)

    manifest_path = archive_path.with_suffix(".manifest.json")
    manifest_path.write_text(manifest_payload + "\n", encoding="utf-8")

    return ConfigurationBackupResult(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _iter_source_files(
    source: ConfigurationBackupSource, base_path: Path
) -> Iterable[tuple[str, Path]]:
    if base_path.is_file():
        if _should_include(Path(base_path.name), source):
            yield base_path.name, base_path
        return

    candidates = sorted(base_path.rglob("*"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        relative_path = candidate.relative_to(base_path)
        if _should_include(relative_path, source):
            yield relative_path.as_posix(), candidate


def _should_include(path: Path, source: ConfigurationBackupSource) -> bool:
    parts = path.parts
    if not source.include_dotfiles and any(part.startswith(".") for part in parts):
        return False
    posix_path = path.as_posix()
    for pattern in source.exclude_patterns:
        if fnmatch(posix_path, pattern) or pattern in parts:
            return False
    if not source.include_patterns:
        return True
    return any(fnmatch(posix_path, pattern) for pattern in source.include_patterns)


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ConfigurationBackupEntry",
    "ConfigurationBackupManifest",
    "ConfigurationBackupPlan",
    "ConfigurationBackupResult",
    "ConfigurationBackupSource",
    "build_configuration_backup",
    "default_configuration_backup_plan",
]
