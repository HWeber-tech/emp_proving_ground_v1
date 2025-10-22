#!/usr/bin/env python3
"""
FAISSPatternMemory - Epic 1: The Predator's Instinct
Upgraded memory system for storing and recalling trading experiences.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

try:  # pragma: no cover - exercised via fallback behaviour in tests
    import faiss  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry in the FAISS index."""

    vector: np.ndarray
    metadata: dict[str, Any]
    timestamp: datetime
    learning_signal_id: str

    @property
    def features(self) -> dict[str, float]:
        """Get the features from metadata."""
        return cast(dict[str, float], self.metadata.get("features", {}))

    @property
    def outcome(self) -> dict[str, float]:
        """Get the outcome from metadata."""
        return cast(dict[str, float], self.metadata.get("outcome", {}))


class _BasePatternMemory:
    """Utility helpers shared between FAISS and fallback implementations."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dimension = int(config.get("vector_dimension", 64))
        self.index_path = Path(config.get("index_path", "data/memory/faiss_index"))
        self.metadata_path = Path(config.get("metadata_path", "data/memory/metadata.json"))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        backup_dir_default = self.metadata_path.parent / "backups"
        self.backup_dir = Path(config.get("backup_dir", backup_dir_default))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_retention = int(config.get("backup_retention", 5))
        self.memory_counter = 0
        self.metadata: dict[str, Any] = {}
        self.max_memories = int(config.get("max_memories", 10000))
        stale_seconds_cfg = config.get("stale_after_seconds")
        stale_days_cfg = config.get("stale_after_days")
        if stale_seconds_cfg is None and stale_days_cfg is not None:
            stale_seconds_cfg = float(stale_days_cfg) * 86400.0
        if stale_seconds_cfg is None:
            stale_seconds_cfg = 30.0 * 86400.0
        stale_seconds = max(float(stale_seconds_cfg), 0.0)
        self._stale_after = timedelta(seconds=stale_seconds)
        self.reinforcement_threshold = max(1, int(config.get("reinforcement_threshold", 3)))
        self.success_pnl_threshold = float(config.get("reinforcement_success_pnl", 0.0))
        signature_fields_cfg = config.get("reinforcement_signature_fields")
        if signature_fields_cfg is None:
            signature_fields_cfg = ["market_condition", "strategy"]
        self.reinforcement_signature_fields = [
            str(field) for field in signature_fields_cfg if field
        ]
        self.decay_interval = int(config.get("decay_interval", 50))
        if self.decay_interval < 0:
            self.decay_interval = 0

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = [ch.lower() for ch in value if ch.isalnum() or ch in {"-", "_"}]
        slug = "".join(cleaned).strip("-_")
        return slug[:40]

    def _should_run_decay(self) -> bool:
        return self.decay_interval > 0 and self.memory_counter % self.decay_interval == 0

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return datetime.utcnow()
        return datetime.utcnow()

    @staticmethod
    def _normalise_signature_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _extract_success_signature(
        self, metadata: Mapping[str, Any] | None
    ) -> tuple[bool, Optional[str]]:
        if not metadata:
            return False, None

        learning = metadata.get("learning_signal")
        if not isinstance(learning, Mapping):
            learning = {}

        outcome = metadata.get("outcome")
        if not isinstance(outcome, Mapping):
            outcome = {}

        pnl_value: Optional[float] = None
        for candidate in (
            learning.get("pnl"),
            metadata.get("pnl"),
            outcome.get("pnl"),
        ):
            if candidate is None:
                continue
            try:
                candidate_float = float(candidate)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate_float):
                pnl_value = candidate_float
                break

        is_success = pnl_value is not None and pnl_value > self.success_pnl_threshold

        signal_type = learning.get("signal_type") if isinstance(learning, Mapping) else None
        if isinstance(signal_type, str) and signal_type.lower() == "profitable":
            is_success = True

        if not is_success:
            return False, None

        context = metadata.get("context")
        if not isinstance(context, Mapping) and isinstance(learning, Mapping):
            context = learning.get("context")
        if not isinstance(context, Mapping):
            context = {}

        signature_payload: dict[str, Any] = {}
        if self.reinforcement_signature_fields:
            for field in self.reinforcement_signature_fields:
                if isinstance(context, Mapping) and field in context:
                    signature_payload[field] = self._normalise_signature_value(context[field])
        else:
            if isinstance(context, Mapping):
                for key, value in context.items():
                    signature_payload[str(key)] = self._normalise_signature_value(value)

        if not signature_payload:
            if isinstance(signal_type, str):
                signature_payload = {"signal_type": signal_type}
            else:
                signature_payload = {"group": "profitable"}

        try:
            signature = json.dumps(signature_payload, sort_keys=True, default=str)
        except TypeError:
            signature = json.dumps(str(signature_payload), sort_keys=True)
        return True, signature

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr

    def _pre_backup(self) -> None:
        """Hook executed before a backup is captured."""

    def _export_index_snapshot(self, destination: Path) -> Optional[str]:
        """Copy index artefacts into the backup folder if present."""

        if self.index_path.exists():
            target = destination / self.index_path.name
            shutil.copy2(self.index_path, target)
            return target.name
        return None

    def _restore_index_snapshot(self, source: Path) -> None:
        """Restore index artefacts from a backup folder when present."""

        snapshot = source / self.index_path.name
        if snapshot.exists():
            shutil.copy2(snapshot, self.index_path)

    def _post_restore(self) -> None:
        """Hook executed after metadata and index files are restored."""

    def _prune_backups(self) -> None:
        if self.backup_retention <= 0:
            return
        backups = sorted(
            [path for path in self.backup_dir.iterdir() if path.is_dir()]
        )
        if len(backups) <= self.backup_retention:
            return
        for path in backups[: len(backups) - self.backup_retention]:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:  # pragma: no cover - race safe
                continue

    def list_backups(self) -> list[Path]:
        """Return available backups sorted from oldest to newest."""

        backups = [path for path in self.backup_dir.iterdir() if path.is_dir()]
        backups.sort()
        return backups

    def create_backup(self, reason: Optional[str] = None) -> dict[str, Any]:
        """Persist the current memory state to a timestamped backup folder."""

        saver = getattr(self, "_save_metadata", None)
        if callable(saver):
            saver()

        self._pre_backup()

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        suffix = self._slugify(reason) if reason else ""
        folder_name = f"{timestamp}_{suffix}" if suffix else timestamp
        backup_path = self.backup_dir / folder_name
        counter = 1
        while backup_path.exists():
            backup_path = self.backup_dir / f"{folder_name}_{counter}"
            counter += 1

        backup_path.mkdir(parents=True, exist_ok=True)

        copied_files: dict[str, str] = {}
        if self.metadata_path.exists():
            target = backup_path / self.metadata_path.name
            shutil.copy2(self.metadata_path, target)
            copied_files["metadata"] = target.name

        index_snapshot = self._export_index_snapshot(backup_path)
        if index_snapshot:
            copied_files["index"] = index_snapshot

        stats = self.get_memory_stats()
        manifest = {
            "created_at": datetime.utcnow().isoformat(),
            "reason": reason,
            "files": copied_files,
            "memory_counter": self.memory_counter,
            "stats": stats,
        }
        with (backup_path / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, default=str, indent=2)

        self._prune_backups()

        return {"backup_path": str(backup_path), "files": copied_files, "stats": stats}

    def _resolve_backup_path(self, backup: Optional[str | Path]) -> Optional[Path]:
        if backup is not None:
            candidate = Path(backup)
            if candidate.is_dir():
                return candidate
            candidate = self.backup_dir / str(backup)
            if candidate.is_dir():
                return candidate
            return None

        backups = self.list_backups()
        return backups[-1] if backups else None

    def restore_backup(self, backup: Optional[str | Path] = None) -> dict[str, Any]:
        """Restore memory state from a previously created backup."""

        backup_path = self._resolve_backup_path(backup)
        if backup_path is None:
            raise FileNotFoundError("No backup available to restore")

        metadata_source = backup_path / self.metadata_path.name
        if metadata_source.exists():
            shutil.copy2(metadata_source, self.metadata_path)
        else:
            raise FileNotFoundError(
                f"Backup missing metadata file: {metadata_source.name}"
            )

        self._restore_index_snapshot(backup_path)

        loader = getattr(self, "_load_metadata", None)
        if callable(loader):
            loader()

        self._post_restore()

        manifest_path = backup_path / "manifest.json"
        manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)

        stats = self.get_memory_stats()
        return {
            "restored_from": str(backup_path),
            "manifest": manifest,
            "stats": stats,
        }

    def store_extreme_episode(
        self, latent_summary: np.ndarray, metadata: dict[str, Any]
    ) -> str:
        """Persist an extreme market episode in the memory index."""

        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")

        if "episode_type" not in metadata:
            raise ValueError("episode_type is required in metadata")

        prepared = dict(metadata)
        prepared.setdefault("captured_at", datetime.utcnow().isoformat())
        prepared.setdefault("episode_summary_version", 1)
        prepared["episode_type"] = str(prepared["episode_type"])
        severity_raw = prepared.get("severity", 0.0)
        try:
            severity_value = float(severity_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            severity_value = 0.0
        prepared["severity"] = max(0.0, min(1.0, severity_value))
        prepared["is_extreme_episode"] = True

        tags_obj = prepared.get("tags")
        tag_items: list[str]
        if isinstance(tags_obj, (list, tuple, set)):
            tag_items = [str(tag) for tag in tags_obj]
        elif isinstance(tags_obj, str):
            tag_items = [tags_obj]
        else:
            tag_items = []
        tag_items.extend(["extreme_episode", prepared["episode_type"]])
        prepared["tags"] = sorted({tag for tag in tag_items if tag})

        vector = np.asarray(latent_summary, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("latent_summary must be a 1D vector")

        logger.info(
            "Recording extreme episode in memory index",
            extra={
                "episode_type": prepared["episode_type"],
                "severity": prepared["severity"],
                "tags": prepared["tags"],
            },
        )

        return self.add_experience(vector, prepared)


if faiss is not None:

    class FAISSPatternMemory(_BasePatternMemory):
        """FAISS-backed implementation when the optional dependency is installed."""

        def __init__(self, config: dict[str, Any]):
            super().__init__(config)
            self.index: Optional[faiss.Index] = None
            self._initialize_index()
            self._load_metadata()

        def _pre_backup(self) -> None:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))

        def _restore_index_snapshot(self, source: Path) -> None:
            snapshot = source / self.index_path.name
            if snapshot.exists():
                shutil.copy2(snapshot, self.index_path)
                self._initialize_index()
            else:
                self.index = faiss.IndexFlatL2(self.dimension)

        def _post_restore(self) -> None:
            if self.index is None:
                self._initialize_index()

        def _initialize_index(self) -> None:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(
                    "Loaded FAISS index with %s entries", int(self.index.ntotal)
                )
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("Created new FAISS index")

        def _load_metadata(self) -> None:
            if not self.metadata_path.exists():
                return
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                self.metadata = json.load(handle)
            self.memory_counter = int(self.metadata.get("__counter__", 0))

        def _save_metadata(self) -> None:
            payload = dict(self.metadata)
            payload["__counter__"] = self.memory_counter
            with self.metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, default=str)

        def add_experience(self, vector: np.ndarray, metadata: dict[str, Any]) -> str:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} != {self.dimension}"
                )

            normalised = self._normalise(vector)
            if self.index is None:
                self._initialize_index()
            assert self.index is not None
            self.index.add(normalised.reshape(1, -1))

            memory_id = f"memory_{self.memory_counter}"
            self.memory_counter += 1
            timestamp = datetime.utcnow().isoformat()
            self.metadata[memory_id] = {
                "vector": normalised.tolist(),
                "metadata": dict(metadata),
                "timestamp": timestamp,
                "index_position": int(self.index.ntotal) - 1,
            }
            self._save_metadata()
            faiss.write_index(self.index, str(self.index_path))
            if self._should_run_decay():
                self.apply_decay_protocol()
            logger.info("Added experience to memory: %s", memory_id)
            return memory_id

        def search_similar(
            self, query_vector: np.ndarray, k: int = 10
        ) -> list[dict[str, Any]]:
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"Query vector dimension {len(query_vector)} != {self.dimension}"
                )

            if self.index is None:
                return []
            normalised = self._normalise(query_vector)
            distances, indices = self.index.search(normalised.reshape(1, -1), k)
            distances = cast(np.ndarray, distances)
            indices = cast(np.ndarray, indices)

            results: list[dict[str, Any]] = []
            for distance, idx in zip(distances[0], indices[0]):
                target = next(
                    (
                        mid
                        for mid, data in self.metadata.items()
                        if isinstance(data, dict)
                        and data.get("index_position") == int(idx)
                    ),
                    None,
                )
                if target is None:
                    continue
                stored = self.metadata[target]
                results.append(
                    {
                        "memory_id": target,
                        "distance": float(distance),
                        "metadata": stored["metadata"],
                        "timestamp": stored["timestamp"],
                    }
                )
            return results

        def get_memory_stats(self) -> dict[str, Any]:
            total = int(self.index.ntotal) if self.index is not None else 0
            return {
                "total_memories": total,
                "dimension": self.dimension,
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "memory_counter": self.memory_counter,
            }

        def clear_memory(self) -> None:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
            self.memory_counter = 0
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            logger.info("Cleared all memories")

        def get_recent_memories(self, count: int = 100) -> list[dict[str, Any]]:
            items = [
                (mid, data)
                for mid, data in self.metadata.items()
                if isinstance(data, dict)
            ]
            sorted_memories = sorted(
                items,
                key=lambda item: item[1].get("timestamp", ""),
                reverse=True,
            )
            return [
                {
                    "memory_id": mid,
                    "metadata": data.get("metadata", {}),
                    "timestamp": data.get("timestamp"),
                }
                for mid, data in sorted_memories[:count]
            ]

        def apply_decay_protocol(self) -> dict[str, Any]:
            now = datetime.utcnow()
            stale_cutoff: datetime | None
            if self._stale_after.total_seconds() > 0:
                stale_cutoff = now - self._stale_after
            else:
                stale_cutoff = None

            removed_ids: list[str] = []
            reinforced_ids: list[str] = []
            success_lookup: dict[str, str] = {}
            keep_entries: list[dict[str, Any]] = []
            changed = False

            for memory_id, record in list(self.metadata.items()):
                if not isinstance(record, dict):
                    removed_ids.append(memory_id)
                    changed = True
                    continue

                vector_raw = record.get("vector")
                if vector_raw is None:
                    removed_ids.append(memory_id)
                    changed = True
                    continue

                try:
                    vector_array = np.asarray(vector_raw, dtype=np.float32)
                except Exception:
                    removed_ids.append(memory_id)
                    changed = True
                    continue

                if vector_array.ndim != 1 or vector_array.shape[0] != self.dimension:
                    removed_ids.append(memory_id)
                    changed = True
                    continue

                timestamp_value = self._parse_timestamp(record.get("timestamp"))
                if stale_cutoff and timestamp_value < stale_cutoff:
                    removed_ids.append(memory_id)
                    changed = True
                    continue

                payload = record.get("metadata")
                if not isinstance(payload, dict):
                    payload = {}

                success, signature = self._extract_success_signature(payload)
                if success and signature is not None:
                    success_lookup[memory_id] = signature
                else:
                    if payload.get("reinforced"):
                        payload["reinforced"] = False
                        payload["reinforcement_count"] = int(
                            payload.get("reinforcement_count", 0)
                        )
                        payload.pop("last_reinforced_at", None)
                        changed = True

                record["metadata"] = payload
                record["timestamp"] = timestamp_value.isoformat()
                keep_entries.append(
                    {
                        "memory_id": memory_id,
                        "record": record,
                        "timestamp": timestamp_value,
                        "vector": vector_array,
                    }
                )

            if not keep_entries:
                if removed_ids:
                    self.metadata = {}
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self._save_metadata()
                    faiss.write_index(self.index, str(self.index_path))
                return {
                    "removed": len(removed_ids),
                    "reinforced": 0,
                    "removed_ids": removed_ids,
                    "reinforced_ids": reinforced_ids,
                    "remaining": 0,
                }

            if self.max_memories > 0 and len(keep_entries) > self.max_memories:
                keep_entries.sort(key=lambda entry: entry["timestamp"])
                overflow = len(keep_entries) - self.max_memories
                for entry in keep_entries[:overflow]:
                    removed_ids.append(entry["memory_id"])
                    success_lookup.pop(entry["memory_id"], None)
                keep_entries = keep_entries[overflow:]
                changed = True

            signature_counts = Counter(success_lookup.values())
            for entry in keep_entries:
                memory_id = entry["memory_id"]
                payload = entry["record"]["metadata"]
                signature = success_lookup.get(memory_id)
                if signature and signature_counts[signature] >= self.reinforcement_threshold:
                    payload["reinforced"] = True
                    payload["reinforcement_count"] = int(
                        payload.get("reinforcement_count", 0)
                    ) + 1
                    payload["last_reinforced_at"] = now.isoformat()
                    entry["timestamp"] = now
                    entry["record"]["timestamp"] = now.isoformat()
                    reinforced_ids.append(memory_id)
                    changed = True
                elif signature:
                    if payload.get("reinforced"):
                        payload["reinforced"] = False
                        payload["reinforcement_count"] = int(
                            payload.get("reinforcement_count", 0)
                        )
                        payload.pop("last_reinforced_at", None)
                        changed = True

            if not changed:
                return {
                    "removed": 0,
                    "reinforced": 0,
                    "removed_ids": [],
                    "reinforced_ids": [],
                    "remaining": len(self.metadata),
                }

            keep_entries.sort(key=lambda entry: entry["timestamp"], reverse=False)

            new_index = faiss.IndexFlatL2(self.dimension)
            new_metadata: dict[str, Any] = {}
            for position, entry in enumerate(keep_entries):
                vector_array = entry["vector"]
                new_index.add(vector_array.reshape(1, -1))
                record = entry["record"]
                record["index_position"] = position
                record["timestamp"] = entry["timestamp"].isoformat()
                new_metadata[entry["memory_id"]] = record

            self.metadata = new_metadata
            self.index = new_index
            self._save_metadata()
            faiss.write_index(self.index, str(self.index_path))

            return {
                "removed": len(removed_ids),
                "reinforced": len(reinforced_ids),
                "removed_ids": removed_ids,
                "reinforced_ids": reinforced_ids,
                "remaining": len(new_metadata),
            }


else:

    class FAISSPatternMemory(_BasePatternMemory):
        """In-memory fallback when the FAISS dependency is unavailable."""

        def __init__(self, config: dict[str, Any]):
            super().__init__(config)
            self._memories: dict[str, MemoryEntry] = {}
            self._load_metadata()
            logger.warning(
                "faiss module not available; using in-memory pattern memory fallback"
            )

        def _load_metadata(self) -> None:
            if not self.metadata_path.exists():
                return
            try:
                with self.metadata_path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except Exception:
                logger.warning("Failed to load fallback memory metadata; starting fresh")
                return

            self.memory_counter = int(raw.get("__counter__", 0))
            entries = raw.get("entries", {})
            self._memories = {}
            for memory_id, payload in entries.items():
                try:
                    entry = MemoryEntry(
                        vector=np.asarray(payload["vector"], dtype=np.float32),
                        metadata=cast(dict[str, Any], payload.get("metadata", {})),
                        timestamp=datetime.fromisoformat(payload["timestamp"]),
                        learning_signal_id=memory_id,
                    )
                except Exception:
                    continue
                self._memories[memory_id] = entry

        def _save_metadata(self) -> None:
            serialised = {
                memory_id: {
                    "vector": entry.vector.tolist(),
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp.isoformat(),
                }
                for memory_id, entry in self._memories.items()
            }
            payload = {"__counter__": self.memory_counter, "entries": serialised}
            with self.metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, default=str)

        def add_experience(self, vector: np.ndarray, metadata: dict[str, Any]) -> str:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} != {self.dimension}"
                )

            normalised = self._normalise(vector)
            memory_id = f"memory_{self.memory_counter}"
            self.memory_counter += 1
            entry = MemoryEntry(
                vector=normalised,
                metadata=dict(metadata),
                timestamp=datetime.utcnow(),
                learning_signal_id=memory_id,
            )
            self._memories[memory_id] = entry
            self._save_metadata()
            if self._should_run_decay():
                self.apply_decay_protocol()
            return memory_id

        def search_similar(
            self, query_vector: np.ndarray, k: int = 10
        ) -> list[dict[str, Any]]:
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"Query vector dimension {len(query_vector)} != {self.dimension}"
                )

            if not self._memories:
                return []

            normalised = self._normalise(query_vector)
            scored: list[tuple[float, MemoryEntry]] = []
            for entry in self._memories.values():
                distance = float(np.linalg.norm(entry.vector - normalised))
                scored.append((distance, entry))

            scored.sort(key=lambda item: item[0])
            results: list[dict[str, Any]] = []
            for distance, entry in scored[:k]:
                results.append(
                    {
                        "memory_id": entry.learning_signal_id,
                        "distance": distance,
                        "metadata": entry.metadata,
                        "timestamp": entry.timestamp.isoformat(),
                    }
                )
            return results

        def get_memory_stats(self) -> dict[str, Any]:
            return {
                "total_memories": len(self._memories),
                "dimension": self.dimension,
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "memory_counter": self.memory_counter,
            }

        def clear_memory(self) -> None:
            self._memories.clear()
            self.memory_counter = 0
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            logger.info("Cleared all fallback memories")

        def get_recent_memories(self, count: int = 100) -> list[dict[str, Any]]:
            entries = sorted(
                self._memories.values(),
                key=lambda entry: entry.timestamp,
                reverse=True,
            )
            return [
                {
                    "memory_id": entry.learning_signal_id,
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp.isoformat(),
                }
                for entry in entries[:count]
            ]

        def apply_decay_protocol(self) -> dict[str, Any]:
            now = datetime.utcnow()
            stale_cutoff = (
                now - self._stale_after if self._stale_after.total_seconds() > 0 else None
            )

            removed_ids: list[str] = []
            reinforced_ids: list[str] = []
            success_lookup: dict[str, str] = {}
            changed = False

            for memory_id, entry in list(self._memories.items()):
                if stale_cutoff and entry.timestamp < stale_cutoff:
                    del self._memories[memory_id]
                    removed_ids.append(memory_id)
                    success_lookup.pop(memory_id, None)
                    changed = True
                    continue

                success, signature = self._extract_success_signature(entry.metadata)
                if success and signature is not None:
                    success_lookup[memory_id] = signature
                else:
                    if entry.metadata.get("reinforced"):
                        entry.metadata["reinforced"] = False
                        entry.metadata["reinforcement_count"] = int(
                            entry.metadata.get("reinforcement_count", 0)
                        )
                        entry.metadata.pop("last_reinforced_at", None)
                        changed = True

            if self.max_memories > 0 and len(self._memories) > self.max_memories:
                sorted_entries = sorted(
                    self._memories.items(), key=lambda item: item[1].timestamp
                )
                overflow = len(sorted_entries) - self.max_memories
                for memory_id, _ in sorted_entries[:overflow]:
                    del self._memories[memory_id]
                    removed_ids.append(memory_id)
                    success_lookup.pop(memory_id, None)
                    changed = True

            signature_counts = Counter(success_lookup.values())
            for memory_id, signature in success_lookup.items():
                count = signature_counts[signature]
                entry = self._memories.get(memory_id)
                if entry is None:
                    continue
                if count >= self.reinforcement_threshold:
                    entry.metadata["reinforced"] = True
                    entry.metadata["reinforcement_count"] = int(
                        entry.metadata.get("reinforcement_count", 0)
                    ) + 1
                    entry.metadata["last_reinforced_at"] = now.isoformat()
                    entry.timestamp = now
                    reinforced_ids.append(memory_id)
                    changed = True
                elif entry.metadata.get("reinforced"):
                    entry.metadata["reinforced"] = False
                    entry.metadata["reinforcement_count"] = int(
                        entry.metadata.get("reinforcement_count", 0)
                    )
                    entry.metadata.pop("last_reinforced_at", None)
                    changed = True

            if changed:
                self._save_metadata()

            return {
                "removed": len(removed_ids),
                "reinforced": len(reinforced_ids),
                "removed_ids": removed_ids,
                "reinforced_ids": reinforced_ids,
                "remaining": len(self._memories),
            }
