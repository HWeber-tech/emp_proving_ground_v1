from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.sentient.memory.faiss_pattern_memory import FAISSPatternMemory


def _memory_config(tmp_path: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "vector_dimension": 4,
        "index_path": str(tmp_path / "index.faiss"),
        "metadata_path": str(tmp_path / "metadata.json"),
        "backup_dir": str(tmp_path / "backups"),
        "decay_interval": 0,
    }
    config.update(overrides)
    return config


def _sample_vector(dimension: int, seed: float = 0.0) -> np.ndarray:
    base = np.linspace(0.1, 0.9, dimension, dtype=np.float32)
    return base + seed


def test_memory_backup_and_restore_roundtrip(tmp_path: Path) -> None:
    memory = FAISSPatternMemory(_memory_config(tmp_path))
    vector = _sample_vector(memory.dimension)
    metadata = {"learning_signal": {"pnl": 1.5, "signal_type": "profitable"}}

    memory_id = memory.add_experience(vector, metadata)
    stats_before = memory.get_memory_stats()

    backup_info = memory.create_backup(reason="pre-maintenance")
    backup_path = Path(backup_info["backup_path"])

    assert backup_path.exists()
    manifest_path = backup_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]["metadata"] == "metadata.json"
    assert manifest["reason"] == "pre-maintenance"

    memory.clear_memory()
    assert memory.get_memory_stats()["total_memories"] == 0

    restore_info = memory.restore_backup()
    assert Path(restore_info["restored_from"]) == backup_path

    stats_after = memory.get_memory_stats()
    assert stats_after["total_memories"] == stats_before["total_memories"]

    restored = memory.search_similar(vector, k=1)
    assert restored and restored[0]["memory_id"] == memory_id


def test_backup_retention_prunes_old_directories(tmp_path: Path) -> None:
    memory = FAISSPatternMemory(_memory_config(tmp_path, backup_retention=2))
    metadata = {"learning_signal": {"pnl": 1.0, "signal_type": "profitable"}}

    for idx in range(3):
        vector = _sample_vector(memory.dimension, seed=idx * 0.01)
        memory.add_experience(vector, metadata)
        memory.create_backup(reason=f"checkpoint-{idx}")

    backups = memory.list_backups()
    assert len(backups) == 2
    # Retention should keep the most recent backups.
    assert all(path.exists() for path in backups)
