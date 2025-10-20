from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.sentient.memory.faiss_pattern_memory import FAISSPatternMemory


def _memory_config(tmp_path: Path, overrides: Dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "vector_dimension": 6,
        "index_path": str(tmp_path / "index.faiss"),
        "metadata_path": str(tmp_path / "metadata.json"),
        "stale_after_seconds": 3600,
        "decay_interval": 0,
        "reinforcement_threshold": 2,
    }
    if overrides:
        base.update(overrides)
    return base


def _update_timestamp(memory: FAISSPatternMemory, memory_id: str, timestamp: datetime) -> None:
    if hasattr(memory, "_memories"):
        entry = memory._memories[memory_id]
        entry.timestamp = timestamp
    else:
        record = memory.metadata[memory_id]
        record["timestamp"] = timestamp.isoformat()


def _get_metadata(memory: FAISSPatternMemory, memory_id: str) -> dict[str, Any]:
    if hasattr(memory, "_memories"):
        return memory._memories[memory_id].metadata
    record = memory.metadata[memory_id]
    return record.get("metadata", {})


def test_decay_protocol_prunes_stale_memories(tmp_path: Path) -> None:
    memory = FAISSPatternMemory(_memory_config(tmp_path, {"stale_after_seconds": 60}))

    base_vector = np.linspace(0.1, 0.6, memory.dimension, dtype=np.float32)
    ids: list[str] = []
    for idx in range(3):
        metadata = {
            "learning_signal": {
                "pnl": float(idx),
                "signal_type": "profitable" if idx > 0 else "loss",
                "context": {"market_condition": "neutral"},
            },
            "context": {"market_condition": "neutral"},
            "outcome": {"pnl": float(idx)},
        }
        memory_id = memory.add_experience(base_vector + idx * 0.01, metadata)
        ids.append(memory_id)

    stale_time = datetime.utcnow() - timedelta(minutes=10)
    _update_timestamp(memory, ids[0], stale_time)
    _update_timestamp(memory, ids[1], stale_time)

    summary = memory.apply_decay_protocol()

    assert summary["removed"] == 2
    assert summary["remaining"] == 1

    stats = memory.get_memory_stats()
    assert stats["total_memories"] == 1
    assert set(summary["removed_ids"]) == {ids[0], ids[1]}


def test_decay_protocol_reinforces_successful_groups(tmp_path: Path) -> None:
    memory = FAISSPatternMemory(
        _memory_config(
            tmp_path,
            {
                "stale_after_seconds": 3600,
                "reinforcement_threshold": 2,
            },
        )
    )

    vector = np.linspace(0.05, 0.35, memory.dimension, dtype=np.float32)
    successful_ids: list[str] = []
    for idx in range(3):
        metadata = {
            "learning_signal": {
                "pnl": 1.0 + idx,
                "signal_type": "profitable",
                "context": {"market_condition": "bull", "strategy": "alpha"},
            },
            "context": {"market_condition": "bull", "strategy": "alpha"},
            "outcome": {"pnl": 1.0 + idx},
        }
        memory_id = memory.add_experience(vector + idx * 0.01, metadata)
        successful_ids.append(memory_id)

    failure_metadata = {
        "learning_signal": {
            "pnl": -0.5,
            "signal_type": "loss",
            "context": {"market_condition": "bear", "strategy": "beta"},
        },
        "context": {"market_condition": "bear", "strategy": "beta"},
        "outcome": {"pnl": -0.5},
    }
    failure_id = memory.add_experience(vector + 0.5, failure_metadata)

    before = datetime.utcnow()
    summary = memory.apply_decay_protocol()

    assert set(summary["reinforced_ids"]) == set(successful_ids)
    assert summary["reinforced"] == len(successful_ids)

    for memory_id in successful_ids:
        stored = _get_metadata(memory, memory_id)
        assert stored.get("reinforced") is True
        assert int(stored.get("reinforcement_count", 0)) >= 1
        reinforced_at = datetime.fromisoformat(stored["last_reinforced_at"])
        assert reinforced_at >= before

    failure_stored = _get_metadata(memory, failure_id)
    assert not failure_stored.get("reinforced", False)
