from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.sentient.learning.real_time_learning_engine import (
    LearningSignal,
    LearningSignalType,
)
from src.sentient.memory.faiss_pattern_memory import FAISSPatternMemory
from src.sentient.sentient_predator import SentientPredator


def _memory_config(tmp_path: Path, dimension: int) -> dict[str, Any]:
    return {
        "vector_dimension": dimension,
        "index_path": str(tmp_path / "index.faiss"),
        "metadata_path": str(tmp_path / "metadata.json"),
    }


def test_store_extreme_episode_persists_metadata(tmp_path: Path) -> None:
    memory = FAISSPatternMemory(_memory_config(tmp_path, dimension=8))

    latent_summary = np.linspace(0.1, 0.8, 8, dtype=np.float32)
    metadata = {
        "episode_type": "flash_crash",
        "severity": 0.9,
        "evidence": {"volatility": 0.12},
    }

    entry_id = memory.store_extreme_episode(latent_summary, metadata)

    assert entry_id.startswith("memory_")

    recent = memory.get_recent_memories(1)
    assert recent, "expected recent memories to include the extreme episode"

    stored = recent[0]["metadata"]
    assert stored["episode_type"] == "flash_crash"
    assert stored["is_extreme_episode"] is True
    assert pytest.approx(0.9) == stored["severity"]
    assert "extreme_episode" in stored["tags"]


class _FakeMemory:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.experiences: list[dict[str, Any]] = []
        self.extreme_episodes: list[dict[str, Any]] = []

    def add_experience(self, vector: np.ndarray, metadata: Dict[str, Any]) -> str:
        self.experiences.append({"vector": vector.copy(), "metadata": dict(metadata)})
        return f"memory_{len(self.experiences) - 1}"

    def search_similar(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def store_extreme_episode(self, vector: np.ndarray, metadata: Dict[str, Any]) -> str:
        payload = dict(metadata)
        payload.setdefault("is_extreme_episode", True)
        tags = payload.get("tags", [])
        if isinstance(tags, str):
            tag_list = [tags]
        elif isinstance(tags, (list, tuple, set)):
            tag_list = [str(tag) for tag in tags]
        else:
            tag_list = []
        tag_list.extend(["extreme_episode", payload.get("episode_type", "")])
        payload["tags"] = sorted({tag for tag in tag_list if tag})

        self.extreme_episodes.append({"vector": vector.copy(), "metadata": payload})
        return f"extreme_{len(self.extreme_episodes) - 1}"

    def get_memory_stats(self) -> dict[str, Any]:
        return {
            "total_memories": len(self.experiences),
            "dimension": self.dimension,
            "memory_counter": len(self.experiences),
        }


class _FakeLearningEngine:
    def __init__(self, signal: LearningSignal) -> None:
        self._signal = signal

    async def process_closed_trade(self, _trade_data: Dict[str, Any]) -> LearningSignal:
        return self._signal

    def get_performance_summary(self) -> dict[str, Any]:
        return {"total_signals": 1}


class _FakeAdaptationController:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    async def generate_adaptations(
        self, *_args: Any, **_kwargs: Any
    ) -> list[Dict[str, Any]]:
        return []

    def get_adaptation_summary(self) -> dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_sentient_predator_records_extreme_episode(tmp_path: Path) -> None:
    config: dict[str, Any] = {
        "memory": _memory_config(tmp_path, dimension=12),
        "extreme_volatility_threshold": 0.08,
        "extreme_volume_threshold": 4.0,
        "extreme_momentum_threshold": 0.03,
        "extreme_order_flow_threshold": 0.015,
    }
    predator = SentientPredator(config)

    dimension = predator.memory.dimension
    predator.memory = _FakeMemory(dimension)

    signal = LearningSignal(
        trade_id="T-1",
        timestamp=datetime.utcnow(),
        signal_type=LearningSignalType.LOSS,
        context={"volatility": 0.12, "market_condition": "flash_crash"},
        outcome={
            "pnl": -2.5,
            "duration": 180.0,
            "max_drawdown": 0.35,
            "max_profit": 0.1,
        },
        features={
            "price_momentum": 0.04,
            "volume_ratio": 6.0,
            "volatility_ratio": 2.5,
            "liquidity_ratio": 0.5,
            "order_flow_imbalance": 0.02,
            "microstructure_score": 0.1,
        },
        metadata={"market_condition": "flash_crash", "strategy": "test"},
    )

    predator.learning_engine = _FakeLearningEngine(signal)
    predator.adaptation_controller = _FakeAdaptationController()
    predator.is_active = True

    trade_data = {
        "trade_id": "T-1",
        "close_time": datetime.utcnow().isoformat(),
        "pnl": -2.5,
        "duration": 180.0,
        "max_drawdown": 0.35,
        "max_profit": 0.1,
        "price_change": -0.05,
        "volume": 500000,
        "avg_volume": 50000,
        "entry_price": 1.2345,
        "spread": 0.0002,
        "volatility": 0.12,
        "order_imbalance": 0.02,
        "liquidity_depth": 100000,
        "recent_trades": [],
        "market_condition": "flash_crash",
    }

    await predator.process_closed_trade(trade_data)

    assert predator.memory.extreme_episodes, "expected extreme episode to be stored"
    episode_metadata = predator.memory.extreme_episodes[0]["metadata"]
    assert episode_metadata["episode_type"] == "flash_crash"
    assert episode_metadata["is_extreme_episode"] is True
    assert episode_metadata["severity"] > 0
