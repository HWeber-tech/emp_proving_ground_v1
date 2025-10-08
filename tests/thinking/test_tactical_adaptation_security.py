from __future__ import annotations
import sys
import types
from dataclasses import dataclass
from typing import Dict, List, Optional

import json

import pytest

fake_index = types.SimpleNamespace(ntotal=0)
fake_faiss = types.SimpleNamespace(
    Index=type("Index", (), {}),
    IndexFlatL2=lambda *_args, **_kwargs: fake_index,
    read_index=lambda *_args, **_kwargs: fake_index,
)
sys.modules.setdefault("faiss", fake_faiss)

from src.thinking.adaptation.tactical_adaptation_engine import TacticalAdaptationEngine


class _FakeStateStore:
    def __init__(self) -> None:
        self.storage: Dict[str, str] = {}
        self.set_calls: List[tuple[str, str, Optional[int]]] = []

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        self.storage[key] = value
        self.set_calls.append((key, value, expire))
        return True

    async def get(self, key: str) -> Optional[str]:
        return self.storage.get(key)

    async def delete(self, key: str) -> bool:  # pragma: no cover - not exercised
        return self.storage.pop(key, None) is not None

    async def keys(self, pattern: str) -> List[str]:  # pragma: no cover - not exercised
        return [key for key in self.storage if key.startswith(pattern.rstrip("*"))]

    async def clear(self) -> bool:  # pragma: no cover - not exercised
        self.storage.clear()
        return True


@dataclass
class _Adaptation:
    parameter_to_adjust: str
    adjustment_factor: float


@pytest.mark.asyncio
async def test_apply_adaptations_ignores_malicious_serialized_state() -> None:
    state_store = _FakeStateStore()
    pattern_memory = object()
    engine = TacticalAdaptationEngine(state_store, pattern_memory)  # type: ignore[arg-type]

    strategy_id = "alpha"
    malicious_blob = '__import__("os").system("echo owned")'
    await state_store.set(f"{engine._strategy_params_key}:{strategy_id}", malicious_blob)

    adaptation = _Adaptation(parameter_to_adjust="position_size_multiplier", adjustment_factor=1.5)

    result = await engine.apply_adaptations([adaptation], strategy_id)

    assert result is True
    stored_value = state_store.storage[f"{engine._strategy_params_key}:{strategy_id}"]
    parsed = json.loads(stored_value)
    assert pytest.approx(parsed["position_size_multiplier"], rel=1e-6) == 1.5
    # Confirm TTL applied for a week
    _, _, expire = state_store.set_calls[-1]
    assert expire == 86400 * 7
